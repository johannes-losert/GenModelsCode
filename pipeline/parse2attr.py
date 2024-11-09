import numpy as np
import json
import pickle
import logging
from datetime import timedelta, datetime
from pyscipopt import Model, quicksum
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel
from typing import List, Dict, Union
import nltk

from pydantic import ValidationError
from pipeline.schemas import (
    Issuer, RewardUnit, Benefit, CreditNeeded, PurchaseCategory, APRType, CreditCardKeyword,
    RewardCategoryThreshold, RewardCategoryRelation, ConditionalSignOnBonus, APR, AnnualFee,
    CreditCardSchema, get_issuer, multiple_nearest, get_primary_reward_unit
)
from pipeline.schemas import CRSchema, CKSchema

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

nltk.download('punkt')

# Example sentences for each attribute to enhance embeddings
attribute_examples = {
    "Annual Fee": [
        "This card has a $95 annual fee, waived for the first year.",
        "Enjoy this card with no annual fee for life.",
        "Annual fee of $50, which is charged monthly at $4.16."
    ],
    "Sign-on Bonus": [
        "Earn a sign-on bonus of 20,000 points after spending $1,000 in the first three months.",
        "Receive a welcome bonus of $200 once you spend $500 in the first three months.",
        "Get 50,000 bonus miles if you spend $2,000 within the first 90 days."
    ],
    "Rewards": [
        "Earn 2% cashback on all purchases.",
        "Get 5 points per dollar spent on travel purchases.",
        "Enjoy 3% cash back on groceries and dining."
    ],
    "APR": [
        "This card has a 0% introductory APR for the first 12 months.",
        "Enjoy a 15.99% variable APR based on your creditworthiness.",
        "APR is 18.24% for balance transfers and purchases."
    ],
    "Benefits": [
        "Includes travel insurance and purchase protection.",
        "Gain access to airport lounges worldwide.",
        "Enjoy complimentary cell phone protection."
    ],
    "Credit Needed": [
        "This card is available to individuals with excellent credit.",
        "Applicants need good to excellent credit for approval.",
        "Designed for people with fair to good credit."
    ]
}

def load_data_from_pickle(pickle_path: str) -> Dict[str, Union[CRSchema, CKSchema]]:
    """Load and return data from a pickle file, handling both CK and CRSchema formats."""
    logger.info(f"Loading data from pickle file: {pickle_path}")
    
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)

    valid_data = {}
    for name, card_data in data.items():
        try:
            if "issuer" in card_data and "attributes" in card_data:
                card = CKSchema(
                    name=name,
                    issuer=card_data.get("issuer", "Unknown"),
                    attributes=card_data.get("attributes", "")
                )
            elif "unparsed_issuer" in card_data and "unparsed_credit_needed" in card_data and "unparsed_card_attributes" in card_data:
                card = CRSchema(
                    name=name,
                    unparsed_issuer=card_data.get("unparsed_issuer", "Unknown"),
                    unparsed_credit_needed=card_data.get("unparsed_credit_needed", "Unknown"),
                    unparsed_card_attributes=card_data.get("unparsed_card_attributes", "")
                )
            else:
                logger.warning(f"Skipping entry '{name}' due to unrecognized format.")
                continue

            valid_data[name] = card

        except ValidationError as e:
            logger.warning(f"Skipping entry '{name}' due to validation error: {e}")

    logger.info(f"Loaded {len(valid_data)} valid entries from {pickle_path}")
    return valid_data

def initialize_nlp_models():
    """Initialize tokenizer and BERT models."""
    logger.info("Initializing BERT model and tokenizer...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    logger.info("BERT model and tokenizer initialized successfully.")
    return tokenizer, bert_model

def parse_text_to_sentences(text: str) -> List[str]:
    """Parse text into sentences using nltk."""
    sentences = nltk.sent_tokenize(text)
    logger.info(f"Parsed text into {len(sentences)} sentences.")
    return sentences

def generate_sentence_embeddings(sentences: List[str], tokenizer, bert_model) -> List[np.ndarray]:
    """Generate embeddings for each sentence."""
    logger.info("Generating embeddings for sentences...")
    embeddings = [
        bert_model(**tokenizer(sentence, return_tensors='pt')).last_hidden_state.mean(dim=1).detach().numpy()
        for sentence in sentences
    ]
    logger.info(f"Generated {len(embeddings)} embeddings for sentences.")
    return embeddings

def compute_attribute_embeddings(attributes: List[str], attribute_examples: Dict[str, List[str]], tokenizer, bert_model) -> List[np.ndarray]:
    """Compute enhanced embeddings for each attribute by including example sentences."""
    logger.info("Computing enhanced embeddings for attributes using example sentences...")
    enhanced_embeddings = []
    for attribute in attributes:
        attribute_name_inputs = tokenizer(attribute, return_tensors='pt')
        attribute_name_outputs = bert_model(**attribute_name_inputs)
        attribute_name_embedding = attribute_name_outputs.last_hidden_state.mean(dim=1).detach().numpy()

        example_embeddings = []
        for example in attribute_examples.get(attribute, []):
            example_inputs = tokenizer(example, return_tensors='pt')
            example_outputs = bert_model(**example_inputs)
            example_embedding = example_outputs.last_hidden_state.mean(dim=1).detach().numpy()
            example_embeddings.append(example_embedding)

        if example_embeddings:
            average_example_embedding = np.mean(example_embeddings, axis=0)
            enhanced_embedding = (attribute_name_embedding + average_example_embedding) / 2
        else:
            enhanced_embedding = attribute_name_embedding

        enhanced_embeddings.append(enhanced_embedding)

    logger.info("Enhanced attribute embeddings computed.")
    return enhanced_embeddings

def compute_similarities(node_embeddings: List[np.ndarray], attribute_embeddings: List[np.ndarray]) -> np.ndarray:
    """Compute similarities between nodes and attributes."""
    logger.info("Computing similarities between sentences and attributes...")
    similarities = np.array([
        [cosine_similarity(node, attr)[0][0] for attr in attribute_embeddings] for node in node_embeddings
    ])
    logger.info("Similarity matrix computed.")
    return similarities

def formulate_and_solve_bqp(similarities: np.ndarray) -> Dict[int, int]:
    """Formulate and solve the BQP problem using SCIP."""
    logger.info("Formulating and solving the BQP problem...")
    model_scip = Model("AttributeAssignment")
    N, M = similarities.shape
    x = {(i, j): model_scip.addVar(vtype='B', name=f"x_{i}_{j}") for i in range(N) for j in range(M)}

    model_scip.setObjective(quicksum(similarities[i, j] * x[i, j] for i in range(N) for j in range(M)), 'maximize')
    for i in range(N):
        model_scip.addCons(quicksum(x[i, j] for j in range(M)) <= 1)
    
    model_scip.optimize()
    
    assignments = {i: j for i in range(N) for j in range(M) if model_scip.getVal(x[i, j]) > 0.5}
    logger.info(f"BQP problem solved with {len(assignments)} assignments.")
    return assignments

def extract_attribute_texts(assignments: Dict[int, int], node_texts: List[str], attributes: List[str]) -> Dict[str, List[str]]:
    """Extract attribute texts based on assignments."""
    attribute_texts = {attr: [] for attr in attributes}
    for i, j in assignments.items():
        attribute_texts[attributes[j]].append(node_texts[i])
    logger.info(f"Extracted following attribute text based on assignments for each card.")
    return attribute_texts

def process_pickle_file(pickle_path: str, tokenizer, bert_model) -> List[Dict[str, List[str]]]:
    """Process each credit card entry in the pickle file and return attribute text mappings."""
    data = load_data_from_pickle(pickle_path)
    attributes = ['Annual Fee', 'Sign-on Bonus', 'Rewards', 'APR', 'Benefits', 'Credit Needed']
    attribute_embeddings = compute_attribute_embeddings(attributes, attribute_examples, tokenizer, bert_model)
    all_attribute_texts = []

    for card_name, card_data in data.items():
        logger.info(f"Processing card: {card_name}")
        text_to_parse = card_data.unparsed_card_attributes if isinstance(card_data, CRSchema) else card_data.attributes
        sentences = parse_text_to_sentences(text_to_parse)
        node_embeddings = generate_sentence_embeddings(sentences, tokenizer, bert_model)
        similarities = compute_similarities(node_embeddings, attribute_embeddings)
        assignments = formulate_and_solve_bqp(similarities)
        attribute_texts = extract_attribute_texts(assignments, sentences, attributes)
        all_attribute_texts.append({card_name: attribute_texts})

    logger.info(f"Processed {len(all_attribute_texts)} cards from {pickle_path}.")
    return all_attribute_texts

def main():
    tokenizer, bert_model = initialize_nlp_models()
    pickle_files = [
        "../extraction_pkl_out/20241109_174232_credit_karma_dict.pkl",
        "../extraction_pkl_out/20241109_002033_cardratings_dict.pkl"
    ]
    
    all_attr_maps = []
    for file_path in pickle_files:
        logger.info(f"Starting processing for file: {file_path}")
        all_attr_maps.extend(process_pickle_file(file_path, tokenizer, bert_model))
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"processed_credit_cards_{timestamp}.json"
    
    with open(output_filename, 'w') as outfile:
        json.dump(all_attr_maps, outfile, indent=4, default=str)
    
    logger.info(f"Processing complete. Results saved to {output_filename}")

if __name__ == "__main__":
    main()
