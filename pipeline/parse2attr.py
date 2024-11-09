# main.py

import numpy as np
from datetime import timedelta
from pyscipopt import Model, quicksum
from sklearn.metrics.pairwise import cosine_similarity
from stanfordcorenlp import StanfordCoreNLP
from transformers import BertTokenizer, BertModel
from typing import List, Dict

from pydantic import ValidationError

from pipeline.schemas import (
    Issuer, RewardUnit, Benefit, CreditNeeded, PurchaseCategory, APRType, CreditCardKeyword,
    RewardCategoryThreshold, RewardCategoryRelation, ConditionalSignOnBonus, APR, AnnualFee,
    CreditCardSchema, get_issuer, multiple_nearest, get_primary_reward_unit
)


def initialize_nlp_models():
    """Initialize Stanford CoreNLP and BERT models."""
    # Initialize Stanford CoreNLP
    nlp = StanfordCoreNLP('http://localhost', port=9000)
    
    # Load BERT model and tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    
    return nlp, tokenizer, bert_model


def parse_text_to_sentences(text: str, nlp: StanfordCoreNLP) -> List[str]:
    """Parse text into sentences."""
    sentences = nlp.sent_tokenize(text)
    return sentences


def generate_sentence_embeddings(sentences: List[str], tokenizer, bert_model) -> List[np.ndarray]:
    """Generate embeddings for each sentence."""
    node_embeddings = []
    for sentence in sentences:
        inputs = tokenizer(sentence, return_tensors='pt')
        outputs = bert_model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).detach().numpy()
        node_embeddings.append(embedding)
    return node_embeddings


def compute_attribute_embeddings(attributes: List[str], tokenizer, bert_model) -> List[np.ndarray]:
    """Compute embeddings for each attribute."""
    attribute_embeddings = []
    for attr in attributes:
        inputs = tokenizer(attr, return_tensors='pt')
        outputs = bert_model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).detach().numpy()
        attribute_embeddings.append(embedding)
    return attribute_embeddings


def compute_similarities(node_embeddings: List[np.ndarray], attribute_embeddings: List[np.ndarray]) -> np.ndarray:
    """Compute similarities between nodes and attributes."""
    N = len(node_embeddings)
    M = len(attribute_embeddings)
    similarities = np.zeros((N, M))
    for i, node_emb in enumerate(node_embeddings):
        for j, attr_emb in enumerate(attribute_embeddings):
            sim = cosine_similarity(node_emb, attr_emb)
            similarities[i, j] = sim
    return similarities


def formulate_and_solve_bqp(similarities: np.ndarray) -> Dict[int, int]:
    """Formulate and solve the BQP problem using SCIP."""
    model_scip = Model("AttributeAssignment")
    
    N, M = similarities.shape
    
    # Create binary variables x_{ij}
    x = {}
    for i in range(N):
        for j in range(M):
            x[i, j] = model_scip.addVar(vtype='B', name=f"x_{i}_{j}")
    
    # Objective function: maximize total similarity
    model_scip.setObjective(
        quicksum(similarities[i, j][0][0] * x[i, j] for i in range(N) for j in range(M)),
        'maximize'
    )
    
    # Constraints: Each node assigned to at most one attribute
    for i in range(N):
        model_scip.addCons(
            quicksum(x[i, j] for j in range(M)) <= 1
        )
    
    # Solve the BQP
    model_scip.optimize()
    
    # Extract assignments
    assignments = {}
    for i in range(N):
        for j in range(M):
            val = model_scip.getVal(x[i, j])
            if val > 0.5:
                assignments[i] = j  # Node i assigned to attribute j
                
    return assignments


def extract_attribute_texts(assignments: Dict[int, int], node_texts: List[str], attributes: List[str]) -> Dict[str, List[str]]:
    """Extract attribute texts based on assignments."""
    attribute_texts = {attr: [] for attr in attributes}
    for i, j in assignments.items():
        attr = attributes[j]
        attribute_texts[attr].append(node_texts[i])
    return attribute_texts


def extract_credit_card_properties(attribute_texts: Dict[str, List[str]]) -> Dict:
    """Extract required properties from the attribute texts."""
    # For simplicity, we'll mock the extraction functions
    
    # Name and issuer
    name = "XYZ Rewards Card"
    issuer = get_issuer(name)
    
    # Benefits
    benefits_text = ' '.join(attribute_texts.get('Benefits', []))
    benefits = multiple_nearest(benefits_text, Benefit)
    
    # Credit needed
    credit_needed_text = ' '.join(attribute_texts.get('Credit Needed', []))
    credit_needed = multiple_nearest(credit_needed_text, CreditNeeded)
    
    # Reward category map
    rewards_text = ' '.join(attribute_texts.get('Rewards', []))
    reward_category_map = [
        RewardCategoryRelation(
            category=PurchaseCategory.DINING,
            reward_unit=RewardUnit.PERCENT_CASHBACK_USD,
            reward_amount=3.0
        ),
        RewardCategoryRelation(
            category=PurchaseCategory.GROCERIES,
            reward_unit=RewardUnit.PERCENT_CASHBACK_USD,
            reward_amount=2.0
        )
    ]
    
    # Sign-on bonus
    sign_on_bonus_text = ' '.join(attribute_texts.get('Sign-on Bonus', []))
    sign_on_bonus = [
        ConditionalSignOnBonus(
            purchase_type=PurchaseCategory.GENERAL,
            condition_amount=1000.0,
            timeframe=timedelta(days=90),
            reward_type=RewardUnit.UNKNOWN,
            reward_amount=20000.0
        )
    ]
    
    # APR
    apr_text = ' '.join(attribute_texts.get('APR', []))
    apr = [
        APR(
            apr=15.99,
            type=APRType.PURCHASE
        )
    ]
    
    # Annual fee
    annual_fee_text = ' '.join(attribute_texts.get('Annual Fee', []))
    annual_fee = AnnualFee(
        fee_usd=95.0,
        waived_for=1
    )
    
    # Primary reward unit
    primary_reward_unit = get_primary_reward_unit(reward_category_map)
    
    # Keywords
    keywords = [CreditCardKeyword.REWARDS_FOCUSED, CreditCardKeyword.NO_ANNUAL_FEE]
    
    properties = {
        'name': name,
        'issuer': issuer,
        'benefits': benefits,
        'credit_needed': credit_needed,
        'reward_category_map': reward_category_map,
        'sign_on_bonus': sign_on_bonus,
        'apr': apr,
        'annual_fee': annual_fee,
        'primary_reward_unit': primary_reward_unit,
        'keywords': keywords
    }
    
    return properties


def create_credit_card_schema_object(properties: Dict) -> CreditCardSchema:
    """Create CreditCardSchema object from extracted properties."""
    try:
        credit_card = CreditCardSchema(**properties)
    except ValidationError as e:
        print(f"Validation error: {e}")
        return None
    return credit_card


def main():
    # Initialize models
    nlp, tokenizer, bert_model = initialize_nlp_models()
    
    # Input text (credit card offer descriptions)
    text = """
    The XYZ Rewards Card offers a welcome bonus of 20,000 points after spending $1,000 in the first three months.
    Earn 3% cashback on dining and 2% on groceries. There is an annual fee of $95, waived for the first year.
    """
    
    # Step 1: Parse text into sentences
    sentences = parse_text_to_sentences(text, nlp)
    node_texts = sentences
    
    # Step 2: Generate embeddings for each sentence (node)
    node_embeddings = generate_sentence_embeddings(sentences, tokenizer, bert_model)
    
    # Step 3: Define attributes and compute their embeddings
    attributes = ['Annual Fee', 'Sign-on Bonus', 'Rewards', 'APR', 'Benefits', 'Credit Needed']
    attribute_embeddings = compute_attribute_embeddings(attributes, tokenizer, bert_model)
    
    # Step 4: Compute similarities between nodes and attributes
    similarities = compute_similarities(node_embeddings, attribute_embeddings)
    
    # Step 5: Formulate and solve the BQP
    assignments = formulate_and_solve_bqp(similarities)
    
    # Step 6: Extract attribute texts based on assignments
    attribute_texts = extract_attribute_texts(assignments, node_texts, attributes)
    
    # Step 7: Extract credit card properties from attribute texts
    properties = extract_credit_card_properties(attribute_texts)
    
    # Step 8: Create CreditCardSchema object
    credit_card = create_credit_card_schema_object(properties)
    
    if credit_card:
        # Output the credit card object
        print(credit_card.json(indent=4, default=str))
    else:
        print("Failed to create CreditCardSchema object.")
    
    # Close Stanford CoreNLP connection
    nlp.close()

if __name__ == "__main__":
    main()
