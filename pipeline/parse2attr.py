import numpy as np
import json
import pickle
import logging
import random
from datetime import datetime
from pyscipopt import Model, quicksum
from sentence_transformers import SentenceTransformer, CrossEncoder
import spacy
import re
from typing import List, Dict, Union, Any
import nltk
from transformers import TRANSFORMERS_CACHE
import sys
from transformers.utils import logging as transformers_logging

from pydantic import ValidationError
from pipeline.schemas import (
    Issuer, RewardUnit, Benefit, CreditNeeded, PurchaseCategory, APRType, CreditCardKeyword,
    RewardCategoryThreshold, RewardCategoryRelation, ConditionalSignOnBonus, APR, AnnualFee,
    CreditCardSchema, get_issuer, multiple_nearest, get_primary_reward_unit
)
from pipeline.schemas import CRSchema, CKSchema
from pipeline.examples import attribute_examples

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load nltk and spaCy for parsing
nltk.download('punkt')
nlp_spacy = spacy.load("en_core_web_sm")

# Load models
try:
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device='cpu')
    logger.info("Models loaded successfully.")
except Exception as e:
    logger.error("Error loading SentenceTransformer or CrossEncoder model: %s", e)
    sys.exit("Model loading failed. Exiting.")

# Define attribute hierarchy as a nested dictionary
attribute_hierarchy = {
    'Annual Fee': {},
    'Sign-on Bonus': {
        'purchase_type': {},
        'condition_amount': {},
        'timeframe': {},
        'reward_type': {},
        'reward_amount': {}
    },
    'Reward Category Map': {
        'category': {},
        'reward_unit': {},
        'reward_amount': {},
        'reward_threshold': {
            'on_up_to_purchase_amount_usd': {},
            'per_timeframe_num_months': {},
            'fallback_reward_amount': {}
        }
    },
    'APR': {
        'apr': {},
        'type': {}
    },
    'Benefits': {},
    'Credit Needed': {}
}

def get_average_embedding(texts: List[str], model: SentenceTransformer) -> np.ndarray:
    embeddings = model.encode(texts, convert_to_tensor=True)
    return embeddings.mean(axis=0).cpu().numpy()

def compute_cross_encoder_similarities(phrases: List[str], attribute_texts: List[str]) -> np.ndarray:
    if not phrases or not attribute_texts:
        return np.array([])

    avg_attribute_embedding = get_average_embedding(attribute_texts, sentence_model)
    phrase_embeddings = sentence_model.encode(phrases, convert_to_tensor=True)
    similarities = (phrase_embeddings @ avg_attribute_embedding) / (
        np.linalg.norm(phrase_embeddings, axis=1) * np.linalg.norm(avg_attribute_embedding)
    )
    similarities = similarities.cpu().numpy()
    similarities = (similarities - np.min(similarities)) / (np.max(similarities) - np.min(similarities) + 1e-6)
    return similarities

def calculate_intra_similarity_matrix(phrases: List[str]) -> np.ndarray:
    embeddings = sentence_model.encode(phrases, convert_to_tensor=True)
    similarity_matrix = np.dot(embeddings, embeddings.T) / (np.linalg.norm(embeddings, axis=1)[:, None] * np.linalg.norm(embeddings, axis=1))
    return similarity_matrix

def load_data_from_pickle(pickle_path: str) -> Dict[str, Union[CRSchema, CKSchema]]:
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

def filter_relevant_text(subtree_text: str) -> bool:
    unwanted_patterns = [r"^\W+$", r"^\d+$", r"^\s*$"]
    return not any(re.search(pattern, subtree_text, re.IGNORECASE) for pattern in unwanted_patterns)

def parse_text_with_spacy(text: str) -> List[str]:
    doc = nlp_spacy(text)
    phrases = set()
    for sent in doc.sents:
        if filter_relevant_text(sent.text):
            phrases.add(sent.text.strip())
        for np in sent.noun_chunks:
            if filter_relevant_text(np.text):
                phrases.add(np.text.strip())
        for token in sent:
            if token.dep_ in ('ROOT', 'advcl', 'xcomp', 'ccomp', 'pcomp', 'acl'):
                subtree = ' '.join([t.text for t in token.subtree])
                if filter_relevant_text(subtree):
                    phrases.add(subtree.strip())
    return list(phrases)

def formulate_and_solve_bqp(similarities_per_attr: List[np.ndarray], attribute_paths: List[str], disjoint_constraints: Dict[str, List[str]], intra_similarity_matrices: List[np.ndarray]) -> Dict[str, List[int]]:
    model = Model("AttributeAssignment")
    num_phrases = len(similarities_per_attr[0])

    # Define variables
    x = {(i, j): model.addVar(vtype="B", name=f"x_{i}_{j}") for i in range(num_phrases) for j in range(len(attribute_paths))}
    num_assignments = {j: model.addVar(vtype="I", name=f"num_assignments_{j}") for j in range(len(attribute_paths))}

    # Objective weights
    similarity_weight = 2
    overassignment_penalty_weight = 1.5

    # Objective: maximize phrase-to-attribute similarity and penalize overassignment
    objective = quicksum(similarity_weight * similarities_per_attr[j][i] * x[i, j] for j in range(len(attribute_paths)) for i in range(num_phrases))
    objective -= overassignment_penalty_weight * quicksum(num_assignments[j] for j in range(len(attribute_paths)))

    model.setObjective(objective, "maximize")

    # Linear hierarchy constraints
    for parent, sub_attrs in disjoint_constraints.items():
        if parent and sub_attrs:
            parent_idx = attribute_paths.index(parent)
            for sub_attr in sub_attrs:
                sub_attr_idx = attribute_paths.index(sub_attr)
                for i in range(num_phrases):
                    model.addCons(x[i, sub_attr_idx] <= x[i, parent_idx])

    # Ensure only one assignment per level
    for level, attrs in disjoint_constraints.items():
        for i in range(num_phrases):
            model.addCons(quicksum(x[i, attribute_paths.index(attr)] for attr in attrs if attr in attribute_paths) <= 1)

    # Constraints for auxiliary variables: enforce num_assignments[j] as the count of assignments for each attribute
    for j in range(len(attribute_paths)):
        model.addCons(num_assignments[j] == quicksum(x[i, j] for i in range(num_phrases)))

    model.optimize()

    # Extract assignments
    assigned_phrases = {attr: [] for attr in attribute_paths}
    for i in range(num_phrases):
        for j, attr in enumerate(attribute_paths):
            if model.getVal(x[i, j]) > 0.5:
                assigned_phrases[attr].append(i)

    logger.debug(f"Assignments after BQP with hierarchy constraints: {assigned_phrases}")
    return assigned_phrases

def build_assigned_hierarchy(assigned_phrases: Dict[str, List[int]], phrases: List[str], hierarchy: Dict[str, Any]) -> Dict[str, Any]:
    def recursive_build(hierarchy_node):
        result = {}
        for key, sub_attrs in hierarchy_node.items():
            full_path = key if isinstance(hierarchy, dict) else f"{hierarchy}.{key}"
            result[key] = {
                "text": [phrases[i] for i in assigned_phrases.get(full_path, [])],
                "sub_attributes": recursive_build(sub_attrs)
            }
        return result
    return recursive_build(hierarchy)

def flatten_hierarchy(hierarchy: Dict[str, Any], path: str = "") -> List[str]:
    paths = []
    for key, sub_attrs in hierarchy.items():
        full_path = f"{path}.{key}" if path else key
        paths.append(full_path)
        paths.extend(flatten_hierarchy(sub_attrs, full_path))
    return paths

def create_disjoint_constraints(attribute_hierarchy: Dict[str, Any]) -> Dict[str, List[str]]:
    disjoint_constraints = {}
    def traverse_hierarchy(hierarchy, path=''):
        for key, sub_attrs in hierarchy.items():
            full_path = f"{path}.{key}" if path else key
            if path not in disjoint_constraints:
                disjoint_constraints[path] = []
            disjoint_constraints[path].append(full_path)
            traverse_hierarchy(sub_attrs, full_path)
    traverse_hierarchy(attribute_hierarchy)
    logger.debug(f"Disjoint constraints created: {disjoint_constraints}")
    return disjoint_constraints

def process_attributes(phrases: List[str], attribute_hierarchy: Dict[str, Any], attribute_examples: Dict[str, List[str]]) -> Dict[str, Any]:
    flat_hierarchy = flatten_hierarchy(attribute_hierarchy)
    similarities_per_attr = []
    intra_similarity_matrices = []

    for attr in flat_hierarchy:
        examples = attribute_examples.get(attr, [])
        if examples:
            similarity = compute_cross_encoder_similarities(phrases, examples)
            similarities_per_attr.append(similarity)
        else:
            similarities_per_attr.append(np.zeros(len(phrases)))
        
        intra_similarity_matrix = calculate_intra_similarity_matrix(phrases)
        intra_similarity_matrices.append(intra_similarity_matrix)

    disjoint_constraints = create_disjoint_constraints(attribute_hierarchy)
    assignments = formulate_and_solve_bqp(similarities_per_attr, flat_hierarchy, disjoint_constraints, intra_similarity_matrices)

    return build_assigned_hierarchy(assignments, phrases, attribute_hierarchy)

def main(max_cards: int = 10):
    pickle_files = ["../extraction_pkl_out/20241109_194651_credit_karma_dict.pkl"]
    all_data = {}
    
    for file_path in pickle_files:
        data = load_data_from_pickle(file_path)
        random_sample = random.sample(list(data.items()), min(max_cards, len(data)))
        
        for card_name, card_data in random_sample:
            text = card_data.unparsed_card_attributes if isinstance(card_data, CRSchema) else card_data.attributes
            assigned_attributes = process_attributes(parse_text_with_spacy(text), attribute_hierarchy, attribute_examples)
            all_data[card_name] = assigned_attributes

    with open("../json_output/credit_card_data.json", 'w') as outfile:
        json.dump(all_data, outfile, indent=4)

if __name__ == "__main__":
    main(max_cards=10)
