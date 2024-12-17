from openai import OpenAI
from typing import Dict, Any, Optional
from data.schemas import (Issuer, RewardUnit, Benefit, CreditNeeded, PurchaseCategory,
                     Vendors, APRType, CreditCardKeyword)

async def generate_credit_card_json(credit_card_description: str, api_key: str) -> Optional[Dict[Any, Any]]:
    """
    Generates a credit card JSON object using OpenAI's structured output capability.
    
    Args:
        prompt (str): Description of the credit card to generate
        api_key (str): OpenAI API key
    
    Returns:
        Optional[Dict[Any, Any]]: Generated credit card data in JSON format or None if refused
    """
    
    client = OpenAI(api_key=api_key)
    
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "issuer": {
                "type": "string",
                "enum": [issuer.value for issuer in Issuer]
            },
            "reward_category_map": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "category": {
                            "type": "string",
                            "enum": [cat.value for cat in PurchaseCategory] + [vendor.value for vendor in Vendors]
                        },
                        "reward_unit": {
                            "type": "string",
                            "enum": [reward.value for reward in RewardUnit if reward != RewardUnit.UNKNOWN]
                        },
                        "reward_amount": {"type": "number"},
                        "reward_threshold": {
                            "type": ["object", "null"],
                            "properties": {
                                "on_up_to_purchase_amount_usd": {"type": "number"},
                                "per_timeframe_num_months": {"type": "integer"},
                                "fallback_reward_amount": {"type": "number"}
                            },
                            "required": ["on_up_to_purchase_amount_usd", 
                                       "per_timeframe_num_months", 
                                       "fallback_reward_amount"],
                            "additionalProperties": False
                        }
                    },
                    "required": ["category", "reward_unit", "reward_amount", "reward_threshold"],
                    "additionalProperties": False
                }
            },
            "benefits": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": [benefit.value for benefit in Benefit]
                }
            },
            "credit_needed": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": [credit.value for credit in CreditNeeded]
                }
            },
            "apr": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "apr": {"type": "number"},
                        "apr_type": {
                            "type": "string",
                            "enum": [apr_type.value for apr_type in APRType]
                        }
                    },
                    "required": ["apr", "apr_type"],
                    "additionalProperties": False
                }
            },
            "sign_on_bonus": {
                "type": ["array", "null"],
                "items": {
                    "type": "object",
                    "properties": {
                        "purchase_type": {
                            "type": "string",
                            "enum": [cat.value for cat in PurchaseCategory if cat != PurchaseCategory.UNKNOWN] + 
                                  [vendor.value for vendor in Vendors if vendor != Vendors.UNKNOWN]
                        },
                        "condition_amount": {"type": "number"},
                        "timeframe": {"type": "integer"},
                        "reward_type": {
                            "type": "string",
                            "enum": [reward.value for reward in RewardUnit 
                                   if reward not in (RewardUnit.PERCENT_CASHBACK_USD, RewardUnit.UNKNOWN)]
                        },
                        "reward_amount": {"type": "number"}
                    },
                    "required": ["purchase_type", "condition_amount", "timeframe", 
                               "reward_type", "reward_amount"],
                    "additionalProperties": False
                }
            },
            "annual_fee": {
                "type": ["object", "null"],
                "properties": {
                    "fee_usd": {"type": "number"},
                    "waived_for": {"type": "integer"}
                },
                "required": ["fee_usd", "waived_for"],
                "additionalProperties": False
            },
            "statement_credit": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "credit_amount": {"type": "number"},
                        "unit": {
                            "type": "string",
                            "enum": [reward.value for reward in RewardUnit if reward != RewardUnit.UNKNOWN]
                        },
                        "categories": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": [cat.value for cat in PurchaseCategory if cat != PurchaseCategory.UNKNOWN]
                            }
                        },
                        "vendors": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": [vendor.value for vendor in Vendors if vendor != Vendors.UNKNOWN]
                            }
                        },
                        "timeframe_months": {"type": "integer"},
                        "max_uses": {"type": "integer"},
                        "description": {"type": "string"}
                    },
                    "required": ["credit_amount", "unit", "categories", "vendors",
                               "timeframe_months", "max_uses", "description"],
                    "additionalProperties": False
                }
            },
            "primary_reward_unit": {
                "type": "string",
                "enum": [reward.value for reward in RewardUnit]
            },
            "keywords": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": [keyword.value for keyword in CreditCardKeyword]
                }
            }
        },
        "required": ["name", "issuer", "reward_category_map", "benefits",
                    "credit_needed", "apr", "statement_credit",
                    "primary_reward_unit", "keywords", "sign_on_bonus", "annual_fee"],
        "additionalProperties": False
    }

    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "credit_card",
            "schema": schema,  # Your existing schema definition
            "strict": True
        }
    }

    system_prompt = """You are a credit card expert. Generate a valid credit card description 
    based on the provided information. Ensure:
    1. All reward rates are realistic (typically 1-5x points/miles or 1-5 percent cash back)
    2. Benefits match the card's annual fee and tier
    3. APRs are current market rates (typically 18-29 percent for purchase APR)
    4. Sign-on bonuses are competitive but realistic
    5. The primary_reward_unit matches the most common reward_unit in reward_category_map
    6. Keywords accurately reflect the card's features
    
    If you cannot generate a valid card, return a refusal message."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-2024-08-06",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": credit_card_description}
            ],
            response_format=response_format
        )
        
        # Check for refusal
        message = response.choices[0].message
        return message.content
    except Exception as e:
        raise Exception(f"Error generating credit card JSON: {str(e)}")