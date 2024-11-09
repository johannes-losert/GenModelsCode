# schemas.py

# Import necessary libraries
from enum import Enum
from typing import List, Union, Optional
from datetime import timedelta
from collections import defaultdict
from pydantic import BaseModel, Field, validator

# Enums as per specifications

class Issuer(str, Enum):
    CAPITAL_ONE = "Capital One"
    CHASE = "Chase"
    AMERICAN_EXPRESS = "American Express"
    CITI = "Citi"
    DISCOVER = "Discover"
    BANK_OF_AMERICA = "Bank of America"
    WELLS_FARGO = "Wells Fargo"
    BARCLAYS = "Barclays"
    US_BANK = "US Bank"
    PNC = "PNC"
    TD_BANK = "TD Bank"
    HSBC = "HSBC"

class RewardUnit(str, Enum):
    CHASE_ULTIMATE_REWARDS = "Chase Ultimate Rewards"
    AMEX_MEMBERSHIP_REWARDS = "American Express Membership Rewards"
    CITI_THANKYOU_POINTS = "Citi ThankYou Points"
    CAPITAL_ONE_MILES = "Capital One Miles"
    WELLS_FARGO_GO_FAR_REWARDS = "Wells Fargo Go Far Rewards"
    BANK_OF_AMERICA_PREFERRED_REWARDS = "Bank of America Preferred Rewards"
    BARCLAYS_ARRIVAL_POINTS = "Barclays Arrival Points"
    DISCOVER_CASHBACK_BONUS = "Discover Cashback Bonus"
    US_BANK_ALTITUDE_POINTS = "U.S. Bank Altitude Points"
    PNC_POINTS = "PNC Points"
    HILTON_HONORS_POINTS = "Hilton Honors Points"
    MARRIOTT_BONVOY_POINTS = "Marriott Bonvoy Points"
    WORLD_OF_HYATT_POINTS = "World of Hyatt Points"
    DELTA_SKYMILES = "Delta SkyMiles"
    UNITED_MILEAGEPLUS = "United MileagePlus"
    AA_ADVANTAGE_MILES = "American Airlines AAdvantage Miles"
    SOUTHWEST_RAPID_REWARDS = "Southwest Rapid Rewards"
    IHG_ONE_REWARDS_POINTS = "IHG One Rewards Points"
    JETBLUE_TRUEBLUE_POINTS = "JetBlue TrueBlue Points"
    ALASKA_MILEAGE_PLAN_MILES = "Alaska Mileage Plan Miles"
    RADISSON_REWARDS_POINTS = "Radisson Rewards Points"
    PERCENT_CASHBACK_USD = "Percent Cashback USD"
    STATEMENT_CREDIT_USD = "Statement Credit USD"
    AVIOS = "Avios"
    AEROPLAN_POINTS = "Aeroplan Points"
    CHOICE_PRIVILEGES_POINTS = "Choice Privileges Points"
    UNKNOWN = "Unknown"

class Benefit(str, Enum):
    AIRPORT_LOUNGE_ACCESS = "airport lounge access"
    CELL_PHONE_PROTECTION = "cell phone protection"
    CONCIERGE_SERVICE = "concierge service"
    EMERGENCY_MEDICAL_INSURANCE = "emergency medical insurance"
    EVENT_TICKET_ACCESS = "event ticket access"
    EXTENDED_RETURN_PERIOD = "extended return period"
    EXTENDED_WARRANTY = "extended warranty"
    FREE_CHECKED_BAGS = "free checked bags"
    GLOBAL_ENTRY_TSA_PRECHECK_CREDIT = "global entry/tsa precheck credit"
    NO_FOREIGN_TRANSACTION_FEES = "no foreign transaction fees"
    PRICE_PROTECTION = "price protection"
    PRIORITY_BOARDING = "priority boarding"
    PURCHASE_PROTECTION = "purchase protection"
    RENTAL_CAR_INSURANCE = "rental car insurance"
    RETURN_PROTECTION = "return protection"
    TRAVEL_ASSISTANCE_SERVICES = "travel assistance services"
    TRAVEL_INSURANCE = "travel insurance"

class CreditNeeded(str, Enum):
    EXCELLENT = "Excellent"  # 720-850
    GOOD = "Good"            # 690-719
    FAIR = "Fair"            # 630-689
    POOR = "Bad"             # 0-629 

class PurchaseCategory(str, Enum):
    ACCOMMODATION = "accommodation"
    ADVERTISING = "advertising"
    BAR = "bar"
    CHARITY = "charity"
    CLOTHING = "clothing"
    DINING = "dining"
    EDUCATION = "education"
    ELECTRONICS = "electronics"
    ENTERTAINMENT = "entertainment"
    FUEL = "fuel"
    GENERAL = "general"
    GROCERIES = "groceries"
    HEALTH = "health"
    HOME = "home"
    INCOME = "income"
    INSURANCE = "insurance"
    INVESTMENT = "investment"
    LOAN = "loan"
    OFFICE = "office"
    PHONE = "phone"
    SERVICE = "service"
    SHOPPING = "shopping"
    SOFTWARE = "software"
    SPORT = "sport"
    TAX = "tax"
    TRANSPORT = "transport"
    TRANSPORTATION = "transportation"
    UTILITIES = "utilities"
    UNKNOWN = "unknown"

class APRType(str, Enum):
    PURCHASE = "Purchase"
    CASH_ADVANCE = "Cash Advance"
    BALANCE_TRANSFER = "Balance Transfer"
    PROMOTIONAL = "Promotional"
    PENALTY = "Penalty"

class CreditCardKeyword(str, Enum):
    BUSINESS = "Business"
    PERSONAL = "Personal"
    STUDENT = "Student"
    TRAVEL = "Travel"
    REWARDS_FOCUSED = "Rewards-focused"
    CUSTOMIZABLE_REWARDS = "Customizable Rewards"
    LOW_APR = "Low APR"
    NO_ANNUAL_FEE = "No Annual Fee"
    CASHBACK = "Cashback"
    BALANCE_TRANSFER = "Balance Transfer"
    SECURED = "Secured"
    HIGH_LIMIT = "High Limit"
    LUXURY = "Luxury"
    AIRLINE = "Airline"
    HOTEL = "Hotel"
    GAS = "Gas"
    GROCERY = "Grocery"
    DINING = "Dining"
    SMALL_BUSINESS = "Small Business"
    INTRO_APR = "Intro APR"

# Helper functions

def strip_up_to_period(text):
    parts = text.split('.', 1) 
    if len(parts) > 1:
        return parts[1].strip()
    return text.strip()

def single_nearest(text: str, enum: Enum):
    if text is None:
        return None
    for enum_element in enum:
        if strip_up_to_period(enum_element.value).lower() in text.lower():
            return enum_element
    return None

def multiple_nearest(text: str, enum: Enum):
    if text is None:
        return []
    out_enums = []
    for enum_element in enum:
        if strip_up_to_period(enum_element.value).lower() in text.lower():
            out_enums.append(enum_element)
    return out_enums

def get_issuer(card_name: str):
    best_issuer = single_nearest(card_name, Issuer)
    if best_issuer:
        return best_issuer
    return "Unknown"

# Data classes using pydantic

class RewardCategoryThreshold(BaseModel):
    on_up_to_purchase_amount_usd: float
    per_timeframe_num_months: int
    fallback_reward_amount: float

    @validator('on_up_to_purchase_amount_usd', 'fallback_reward_amount')
    def amounts_must_be_positive(cls, v):
        if v < 0:
            raise ValueError('Amount must be positive')
        return v

    @validator('per_timeframe_num_months')
    def months_must_be_positive(cls, v):
        if v < 0:
            raise ValueError('Timeframe must be positive')
        return v

class RewardCategoryRelation(BaseModel):
    category: Union[PurchaseCategory, str]  # Adjusted to allow string for simplicity
    reward_unit: RewardUnit
    reward_amount: float
    reward_threshold: Optional[RewardCategoryThreshold] = None

    @validator('reward_amount')
    def amount_must_be_reasonable(cls, v):
        if v < 0 or v > 100:
            raise ValueError('Amount must be positive and less than 100 to be reasonable')
        return v

class ConditionalSignOnBonus(BaseModel):
    purchase_type: Union[PurchaseCategory, str]  # Adjusted to allow string
    condition_amount: float
    timeframe: timedelta
    reward_type: RewardUnit
    reward_amount: float

    @validator('condition_amount', 'reward_amount')
    def amounts_must_be_positive(cls, v):
        if v < 0:
            raise ValueError('Amount must be positive')
        return v

    @validator('timeframe', pre=True)
    def parse_timeframe(cls, v):
        if isinstance(v, (int, float)):
            return timedelta(days=30 * v)
        elif isinstance(v, timedelta):
            return v
        else:
            raise ValueError('Invalid timeframe')

class APR(BaseModel):
    apr: float
    type: APRType

    @validator('apr')
    def apr_must_be_reasonable(cls, v):
        if v < 0 or v > 100:
            raise ValueError('APR must be between 0 and 100')
        return v

class AnnualFee(BaseModel):
    fee_usd: float
    waived_for: int

    @validator('fee_usd')
    def fee_must_be_non_negative(cls, v):
        if v < 0:
            raise ValueError('Fee must be non-negative')
        return v

    @validator('waived_for')
    def waived_for_must_be_non_negative(cls, v):
        if v < 0:
            raise ValueError('Waived for must be non-negative')
        return v

class CreditCardSchema(BaseModel):
    name: str
    issuer: Union[Issuer, str]
    benefits: List[Benefit]
    credit_needed: List[CreditNeeded]
    reward_category_map: List[RewardCategoryRelation]
    sign_on_bonus: Optional[List[ConditionalSignOnBonus]] = None
    apr: List[APR]
    annual_fee: Optional[AnnualFee] = None
    primary_reward_unit: RewardUnit
    keywords: List[CreditCardKeyword]

# Function to get primary reward unit

def get_primary_reward_unit(reward_category_map):
    if not reward_category_map:
        return RewardUnit.UNKNOWN
    reward_unit_counts = defaultdict(int)
    for relation in reward_category_map:
        reward_unit_counts[relation.reward_unit] += 1
    primary_reward_unit = max(reward_unit_counts, key=reward_unit_counts.get)
    return primary_reward_unit
