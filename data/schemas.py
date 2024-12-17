from enum import Enum
from typing import List, Optional, Union
from pydantic import BaseModel, ConfigDict, field_validator
from datetime import timedelta

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
    GOOD = "Good"  # 690-719
    FAIR = "Fair"  # 630-689
    POOR = "Bad"  # 0-629

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

class Vendors(str, Enum):
    AMAZON = "Amazon"
    TARGET = "Target"
    WALGREENS = "Walgreens"
    WALMART = "Walmart"
    KROGER = "Kroger"
    LOWES = "Lowes"
    ALDI = "Aldi"
    COSTCO = "Costco"
    UNKNOWN = "Unknown"

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

class RewardCategoryThreshold(BaseModel):
    on_up_to_purchase_amount_usd: float
    per_timeframe_num_months: int
    fallback_reward_amount: float

    model_config = ConfigDict(from_attributes=True)

    @field_validator('fallback_reward_amount')
    def amount_must_be_reasonable(cls, v: float) -> float:
        if v < 0 or v > 10:
            raise ValueError('Amount must be positive and less than 10 to be reasonable')
        return v

class RewardCategoryRelation(BaseModel):
    category: Union[PurchaseCategory, Vendors]
    reward_unit: RewardUnit
    reward_amount: float
    reward_threshold: Optional[RewardCategoryThreshold] = None

    model_config = ConfigDict(from_attributes=True)

    @field_validator('reward_amount')
    def amount_must_be_reasonable(cls, v: float) -> float:
        if v < 0 or v > 20:
            raise ValueError('Amount must be positive and less than 20 to be reasonable')
        return v

class APR(BaseModel):
    apr: float
    apr_type: APRType

    model_config = ConfigDict(from_attributes=True)

    @field_validator('apr')
    def apr_must_be_reasonable(cls, v: float) -> float:
        if v < 0 or v > 100:
            raise ValueError('APR must be positive and less than 100 to be reasonable')
        return v

class AnnualFee(BaseModel):
    fee_usd: float
    waived_for: int

    model_config = ConfigDict(from_attributes=True)

    @field_validator('fee_usd', 'waived_for')
    def values_must_be_non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError('Value must be non-negative')
        return v

class ConditionalSignOnBonus(BaseModel):
    purchase_type: Union[PurchaseCategory, Vendors]
    condition_amount: float
    timeframe: timedelta
    reward_type: RewardUnit
    reward_amount: float

    model_config = ConfigDict(from_attributes=True)

    @field_validator('timeframe', mode='before')
    def parse_as_months(cls, v):
        if isinstance(v, (int, float)):
            return timedelta(days=30 * v)
        return v

class PeriodicStatementCredit(BaseModel):
    credit_amount: float
    unit: RewardUnit
    categories: List[PurchaseCategory]
    vendors: List[Vendors]
    timeframe_months: int
    max_uses: int
    description: str

    model_config = ConfigDict(from_attributes=True)

    @field_validator('timeframe_months')
    def timeframe_must_be_positive(cls, v: int) -> int:
        if v < 0:
            raise ValueError('Timeframe must be positive')
        return v

class CreditCard(BaseModel):
    name: str
    issuer: Issuer
    reward_category_map: List[RewardCategoryRelation]
    benefits: List[Benefit]
    credit_needed: List[CreditNeeded]
    apr: List[APR]
    sign_on_bonus: Optional[List[ConditionalSignOnBonus]] = None
    annual_fee: Optional[AnnualFee] = None
    statement_credit: List[PeriodicStatementCredit]
    primary_reward_unit: RewardUnit
    keywords: List[CreditCardKeyword]

    model_config = ConfigDict(from_attributes=True)

    def __eq__(self, other):
        if not isinstance(other, CreditCard):
            return False
        return self.name == other.name and self.issuer == other.issuer

    def __hash__(self):
        return hash((self.name, self.issuer))