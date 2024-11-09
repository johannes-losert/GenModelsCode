from pydantic import BaseModel, ConfigDict

class CardScrapeSchema(BaseModel):
    name: str
    unparsed_issuer: str
    unparsed_credit_needed: str
    unparsed_card_attributes: str
    
    model_config = ConfigDict(from_attributes=True)