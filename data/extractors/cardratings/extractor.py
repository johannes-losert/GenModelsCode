from data.extractors.creditcard_extractor import CreditCardExtractor, CreditCardDescription
from typing import List

SITE_NAME = "CardRatings"

class CardRatingsExtractor(CreditCardExtractor):
    """Implementation of CreditCardExtractor for CardRatings website."""

    @property
    def site_name(self) -> str:
        return SITE_NAME

    def _process_html(self, html_content: str) -> List[CreditCardDescription]:
        from bs4 import BeautifulSoup
        
        soup = BeautifulSoup(html_content, 'html.parser')
        cards = soup.find_all(class_="CardDetails")
        descriptions = []

        for card in cards:
            card_texts = []
            
            card_title = card.find('h2')
            if card_title:
                card_texts.append(card_title.get_text(strip=True))
            
            credit_div = card.find(class_='rightDetail')
            if credit_div:
                credit_section = credit_div.find(class_='credit_div')
                if credit_section:
                    issuer_elem = credit_section.find(class_='apply_now_bank_name')
                    if issuer_elem:
                        card_texts.append(f"Issuer: {issuer_elem.get_text(strip=True)}")
                    
                    credit_elem = credit_section.find(class_='credit_needed')
                    if credit_elem:
                        card_texts.append(f"Credit Required: {credit_elem.get_text(strip=True)}")

            mid_detail = card.find(class_="midDetail")
            if mid_detail:
                long_desc = mid_detail.find(class_="longDescription")
                if long_desc:
                    card_texts.append(long_desc.get_text(strip=True))
                
                ul_elements = mid_detail.find_all("ul")
                for ul in ul_elements:
                    li_items = ul.find_all("li")
                    for li in li_items:
                        card_texts.append(li.get_text(strip=True))

            if card_texts:
                full_description = " ".join(text for text in card_texts if text)
                descriptions.append(CreditCardDescription(description=full_description, site=SITE_NAME))

        return descriptions
