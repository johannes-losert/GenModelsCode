from bs4 import BeautifulSoup
from typing import List
from data.extractors.creditcard_extractor import CreditCardExtractor, CreditCardDescription

SITE_NAME = "CreditKarma"

class CreditKarmaExtractor(CreditCardExtractor):
    """Implementation of CreditCardExtractor for Credit Karma website."""

    @property
    def site_name(self) -> str:
        return SITE_NAME
    
    def _process_html(self, html_content: str) -> List[CreditCardDescription]:
        soup = BeautifulSoup(html_content, 'html.parser')
        cards = soup.find_all('div', class_='ck-card')
        descriptions = []

        for card in cards:
            card_texts = []
            
            # Extract card name
            name_element = card.find('h2', class_='lh-copy mv0 f4 f3-ns')
            if name_element:
                card_texts.append(name_element.get_text(strip=True))
            
            # Extract tagline/description if available
            tagline = card.find('span', class_='f4 lh-copy ck-green-70 b')
            if tagline:
                card_texts.append(tagline.get_text(strip=True))
            
            # Extract features (rewards rate, annual fee, welcome bonus)
            features = card.find_all('li', class_='flex flex-column bb b--light-gray pv2 ph3')
            for feature in features:
                feature_text = feature.get_text(" ", strip=True)
                if feature_text:
                    card_texts.append(feature_text)
            
            # Extract collapser section attributes
            collapser_items = card.find_all('div', class_='ck-link-parent lh-copy ml2 f5 f4-ns')
            for item in collapser_items:
                text = item.get_text(strip=True)
                if text:
                    card_texts.append(text)
            
            # Extract nested list items
            list_items = card.find_all('ul', class_='list')
            for ul in list_items:
                for li in ul.find_all('li'):
                    text = li.get_text(" ", strip=True)
                    if text:
                        card_texts.append(text)
            
            # Combine all texts for this card with proper spacing
            if card_texts:
                full_description = " ".join(text for text in card_texts if text)
                descriptions.append(CreditCardDescription(description=full_description, site=SITE_NAME))

        return descriptions