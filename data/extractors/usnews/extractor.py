from bs4 import BeautifulSoup
from typing import List, Optional
from data.extractors.creditcard_extractor import CreditCardExtractor, CreditCardDescription
from tqdm import tqdm

SITE_NAME = "USNews"

class USNewsExtractor(CreditCardExtractor):
    """Implementation of CreditCardExtractor for USNews website."""

    @property 
    def site_name(self) -> str:
        return SITE_NAME

    def _process_html(self, html_content: str) -> List[CreditCardDescription]:
        soup = BeautifulSoup(html_content, 'html.parser')
        descriptions = []

        # Use a more flexible approach than strict regex for container selection
        cards = soup.select('div[class*="CardDetailContainer"]')
        
        # Initialize progress bar
        pbar = tqdm(total=len(cards), desc="Extracting credit cards", unit="card")

        for card in cards:
            card_texts = []

            # Extract card name/title
            card_name = card.select_one('h5[class*="Heading-"]')
            if card_name:
                card_texts.append(f"Card Name: {card_name.get_text(strip=True)}")
                pbar.set_postfix_str(f"Processing: {card_name.get_text(strip=True)[:30]}...")

            # Extract summary/intro
            intro = card.select_one('p[class*="ProfileIntroParagraph"]')
            if intro:
                card_texts.append(f"Summary: {intro.get_text(strip=True)}")

            # Extract data points (label-value pairs)
            self._extract_data_points(card, card_texts)

            # Extract pros
            self._extract_pros(card, card_texts)

            # Extract cons
            self._extract_cons(card, card_texts)

            # Extract rates and fees if available
            self._extract_rates_and_fees(soup, card_texts)

            # Extract additional sections with headers
            self._extract_additional_sections(soup, card_texts)

            # Extract Editor's Take section
            self._extract_editors_take(soup, card_texts)

            # Extract Alternative Pick section if present
            self._extract_alternative_pick(soup, card_texts)

            if card_texts:
                full_description = "\n".join(text for text in card_texts if text)
                descriptions.append(CreditCardDescription(description=full_description))
            
            # Update progress bar
            pbar.update(1)

        # Close progress bar
        pbar.close()
        return descriptions

    def _extract_data_points(self, card, card_texts: List[str]):
        data_points = card.select('div[class*="DataPointCell"]')
        if data_points:
            card_texts.append("Key Details:")
            for point in data_points:
                label = point.select_one('p[class*="DataPointLabel"]')
                value = None
                for p in point.find_all('p'):
                    if 'DataPointLabel' not in (p.get('class') or ['']):
                        value = p
                        break
                if label and value:
                    card_texts.append(f"{label.get_text(strip=True)}: {value.get_text(strip=True)}")

    def _extract_pros(self, card, card_texts: List[str]):
        pros_section = card.select_one('div[class*="ProConCell"] h3:contains("Pros")')
        if pros_section:
            pros_container = pros_section.find_parent('div', class_=lambda c: c and 'ProConCell' in c)
            if pros_container:
                pros_list = pros_container.find('ul')
                if pros_list:
                    card_texts.append("Pros:")
                    for pro_item in pros_list.find_all('li'):
                        text = pro_item.find('p')
                        if text:
                            card_texts.append(f"- {text.get_text(strip=True)}")

    def _extract_cons(self, card, card_texts: List[str]):
        procon_cells = card.select('div[class*="ProConCell"]')
        if len(procon_cells) > 1:
            cons_section = procon_cells[-1].select_one('h3:contains("Cons")')
            if cons_section:
                cons_list = procon_cells[-1].find('ul')
                if cons_list:
                    card_texts.append("Cons:")
                    for con_item in cons_list.find_all('li'):
                        text = con_item.find('p')
                        if text:
                            card_texts.append(f"- {text.get_text(strip=True)}")

    def _extract_rates_and_fees(self, soup, card_texts: List[str]):
        rates_fees = soup.find('div', id='rates-fees')
        if rates_fees:
            card_texts.append("Rates and Fees:")
            fees_list = rates_fees.select('li[class*="List__ListItem"]')
            for fee in fees_list:
                fee_text = fee.get_text(strip=True)
                if fee_text:
                    card_texts.append(f"- {fee_text}")

    def _extract_additional_sections(self, soup, card_texts: List[str]):
        sections = soup.select('h2[class*="AnchorSpanTag"]')
        for section in sections:
            section_title = section.get_text(strip=True)
            if section_title:
                next_content = section.find_next(['p','div'])
                if next_content:
                    content_text = next_content.get_text(strip=True)
                    if content_text:
                        card_texts.append(f"{section_title}:")
                        card_texts.append(content_text)

    def _extract_editors_take(self, soup, card_texts: List[str]):
        editors_take = soup.select_one('div[class*="CreditCardEditorsTakeEnhancement"]')
        if editors_take:
            quote = editors_take.select_one('div[class*="Raw-"]')
            if quote:
                card_texts.append("Editor's Take:")
                card_texts.append(quote.get_text(strip=True))

    def _extract_alternative_pick(self, soup, card_texts: List[str]):
        alt_pick = soup.select_one('div[class*="AlternativeCard__Container"]')
        if alt_pick:
            card_texts.append("Alternative Pick:")
            alt_description = alt_pick.select_one('em[class*="Em__Strong"]')
            if alt_description:
                card_texts.append(alt_description.get_text(strip=True))