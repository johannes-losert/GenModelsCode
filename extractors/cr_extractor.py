from bs4 import BeautifulSoup
from typing import List
from pipeline.schemas import CRSchema
from datetime import datetime
import pickle

# Load HTML content from file
file_name = "sites/Complete Credit Card List - Cardratings.com.html"
with open(file_name, 'r', encoding='utf-8') as file:
    html_content = file.read()

# Initialize BeautifulSoup
soup = BeautifulSoup(html_content, 'html.parser')
cards = soup.find_all(class_="CardDetails")

# List to store card data
out_card_list = []

# Extract card data
for card in cards:
    description_used = None
    credit_div = card.find(class_='rightDetail').find(class_='credit_div')
    issuer = credit_div.find(class_='apply_now_bank_name')
    credit_needed = credit_div.find(class_='credit_needed')
    card_title = card.find('h2')
    mid_detail = card.find(class_="midDetail")
    
    # Determine long or medium description
    if mid_detail and mid_detail.find(class_="longDescription"):
        card_attributes = mid_detail.find("ul").find_all("li")
    else:
        card_attributes = card.find("ul")
        if card_attributes:
            card_attributes = card_attributes.find_all("li")

    # Process card attributes
    processed_card_attributes = ""
    if card_attributes:
        processed_card_attributes = "\n - ".join(
            attr.get_text(strip=True).replace("\n", "") for attr in card_attributes
        )

    # Create CardScrapeSchema object for each card
    card_data = CRSchema(
        name=card_title.get_text(strip=True),
        description_used=description_used,
        unparsed_issuer=issuer.get_text(strip=True) if issuer else "Unknown",
        unparsed_credit_needed=credit_needed.get_text(strip=True) if credit_needed else "Unknown",
        unparsed_card_attributes=processed_card_attributes
    )

    out_card_list.append(card_data)

# Convert list of CardScrapeSchema objects to a dictionary
card_data_dict = {card.name: card.dict() for card in out_card_list}

# Save the dictionary as a pickle file with a timestamped filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"{timestamp}_cardratings_dict.pkl"
with open(filename, 'wb') as file:
    pickle.dump(card_data_dict, file)

print(f"Data saved to {filename}")