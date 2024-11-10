import pickle
from datetime import datetime
from bs4 import BeautifulSoup

# Load HTML content from file
file_name = "../sites/Compare Credit Cards & Apply Online _ Credit Karma.html"
with open(file_name, 'r', encoding='utf-8') as file:
    html_content = file.read()

soup = BeautifulSoup(html_content, 'html.parser')

# Define a list of common issuers to help identify issuer from card name or description
issuers = [
    "American Express", "Bank of America", "Barclays", "Capital One", "Chase",
    "Citi", "Discover", "Synchrony", "U.S. Bank", "Wells Fargo"
]

# Initialize dictionary to store card data
cards_data = {}

# Locate all credit card containers by the common class 'ck-card'
card_containers = soup.find_all('div', class_='ck-card')

# Loop through each card container
for card in card_containers:
    # Initialize fields for each card
    name = unparsed_issuer = unparsed_credit_needed = unparsed_card_attributes = None

    # Extract the card name
    name = card.find('h2', class_='lh-copy mv0 f4 f3-ns').get_text(strip=True) if card.find('h2', class_='lh-copy mv0 f4 f3-ns') else None
    
    # Skip if no name is found (or if the card is already added)
    if not name or name in cards_data:
        continue

    # Identify the issuer based on name keywords
    for issuer in issuers:
        if issuer.lower() in name.lower():
            unparsed_issuer = issuer
            break

    # Attempt to gather all descriptive text (attributes) and combine
    descriptions = []
    
    # Add the tagline/description if available
    description_tag = card.find('span', class_='f4 lh-copy ck-green-70 b')
    if description_tag:
        descriptions.append(description_tag.get_text(strip=True))

    # Extract features like rewards rate, annual fee, and welcome bonus
    features = card.find_all('li', class_='flex flex-column bb b--light-gray pv2 ph3')
    for feature in features:
        feature_text = feature.get_text(" ", strip=True)
        descriptions.append(feature_text)
        
        # Check for credit needed in the feature text
        if unparsed_credit_needed is None and any(term in feature_text for term in ["Good", "Excellent", "Fair", "Poor"]):
            unparsed_credit_needed = feature_text

    # Collect additional attributes from the nested lists in the collapser section
    collapser_items = card.find_all('div', class_='ck-link-parent lh-copy ml2 f5 f4-ns')
    for item in collapser_items:
        descriptions.append(item.get_text(strip=True))

    # Collect attributes from the `list` class within nested `<ul>` tags
    list_items = card.find_all('ul', class_='list')
    for ul in list_items:
        list_elements = ul.find_all('li')
        for li in list_elements:
            list_text = li.get_text(" ", strip=True)
            descriptions.append(list_text)

    # Combine all descriptions into a single string for `unparsed_card_attributes`
    unparsed_card_attributes = ". ".join(descriptions)

    # Store the extracted information in the dictionary
    cards_data[name] = {
        "issuer": unparsed_issuer,
        "attributes": unparsed_card_attributes
    }

# Get the current timestamp and format it
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Define the pickle file name with the timestamp
pickle_filename = f"{timestamp}_credit_karma_dict.pkl"

# Save the dictionary to a pickle file
with open(pickle_filename, 'wb') as pkl_file:
    pickle.dump(cards_data, pkl_file)

print(f"Data saved to {pickle_filename}")
