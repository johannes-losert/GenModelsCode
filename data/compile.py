import csv
import asyncio
from tqdm import tqdm
from data.openai_structure import generate_credit_card_json
from data.extractors.cardratings.extractor import CardRatingsExtractor
from data.extractors.creditkarma.extractor import CreditKarmaExtractor  
from data.extractors.usnews.extractor import USNewsExtractor

API_KEY = "You ain't getting my key hehe"

extractors = [CardRatingsExtractor(), CreditKarmaExtractor(), USNewsExtractor()]

with open('data.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['dataset', 'input', 'output'])
    for extractor in tqdm(extractors, desc='Extractors'):
        results = extractor.extract()
        dataset = extractor.site_name
        for result in tqdm(results, desc=dataset):
            input = result
            output = asyncio.run(generate_credit_card_json(result.description, API_KEY))
            writer.writerow([dataset, input, output])

