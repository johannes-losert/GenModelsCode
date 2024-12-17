from bs4 import BeautifulSoup

# Load the HTML content (replace 'page.html' with your HTML file path)
html_file = 'usnews_links.html'
output_file = 'links.txt'

with open(html_file, 'r', encoding='utf-8') as file:
    soup = BeautifulSoup(file, 'html.parser')

# Find all links
links = soup.find_all('a', href=True)

# Extract HTTP links
http_links = [link['href'] for link in links if link['href'].startswith('http')]

# Save the links to a .txt file
with open(output_file, 'w', encoding='utf-8') as file:
    for link in http_links:
        file.write(link + '\n')

print(f"Extracted {len(http_links)} links and saved to {output_file}")
