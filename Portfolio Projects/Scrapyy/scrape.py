import pandas as pd
df = pd.read_excel('input.xlsx')  

print(df.head())

from bs4 import BeautifulSoup
import requests

def extract_article_content(url):
    try:
        # Fetch the content of the URL
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad status codes
        
        # Parse the content with BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find the article title and text
        title = soup.find('h1').get_text()  
        article_body = soup.find('div', class_='td-post-content tagdiv-type')
        text = article_body.get_text(separator='\n') if article_body else 'No article content found'
        
        return title + '\n' + text
    except Exception as e:
        print(f"Failed to fetch {url} due to {e}")
        return None
for index, row in df.iterrows():
    content = extract_article_content(row['URL'])
    if content:
        with open(f"{row['URL_ID']}.txt", 'w', encoding='utf-8') as file:
            file.write(content)
