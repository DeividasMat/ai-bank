import json
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
import logging
from openai import OpenAI

class SearchAgent:
    def __init__(self, perplexity_api_key: str):
        try:
            self.client = OpenAI(
                api_key=perplexity_api_key,
                base_url="https://api.perplexity.ai"
            )
            # Test API connection
            test_response = self.client.chat.completions.create(
                model="llama-3.1-sonar-large-128k-online",
                messages=[{"role": "user", "content": "Test connection"}],
            )
            print("\n✓ Perplexity API connection successful")
        except Exception as e:
            print(f"\n✗ Perplexity API Error: {str(e)}")
            raise

        self.logger = logging.getLogger(__name__)
        
        # Create directories
        self.directories = {
            "articles": Path("document_processing/articles"),
            "raw_responses": Path("document_processing/articles/raw_responses")
        }
        for dir_path in self.directories.values():
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"✓ Created directory: {dir_path}")

    def get_company_name(self, json_path: str) -> str:
        """Extract company name from the financial data JSON."""
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            # Get the first key (company name) from the JSON
            company_name = list(data.keys())[0].replace('_', ' ')
            print(f"Extracted company name: {company_name}")
            return company_name
        except Exception as e:
            self.logger.error(f"Error extracting company name: {e}")
            raise

    def search_articles(self, company_name: str) -> List[Dict[str, Any]]:
        """Search for articles using Perplexity API."""
        try:
            print(f"\nSearching for articles about {company_name}...")
            print("="*50)
            
            # Create messages for the API
            messages = [
                {
                    "role": "system",
                    "content": "You are a financial analyst focused on identifying risks, challenges, and negative news about companies. Search for recent critical coverage and concerning developments."
                },
                {
                    "role": "user",
                    "content": f"""Find and analyze negative news articles about {company_name} from 2024.
                    Focus specifically on:
                    1. Stock price drops or market value losses
                    2. Subscriber/customer losses
                    3. Competition taking market share
                    4. Failed projects or initiatives
                    5. Management controversies
                    6. Layoffs or cost cutting
                    7. Regulatory problems
                    8. Poor financial results
                    9. Analyst downgrades
                    10. Any other negative developments

                    For each article provide:
                    ARTICLE
                    Title: [exact negative headline]
                    Date: [2024 date]
                    Source: [publication name]
                    URL: [full URL]
                    Summary: [focus on the negative aspects and problems]
                    END

                    Find at least 15 different negative articles from 2024.
                    Include only factual criticism, not speculation."""
                }
            ]

            # Make API call
            print("\nCalling Perplexity API...")
            print("This may take a few moments...")
            response = self.client.chat.completions.create(
                model="llama-3.1-sonar-large-128k-online",
                messages=messages,
            )
            
            # Save raw response
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            raw_path = self.directories["raw_responses"] / f"raw_response_{timestamp}.txt"
            with open(raw_path, 'w', encoding='utf-8') as f:
                f.write(response.choices[0].message.content)
            print(f"✓ Saved raw response to: {raw_path}")
            
            # Extract content and parse articles
            content = response.choices[0].message.content
            articles = self._parse_articles(content)
            print(f"\n✓ Found {len(articles)} articles")
            
            # Print article titles
            print("\nArticles found:")
            print("-" * 50)
            for i, article in enumerate(articles, 1):
                print(f"{i}. {article['title']}")
                print(f"   Source: {article['source']}")
                print(f"   Date: {article['date']}")
                print("-" * 50)
            
            return articles

        except Exception as e:
            self.logger.error(f"Error in article search: {e}")
            print(f"\n✗ API Error: {str(e)}")
            raise

    def _parse_articles(self, content: str) -> List[Dict[str, Any]]:
        """Parse the API response into structured article data."""
        articles = []
        current_article = {}
        
        for line in content.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            if line == 'ARTICLE':
                current_article = {}
            elif line == 'END':
                if current_article:
                    articles.append(current_article)
            elif ':' in line:
                key, value = line.split(':', 1)
                current_article[key.strip()] = value.strip()
            
            if current_article.get('summary'):
                current_article['summary'] += ' ' + line
        
        return articles

    def save_articles(self, company_name: str, articles: List[Dict[str, Any]]) -> str:
        """Save articles to a JSON file."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{company_name.lower().replace(' ', '_')}_articles_{timestamp}.json"
            output_path = self.directories["articles"] / filename
            
            data = {
                "company_name": company_name,
                "articles": articles,
                "total_articles": len(articles),
                "extraction_date": datetime.now().isoformat()
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            print(f"\n✓ Saved {len(articles)} articles to:")
            print(f"  {output_path}")
            return str(output_path)

        except Exception as e:
            self.logger.error(f"Error saving articles: {e}")
            raise

    def process(self, json_path: str) -> str:
        """Main process to find and save articles."""
        try:
            print("\nStarting Article Search")
            print("="*50)
            
            # Get company name
            company_name = self.get_company_name(json_path)
            print(f"\nSearching articles for: {company_name}")
            
            # Search for articles
            articles = self.search_articles(company_name)
            print(f"\nFound {len(articles)} articles")
            
            # Save articles
            output_path = self.save_articles(company_name, articles)
            print(f"\nSaved articles to: {output_path}")
            
            return output_path

        except Exception as e:
            self.logger.error(f"Error in search process: {e}")
            raise 