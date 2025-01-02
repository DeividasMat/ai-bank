from typing import Dict, Any, List
from datetime import datetime
from src.agents.search_agent import SearchAgent
from dotenv import load_dotenv
import os
from pathlib import Path
from openai import OpenAI

# Load environment variables
load_dotenv()
PERPLEXITY_API_KEY = os.getenv('PERPLEXITY_API_KEY')

class MarketResearch:
    def __init__(self):
        """Initialize MarketResearch with Perplexity API client."""
        perplexity_api_key = os.getenv("PERPLEXITY_API_KEY")
        if not perplexity_api_key:
            raise ValueError("PERPLEXITY_API_KEY environment variable not set")
        
        self.client = OpenAI(
            api_key=perplexity_api_key,
            base_url="https://api.perplexity.ai"
        )

    def analyze(self, financial_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get market research data for the report"""
        try:
            # Get company name from financial data
            company_name = list(financial_data.keys())[0].replace('_', ' ')
            
            # Get negative news and lawsuits using Perplexity
            citations = self.search_agent.search_articles(company_name)
            
            # Format data for the report
            market_data = {
                "company": company_name,
                "market_analysis": {
                    "negative_news": [],
                    "lawsuits": [],
                    "regulatory_issues": [],
                    "competition_threats": []
                },
                "sources": citations
            }
            
            # Categorize findings
            for citation in citations:
                if 'lawsuit' in citation.get('title', '').lower():
                    market_data["market_analysis"]["lawsuits"].append(citation)
                elif 'regulation' in citation.get('title', '').lower():
                    market_data["market_analysis"]["regulatory_issues"].append(citation)
                elif 'competitor' in citation.get('title', '').lower():
                    market_data["market_analysis"]["competition_threats"].append(citation)
                else:
                    market_data["market_analysis"]["negative_news"].append(citation)
            
            return market_data

        except Exception as e:
            print(f"Error in market research: {e}")
            return {}

    def search_articles(self, company_name: str) -> None:
        """Search for articles using Perplexity API."""
        try:
            print(f"\nSearching for articles about {company_name}...")
            print("==================================================\n")
            
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a financial research assistant. Please find and analyze "
                        "recent news articles and market information about the specified company."
                    )
                },
                {
                    "role": "user",
                    "content": f"""
                    Find recent news articles and market analysis about {company_name}.
                    Focus on:
                    1. Financial performance and metrics
                    2. Market position and competitive analysis
                    3. Strategic initiatives and developments
                    4. Industry trends and market conditions
                    5. Regulatory environment and risks
                    
                    Please provide detailed information with specific sources, dates, and citations.
                    """
                }
            ]
            
            print("Calling Perplexity API...")
            print("This may take a few moments...")
            
            response = self.client.chat.completions.create(
                model="llama-3.1-sonar-large-128k-online",
                messages=messages
            )
            
            # Save raw response
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            raw_response_dir = Path("document_processing/articles/raw_responses")
            raw_response_path = raw_response_dir / f"raw_response_{timestamp}.txt"
            
            with open(raw_response_path, 'w') as f:
                f.write(response.choices[0].message.content)
            
            print(f"✓ Saved raw response to: {raw_response_path}")
            
        except Exception as e:
            print(f"Error in search_articles: {e}")
            raise

# Create the agent instance
market_research_agent = MarketResearch() 