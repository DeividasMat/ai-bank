from typing import Dict, Any, List
from datetime import datetime
from src.agents.search_agent import SearchAgent
from dotenv import load_dotenv
import os
from pathlib import Path
from openai import OpenAI
import json

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

    def get_latest_company_name(self) -> str:
        """Get company name from the latest JSON file using GPT-4."""
        try:
            # Get the absolute path to the extracted_data directory
            base_dir = Path.cwd()
            json_dir = base_dir / "document_processing" / "extracted_data"
            
            print(f"\nLooking for JSON files in: {json_dir}")
            
            if not json_dir.exists():
                raise FileNotFoundError(f"Directory not found: {json_dir}")
            
            json_files = list(json_dir.glob("*.json"))
            if not json_files:
                raise FileNotFoundError(f"No JSON files found in {json_dir}")
            
            # Get the most recent JSON file
            latest_json = max(json_files, key=lambda x: x.stat().st_mtime)
            print(f"Found latest JSON file: {latest_json}")
            
            # Read the JSON content
            with open(latest_json, 'r') as f:
                data = json.load(f)
            
            def extract_with_gpt4(json_data: dict) -> str:
                """Helper function to extract company name using GPT-4."""
                messages = [
                    {
                        "role": "system",
                        "content": """You are a financial data analyzer. Your task is to find the actual company name 
                        in this financial data. Return ONLY the real company name, no other text. 
                        If you see 'company_name' as a literal string, ignore it and find the actual company name 
                        from the financial data."""
                    },
                    {
                        "role": "user",
                        "content": f"""Find the actual company name from this financial data. 
                        Return ONLY the company name, nothing else.
                        Ignore any literal 'company_name' strings and find the real company name.
                        
                        JSON Data:
                        {json.dumps(json_data, indent=2)}"""
                    }
                ]
                
                client = OpenAI()
                response = client.chat.completions.create(
                    model="gpt-4",
                    messages=messages,
                    temperature=0,
                    max_tokens=50
                )
                
                return response.choices[0].message.content.strip()
            
            # First attempt with direct access
            company_name = data.get("company_name", "")
            print(f"\nInitially found name: {company_name}")
            
            # If we got "company_name" or similar invalid values, try GPT-4
            if not company_name or company_name.lower() in ["company_name", "unknown", "none", "company"]:
                print("Initial name invalid, trying GPT-4 extraction...")
                company_name = extract_with_gpt4(data)
                print(f"GPT-4 extracted name: {company_name}")
                
                # If still invalid, try one more time with different prompt
                if company_name.lower() in ["company_name", "unknown", "none", "company"]:
                    print("First GPT-4 attempt invalid, trying again with financial data focus...")
                    messages = [
                        {
                            "role": "system",
                            "content": """You are a financial document analyzer. Look through the financial statements,
                            metrics, and data to identify the actual company this data belongs to. Return ONLY the 
                            company name, no other text."""
                        },
                        {
                            "role": "user",
                            "content": f"What company does this financial data belong to?\n{json.dumps(data, indent=2)}"
                        }
                    ]
                    
                    response = client.chat.completions.create(
                        model="gpt-4",
                        messages=messages,
                        temperature=0,
                        max_tokens=50
                    )
                    
                    company_name = response.choices[0].message.content.strip()
                    print(f"Second GPT-4 attempt extracted: {company_name}")
            
            # Final validation
            if not company_name or company_name.lower() in ["company_name", "unknown", "none", "company"]:
                raise ValueError(f"Could not extract valid company name after multiple attempts")
            
            print(f"\nFinal extracted company name: {company_name}")
            return company_name
                
        except Exception as e:
            print(f"\nError getting company name: {e}")
            print(f"Current working directory: {Path.cwd()}")
            raise

    def get_associated_names(self, company_name: str) -> list:
        """Get list of all associated company names and tables."""
        try:
            print(f"\nFinding all associated names for {company_name}...")
            
            messages = [
                {
                    "role": "system",
                    "content": "You are a company data analyst. List all associated names and tables for the given company."
                },
                {
                    "role": "user",
                    "content": f"""
                    List of tables and associated names for {company_name}.
                    
                    Please provide:
                    1. List all table names from financial statements
                    2. List all associated company names, including:
                       - Legal entity names
                       - Trading names
                       - Brand names
                       - Subsidiary names
                       - Parent company names
                       - Historical names
                       
                    Return ONLY the names in a clear list format, no additional text.
                    """
                }
            ]
            
            print("Getting associated names...")
            response = self.client.chat.completions.create(
                model="llama-3.1-sonar-large-128k-online",
                messages=messages
            )
            
            names = response.choices[0].message.content.strip().split('\n')
            names = [name.strip() for name in names if name.strip()]
            print("\nFound associated names:")
            for name in names:
                print(f"- {name}")
            
            return names
            
        except Exception as e:
            print(f"Error getting associated names: {e}")
            raise

    def search_articles(self, company_name: str = None) -> None:
        """Search for negative news and lawsuits using Perplexity API."""
        try:
            # Get company name if not provided
            if company_name is None:
                company_name = self.get_latest_company_name()
            
            # First get all associated names
            associated_names = self.get_associated_names(company_name)
            
            print(f"\nSearching for negative news and lawsuits...")
            print("==================================================\n")
            
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a thorough investigative researcher. Find and analyze negative news, "
                        "lawsuits, controversies, and potential risks about the specified company and its associated entities. "
                        "Be comprehensive and factual in your findings."
                    )
                },
                {
                    "role": "user",
                    "content": f"""
                    Research and analyze negative information about the following company names:
                    {json.dumps(associated_names, indent=2)}
                    
                    For each name/entity, analyze:
                    1. Active and past lawsuits
                    2. Legal troubles and investigations
                    3. Regulatory violations and fines
                    4. Corporate scandals or controversies
                    5. Environmental violations
                    6. Labor disputes
                    7. Customer complaints and product issues
                    8. Financial irregularities or concerns
                    
                    Please provide:
                    - Which company name/entity each issue relates to
                    - Specific dates and details of incidents
                    - Status of legal proceedings
                    - Financial impact when available
                    - Sources and citations for each issue
                    - Current status of each situation
                    
                    Focus on factual information and credible sources only.
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
            raw_response_dir.mkdir(parents=True, exist_ok=True)
            raw_response_path = raw_response_dir / f"raw_response_{timestamp}.txt"
            
            with open(raw_response_path, 'w') as f:
                f.write(response.choices[0].message.content)
            
            print(f"✓ Saved raw response to: {raw_response_path}")
            
        except Exception as e:
            print(f"Error in search_articles: {e}")
            raise

# Create the agent instance
market_research_agent = MarketResearch() 