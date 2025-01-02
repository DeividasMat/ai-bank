import os
from pathlib import Path
from dotenv import load_dotenv
from src.agents.search_agent import SearchAgent

# Load environment variables from .env file
load_dotenv()

# Get API key from .env
PERPLEXITY_API_KEY = os.getenv('PERPLEXITY_API_KEY')
if not PERPLEXITY_API_KEY:
    raise ValueError("PERPLEXITY_API_KEY not found in .env file")

def run_search():
    try:
        # Updated path to correct directory
        extracted_dir = Path("document_processing/extracted_data")
        json_files = list(extracted_dir.glob("*.json"))
        if not json_files:
            print("No extracted JSON files found in document_processing/extracted_data")
            return
            
        latest_json = max(json_files, key=lambda x: x.stat().st_mtime)
        print(f"\nFound latest financial data: {latest_json}")
        
        # Run search agent
        search_agent = SearchAgent(perplexity_api_key=PERPLEXITY_API_KEY)
        articles_path = search_agent.process(str(latest_json))
        
        print(f"\nArticles saved to: {articles_path}")
        
    except Exception as e:
        print(f"Search error: {e}")

if __name__ == "__main__":
    run_search() 