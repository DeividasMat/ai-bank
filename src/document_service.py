import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time
import logging
from pathlib import Path
from agents.document_processor import FinancialDocumentProcessor
from agents.pe_analysis import PEAnalysisAgent
from typing import Dict, Any
from tqdm import tqdm
from src.agents.search_agent import SearchAgent
import sys
from dotenv import load_dotenv
import json
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import PyPDF2
from datetime import datetime

# Load environment variables from .env file
load_dotenv(dotenv_path=Path(__file__).parent.parent / '.env')

# Add src directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
from agents.document_processor import FinancialDocumentProcessor  # This is the correct class name

class DocumentEventHandler(FileSystemEventHandler):
    def __init__(self, processor, pe_analyzer):
        self.processor = processor
        self.pe_analyzer = pe_analyzer
        self.logger = logging.getLogger(__name__)

    def on_created(self, event):
        if event.is_directory:
            return
            
        file_path = Path(event.src_path)
        if file_path.suffix.lower() != '.pdf':
            print(f"Ignoring non-PDF file: {file_path.name}")
            return
            
        print(f"\nProcessing new file: {file_path.name}")
        self.process_document(str(file_path))

    def process_document(self, file_path: str) -> Dict[str, Any]:
        processor = FinancialDocumentProcessor()
        with tqdm(total=100, desc=f"Processing {Path(file_path).name}") as pbar:
            result = processor.process_document(file_path)
            pbar.update(100)
        return result

def run_processor(watch_mode: bool = False):
    print("Starting document processor...")
    
    # Initialize processors
    processor = FinancialDocumentProcessor()
    pe_analyzer = PEAnalysisAgent()
    
    # Set up the event handler and observer
    upload_dir = Path("document_processing/upload")
    if not upload_dir.exists():
        print(f"Creating upload directory: {upload_dir}")
        upload_dir.mkdir(parents=True, exist_ok=True)
    
    # List existing files
    existing_files = list(upload_dir.glob("*.pdf"))
    if existing_files:
        print("\nFound existing PDF files:")
        for file in existing_files:
            print(f"- {file.name}")
    else:
        print("\nNo existing PDF files found in upload directory")
    
    event_handler = DocumentEventHandler(processor, pe_analyzer)
    
    # Process existing files
    for file in existing_files:
        print(f"\nProcessing existing file: {file.name}")
        event_handler.process_document(str(file))
    
    if watch_mode:
        observer = Observer()
        observer.schedule(event_handler, str(upload_dir), recursive=False)
        observer.start()
        
        print(f"\nWatching directory: {upload_dir}")
        print("Waiting for new PDF files... (Press Ctrl+C to stop)")
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            observer.stop()
            print("\nDocument processor stopped")
        
        observer.join()

def extract_company_name(pdf_text: str) -> str:
    """Extract company name using GPT-4."""
    try:
        print("\n" + "="*50)
        print("EXTRACTING COMPANY NAME")
        print("="*50)
        
        llm = ChatOpenAI(model_name="gpt-4", temperature=0)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a financial document analyzer. Your task is to extract the exact company name 
                      from financial documents. Return ONLY the company name, no additional text."""),
            ("user", """Please analyze this text and extract the company name.
                    Return ONLY the company name, nothing else.
                    
                    Document text:
                    {text}""")
        ])
        
        print("\nAnalyzing document for company name...")
        response = llm.invoke(prompt.format(text=pdf_text[:3000]))  # First 3000 chars should contain company name
        
        company_name = response.content.strip()
        print(f"Found company name: {company_name}")
        
        return company_name
        
    except Exception as e:
        print(f"Error extracting company name: {e}")
        return "Unknown Company"

def process_document(file_path: str) -> Dict[str, Any]:
    """Process a document and extract financial data."""
    try:
        print("Processing document...")
        processor = FinancialDocumentProcessor()
        
        # Look for any PDF file in the upload directory
        upload_dir = Path("document_processing/upload")
        if not upload_dir.exists():
            print(f"Creating upload directory: {upload_dir}")
            upload_dir.mkdir(parents=True, exist_ok=True)
            
        pdf_files = list(upload_dir.glob("*.pdf"))
        if not pdf_files:
            print("\nNo PDF files found in upload directory!")
            print(f"Please place your PDF file in: {upload_dir.absolute()}")
            print("Then run this script again.")
            return None
            
        # Use the most recently modified PDF file
        latest_pdf = max(pdf_files, key=lambda x: x.stat().st_mtime)
        print(f"Processing file: {latest_pdf}")
        
        # Extract text for company name
        with open(latest_pdf, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            # Get text from first few pages
            for i in range(min(3, len(reader.pages))):
                text += reader.pages[i].extract_text() + "\n"
        
        # Extract company name using GPT-4
        company_name = extract_company_name(text)
        
        # Process document
        extracted_data = processor.process_document(str(latest_pdf))
        
        # Combine data with company name
        enhanced_data = {
            "company_name": company_name,
            "financial_data": extracted_data,
            "processing_date": datetime.now().isoformat(),
            "source_file": str(latest_pdf.name)
        }
        
        # Save enhanced JSON
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = Path("document_processing/extracted_data") / f"financial_data_{timestamp}.json"
        json_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(json_path, 'w') as f:
            json.dump(enhanced_data, f, indent=2)
            
        print(f"\nSaved enhanced data to: {json_path}")
        return enhanced_data
        
    except Exception as e:
        print(f"Error in processing: {str(e)}")
        raise

def main():
    """Main function to process documents."""
    try:
        # Create extracted_data directory if it doesn't exist
        json_dir = Path("document_processing/extracted_data")
        json_dir.mkdir(parents=True, exist_ok=True)
        
        # Process new document first
        file_path = "document_processing/upload"
        print("\nProcessing new document...")
        extracted_data = process_document(file_path)
        
        if extracted_data:
            print("\nProcessing complete!")
            print(json.dumps(extracted_data, indent=2))
            
            # Now check for JSON files after processing
            json_files = list(json_dir.glob("*.json"))
            if json_files:
                latest_json = max(json_files, key=lambda x: x.stat().st_mtime)
                print(f"\nLatest extracted data saved to: {latest_json}")
            
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    # Set watch_mode=True if you want to keep watching for new files
    run_processor(watch_mode=False)
    main() 