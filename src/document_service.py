from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time
import logging
from pathlib import Path
from agents.document_processor import FinancialDocumentProcessor
from agents.pe_analysis import PEAnalysisAgent
from typing import Dict, Any
from tqdm import tqdm
from document_processor import DocumentProcessor
from src.agents.search_agent import SearchAgent
from src.config import PERPLEXITY_API_KEY

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

def process_document(file_path: str) -> Dict[str, Any]:
    processor = DocumentProcessor()
    result = processor.process_document(file_path)
    
    # After document processing, run the search agent
    try:
        extracted_dir = Path("document_processing/extracted")
        json_files = list(extracted_dir.glob("*.json"))
        if json_files:
            latest_json = max(json_files, key=lambda x: x.stat().st_mtime)
            search_agent = SearchAgent(perplexity_api_key=PERPLEXITY_API_KEY)
            search_agent.process(str(latest_json))
    except Exception as e:
        print(f"Article search error: {e}")
    
    return result

def main():
    try:
        # Step 1: Process the document
        file_path = "document_processing/upload/Netflix-10-K-01262024.pdf"
        print("\nProcessing document...")
        extracted_data = process_document(file_path)
        
        # Get the path of the last created JSON file in the extracted directory
        extracted_dir = Path("document_processing/extracted")
        json_files = list(extracted_dir.glob("*.json"))
        latest_json = max(json_files, key=lambda x: x.stat().st_mtime)
        
        # Step 2: Search for articles
        print("\nStarting article search...")
        search_agent = SearchAgent(perplexity_api_key=PERPLEXITY_API_KEY)
        articles_path = search_agent.process(str(latest_json))
        
        print("\nProcess completed successfully!")
        print(f"1. Financial data: {latest_json}")
        print(f"2. Articles data: {articles_path}")

    except Exception as e:
        print(f"Error in processing: {str(e)}")
        raise

if __name__ == "__main__":
    # Set watch_mode=True if you want to keep watching for new files
    run_processor(watch_mode=False)
    main() 