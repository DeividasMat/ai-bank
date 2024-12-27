from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import os
import time
import json
from datetime import datetime
from typing import Optional
import logging
from pathlib import Path
from langchain_openai import ChatOpenAI

class FinancialDocumentProcessor:
    """Processes financial documents and extracts key information."""
    
    def __init__(self, model_name: str = "gpt-4o"):
        self.base_dir = Path("document_processing")
        self.setup_directories()
        self.llm = ChatOpenAI(model_name=model_name)
        self.setup_logging()

    def setup_directories(self):
        """Create necessary directory structure."""
        directories = {
            "upload": self.base_dir / "upload",
            "processing": self.base_dir / "processing",
            "completed": self.base_dir / "completed",
            "failed": self.base_dir / "failed",
            "logs": self.base_dir / "logs",
            "extracted": self.base_dir / "extracted_data"
        }
        
        for directory in directories.values():
            directory.mkdir(parents=True, exist_ok=True)
            
        self.directories = directories

    def setup_logging(self):
        """Configure logging."""
        log_file = self.directories["logs"] / "processing.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def extract_financial_data(self, text: str) -> dict:
        """Extract financial data using LLM."""
        prompt = """
        Extract key financial information from this document. Focus on:
        1. Financial Statements
           - Balance Sheet items
           - Income Statement items
           - Cash Flow items
        2. Key Metrics and Ratios
        3. Important Notes or Disclosures

        Return the data in a structured JSON format.
        
        Text:
        {text}
        """
        
        try:
            response = self.llm.invoke(prompt.format(text=text[:4000]))  # Limit text size
            return json.loads(response.content)
        except Exception as e:
            self.logger.error(f"Error extracting financial data: {e}")
            return {"error": str(e)}

    def process_document(self, file_path: str) -> dict:
        """Process a financial document and extract information."""
        try:
            # Read PDF content
            from pypdf import PdfReader
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"

            # Extract data
            extracted_data = self.extract_financial_data(text)
            
            # Save extracted data
            filename = Path(file_path).stem
            output_path = self.directories["extracted"] / f"{filename}_data.json"
            with open(output_path, 'w') as f:
                json.dump(extracted_data, f, indent=2)

            return {
                "status": "success",
                "output_file": str(output_path),
                "data": extracted_data
            }

        except Exception as e:
            self.logger.error(f"Error processing document: {e}")
            return {
                "status": "error",
                "error": str(e)
            }

class DocumentEventHandler(FileSystemEventHandler):
    """Handles file system events for document processing."""
    
    def __init__(self, processor: FinancialDocumentProcessor):
        self.processor = processor
        self.logger = processor.logger

    def on_created(self, event):
        if not event.is_directory and event.src_path.lower().endswith('.pdf'):
            self.process_new_document(event.src_path)

    def process_new_document(self, file_path: str):
        """Process a newly uploaded document."""
        try:
            filename = Path(file_path).name
            self.logger.info(f"New document detected: {filename}")

            # Move to processing directory
            processing_path = self.processor.directories["processing"] / filename
            Path(file_path).rename(processing_path)
            
            # Process the document
            result = self.processor.process_document(str(processing_path))

            if result["status"] == "error":
                raise Exception(result["error"])

            # Move to completed directory
            completed_path = self.processor.directories["completed"] / filename
            processing_path.rename(completed_path)

            self.logger.info(f"Successfully processed document: {filename}")
            
        except Exception as e:
            self.logger.error(f"Error processing {file_path}: {str(e)}")
            try:
                # Move to failed directory
                failed_path = self.processor.directories["failed"] / Path(file_path).name
                Path(file_path).rename(failed_path)
            except Exception as move_error:
                self.logger.error(f"Error moving failed file: {str(move_error)}") 