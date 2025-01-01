from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import os
import time
import json
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
import logging
from pathlib import Path
from langchain_openai import ChatOpenAI
import PyPDF2
from langchain.prompts import ChatPromptTemplate
from PyPDF2 import PdfWriter

class FinancialDocumentProcessor:
    """Processes financial documents and extracts key information using gpt-4o."""
    
    def __init__(self, model_name: str = "gpt-4o"):
        self.base_dir = Path("document_processing")
        self.setup_directories()
        self.setup_logging()
        self.llm = ChatOpenAI(model_name=model_name)

    def setup_directories(self):
        """Create necessary directory structure."""
        directories = {
            "upload": self.base_dir / "upload",
            "processing": self.base_dir / "processing",
            "completed": self.base_dir / "completed",
            "failed": self.base_dir / "failed",
            "logs": self.base_dir / "logs",
            "extracted": self.base_dir / "extracted_data",
            "financial_pages": self.base_dir / "financial_pages"
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

    def find_financial_pages(self, pdf_text: str) -> List[int]:
        """STEP 1: Use GPT-4o to identify pages containing financial statements."""
        try:
            print("\n" + "="*50)
            print("STEP 1: IDENTIFYING FINANCIAL PAGES")
            print("="*50)
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a financial expert. Find all pages that contain financial tables and statements."),
                ("user", """Please identify ALL pages that contain:
                - Balance Sheet
                - Income Statement
                - Cash Flow Statement
                - Notes to Financial Statements
                - Any supporting financial tables

                Return ONLY the page numbers where these appear.

                Document content:
                {pdf_text}""")
            ])
            
            response = self.llm.invoke(prompt.format(pdf_text=pdf_text))
            
            print("\nGPT-4o Response:")
            print("-" * 40)
            print(response.content)
            
            # Extract numbers from the response
            import re
            numbers = re.findall(r'\b\d+\b', response.content)
            financial_pages = [int(num) for num in numbers]
            
            if not financial_pages:
                raise ValueError("No page numbers found in GPT-4o response")
                
            print(f"\nIdentified financial pages: {financial_pages}")
            return financial_pages
                
        except Exception as e:
            self.logger.error(f"Error identifying financial pages: {e}")
            raise

    def create_financial_pages_pdf(self, input_path: str, pages: List[int], output_filename: str) -> str:
        """STEP 2: Create a new PDF with only the financial statement pages."""
        try:
            print("\n" + "="*50)
            print("STEP 2: CREATING NEW PDF WITH FINANCIAL PAGES")
            print("="*50)
            
            reader = PyPDF2.PdfReader(input_path)
            writer = PdfWriter()
            
            print("\nExtracting these pages:", pages)
            for page_num in pages:
                idx = page_num - 1
                if 0 <= idx < len(reader.pages):
                    writer.add_page(reader.pages[idx])
                    print(f"Added page {page_num}")
            
            # Create a new filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"financial_statements_{timestamp}.pdf"
            output_path = self.directories["financial_pages"] / output_filename
            
            with open(output_path, "wb") as output_file:
                writer.write(output_file)
                
            print(f"\nCreated new PDF with financial pages: {output_path}")
            print("\nNew PDF contains only the selected financial statement pages.")
            
            return str(output_path)
            
        except Exception as e:
            self.logger.error(f"Error creating financial pages PDF: {e}")
            raise

    def extract_from_financial_pdf(self, financial_pdf_path: str) -> Dict[str, Any]:
        """STEP 3: Extract data using GPT-4o."""
        try:
            print("\n" + "="*50)
            print("STEP 3: EXTRACTING DATA USING GPT-4o")
            print("="*50)
            
            # Read the financial pages PDF
            with open(financial_pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                financial_text = ""
                for page in reader.pages:
                    financial_text += page.extract_text() + "\n"
            
            print("\nSending to GPT-4o for extraction...")
            
            # Simple extraction prompt
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a financial expert."),
                ("user", "Extract financial information from this file and format it as JSON:\n\n{text}")
            ])
            
            response = self.llm.invoke(prompt.format(text=financial_text))
            
            try:
                # Clean the response
                cleaned_response = response.content.strip()
                if not cleaned_response.startswith('{'):
                    start = cleaned_response.find('{')
                    if start != -1:
                        cleaned_response = cleaned_response[start:]
                        end = cleaned_response.rfind('}') + 1
                        cleaned_response = cleaned_response[:end]
                
                extracted_data = json.loads(cleaned_response)
                
                # Save the JSON
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = self.directories["extracted"] / f"financial_data_{timestamp}.json"
                
                with open(output_path, 'w') as f:
                    json.dump(extracted_data, f, indent=2)
                
                print("\nExtracted Data:")
                print(json.dumps(extracted_data, indent=2))
                print(f"\nSaved to: {output_path}")
                
                return extracted_data
                
            except json.JSONDecodeError as e:
                print("\nError: Invalid JSON in response")
                print("\nRaw response:")
                print(cleaned_response)
                raise ValueError(f"Could not parse JSON: {e}")
                
        except Exception as e:
            self.logger.error(f"Error extracting from financial PDF: {e}")
            raise

    def process_document(self, file_path: str) -> Dict[str, Any]:
        """Process document in three steps."""
        # Step 1: Read PDF and identify financial pages
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for i, page in enumerate(reader.pages):
                text += f"\n=== Page {i+1} ===\n{page.extract_text()}\n"
        
        financial_pages = self.find_financial_pages(text)

        # Step 2: Create financial pages PDF
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        financial_pdf_name = f"financial_statements_{timestamp}.pdf"
        financial_pdf_path = self.create_financial_pages_pdf(
            file_path,
            financial_pages,
            financial_pdf_name
        )

        # Step 3: Extract data
        extracted_data = self.extract_from_financial_pdf(financial_pdf_path)
        
        # Print completion message
        print("\n" + "="*50)
        print("PROCESSING COMPLETE")
        print("="*50)
        print(f"1. Original PDF: {file_path}")
        print(f"2. Financial Pages PDF: {financial_pdf_path}")
        print(f"3. Extracted Data: {json.dumps(extracted_data, indent=2)}")
        
        # Return the data without raising any errors
        return extracted_data

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