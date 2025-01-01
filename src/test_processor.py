from agents.document_processor import FinancialDocumentProcessor
from pathlib import Path
import sys

def test_processor():
    print("Starting processor test...")
    
    # Initialize processor
    processor = FinancialDocumentProcessor(model_name="gpt-4o")
    print(f"Processor initialized with directories:")
    for name, path in processor.directories.items():
        print(f"- {name}: {path}")
    
    # Check if test file exists
    test_file = Path("document_processing/upload/test.pdf")
    if not test_file.exists():
        print(f"\nError: No test file found at {test_file}")
        print("Please place a PDF file in the upload directory")
        return
    
    print(f"\nProcessing test file: {test_file}")
    
    # Process the document
    result = processor.process_document(str(test_file))
    
    print("\nProcessing result:")
    print(f"Status: {result.get('status', 'unknown')}")
    if result.get('error'):
        print(f"Error: {result['error']}")
    elif result.get('data'):
        print("Extracted data:")
        print(result['data'])
    
    print("\nCheck the logs for more details:")
    print(f"Log file: {processor.directories['logs']}/processing.log")

if __name__ == "__main__":
    test_processor() 