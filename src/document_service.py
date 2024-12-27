from watchdog.observers import Observer
from agents.document_processor import FinancialDocumentProcessor, DocumentEventHandler
import time
import argparse
import logging
from pathlib import Path

def run_processor(model_name: str = "gpt-4o"):
    """Run the document processing service."""
    print("Initializing document processor...")
    processor = FinancialDocumentProcessor(model_name=model_name)
    event_handler = DocumentEventHandler(processor)
    
    observer = Observer()
    observer.schedule(event_handler, str(processor.directories["upload"]), recursive=False)
    observer.start()
    
    print(f"\n{'='*50}")
    print(f"Document processor is running!")
    print(f"Watching directory: {processor.directories['upload']}")
    print(f"Using model: {model_name}")
    print(f"{'='*50}\n")
    print("Waiting for PDF files... (Press Ctrl+C to stop)")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        print("\nDocument processor stopped")
    
    observer.join()

def main():
    parser = argparse.ArgumentParser(description='Run the document processing service')
    parser.add_argument('--model', type=str, default='gpt-4', help='LLM model to use')
    
    args = parser.parse_args()
    
    run_processor(args.model)

if __name__ == "__main__":
    main() 