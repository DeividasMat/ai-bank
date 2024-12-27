import os

def ensure_report_directory():
    """Ensures the reports directory exists."""
    os.makedirs("reports", exist_ok=True) 