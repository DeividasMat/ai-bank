from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph, START

from agents.fundamentals import fundamentals_agent
from agents.market_data import market_data_agent
from agents.portfolio_manager import portfolio_management_agent
from agents.technicals import technical_analyst_agent
from agents.risk_manager import risk_management_agent
from agents.sentiment import sentiment_agent
from agents.report import report_agent
from agents.state import AgentState
from tools.setup import ensure_report_directory
from agents.pe_analysis import PEAnalysisAgent

import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess

##### Run the Hedge Fund #####
def create_workflow(generate_report: bool = False):
    """Create the workflow graph with optional report generation."""
    workflow = StateGraph(AgentState)

    # Add nodes for each agent
    workflow.add_node("market_data_agent", market_data_agent)
    workflow.add_node("technical_analyst_agent", technical_analyst_agent)
    workflow.add_node("fundamentals_agent", fundamentals_agent)
    workflow.add_node("sentiment_agent", sentiment_agent)
    workflow.add_node("risk_management_agent", risk_management_agent)
    workflow.add_node("portfolio_management_agent", portfolio_management_agent)
    
    if generate_report:
        workflow.add_node("report_agent", report_agent)

    # Set the entry point
    workflow.set_entry_point("market_data_agent")

    # Define the analysis workflow
    workflow.add_edge("market_data_agent", "technical_analyst_agent")
    workflow.add_edge("market_data_agent", "fundamentals_agent")
    workflow.add_edge("market_data_agent", "sentiment_agent")

    workflow.add_edge("technical_analyst_agent", "risk_management_agent")
    workflow.add_edge("fundamentals_agent", "risk_management_agent")
    workflow.add_edge("sentiment_agent", "risk_management_agent")

    workflow.add_edge("risk_management_agent", "portfolio_management_agent")
    
    if generate_report:
        workflow.add_edge("portfolio_management_agent", "report_agent")
        workflow.add_edge("report_agent", END)
    else:
        workflow.add_edge("portfolio_management_agent", END)

    return workflow.compile()

def run_hedge_fund(ticker: str, start_date: str, end_date: str, portfolio: dict, show_reasoning: bool = False, generate_report: bool = False):
    """
    Run the hedge fund analysis pipeline.
    """
    # Check for extracted data first
    extracted_data_path = Path("document_processing/extracted_data")
    json_files = list(extracted_data_path.glob("*.json"))
    extracted_data = None
    
    if json_files:
        # Use the most recent JSON file if multiple exist
        latest_file = max(json_files, key=lambda x: x.stat().st_mtime)
        with open(latest_file, 'r') as f:
            extracted_data = json.load(f)
    
    # Create workflow based on whether report generation is needed
    app = create_workflow(generate_report)
    
    final_state = app.invoke(
        {
            "messages": [
                HumanMessage(
                    content="Make a trading decision based on the provided data.",
                )
            ],
            "data": {
                "ticker": ticker,
                "portfolio": portfolio,
                "start_date": start_date,
                "end_date": end_date,
                "extracted_data": extracted_data  # Add extracted data to the state
            },
            "metadata": {
                "show_reasoning": show_reasoning,
            }
        },
    )
    return final_state

def validate_date(date_str: str, date_name: str) -> None:
    """Validate date string format."""
    try:
        if date_str:
            datetime.strptime(date_str, '%Y-%m-%d')
    except ValueError:
        raise ValueError(f"{date_name} must be in YYYY-MM-DD format")

def get_default_dates():
    """Get default date range (3 months)."""
    end_date = datetime.now()
    start_date = end_date.replace(
        month=end_date.month - 3 if end_date.month > 3 
        else end_date.month + 9,
        year=end_date.year if end_date.month > 3 
        else end_date.year - 1
    )
    return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')

def add_pe_analysis(args, existing_analysis=None):
    """Add PE analysis without interfering with existing functionality"""
    try:
        pe_agent = PEAnalysisAgent()
        
        if existing_analysis:
            # If we have existing analysis, use it
            temp_path = Path("document_processing/extracted_data/temp_analysis.json")
            temp_path.parent.mkdir(parents=True, exist_ok=True)
            with open(temp_path, 'w') as f:
                json.dump(existing_analysis, f, indent=2)
            
            pe_result = pe_agent.analyze_company(str(temp_path))
            
            # Clean up
            if temp_path.exists():
                temp_path.unlink()
                
        elif args.file:
            # Use provided file for standalone PE analysis
            pe_result = pe_agent.analyze_company(args.file)
        else:
            return None

        if "error" not in pe_result:
            print("\n=== Private Equity Analysis ===")
            print("\nFinancial Metrics:")
            print(pe_result["formatted_metrics"])
            print("\nMetrics Analysis:")
            print(pe_result["metrics_summary"])
            
        return pe_result
        
    except Exception as e:
        print(f"\nPE Analysis Error: {str(e)}")
        return {"error": str(e)}

def run_financial_analysis():
    # ... (keep existing financial analysis code)
    
    # Add search service after financial analysis
    try:
        print("\nStarting Article Search...")
        subprocess.run(["poetry", "run", "python", "src/search_service.py"], check=True)
        print("Article Search Complete")
    except Exception as e:
        print(f"Article search error: {e}")

def main():
    print("Starting main process...")
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='AI Bank Analysis Tools')
    parser.add_argument('--report', action='store_true', help='Generate analysis report')
    parser.add_argument('--show-reasoning', action='store_true', help='Show agent reasoning')
    args = parser.parse_args()
    
    print(f"Report flag: {args.report}")
    
    if args.report:
        print("Initializing report generation...")
        try:
            # Create an instance of report agent
            from agents.report import report_agent
            
            # Create initial state
            initial_state = AgentState(
                messages=[],
                data={},
                metadata={"show_reasoning": args.show_reasoning}
            )
            
            # Call report agent directly
            print("Calling report agent...")
            final_state = report_agent(initial_state)
            print("Report agent finished")
            
        except Exception as e:
            print(f"Error during report generation: {str(e)}")
            raise e

        run_financial_analysis()  # This will now include the search

if __name__ == "__main__":
    main()