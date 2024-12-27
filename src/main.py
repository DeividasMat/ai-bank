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

import argparse
from datetime import datetime
import json

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
    
    Args:
        ticker (str): Stock ticker symbol
        start_date (str): Analysis start date
        end_date (str): Analysis end date
        portfolio (dict): Current portfolio state
        show_reasoning (bool): Whether to show detailed agent reasoning
        generate_report (bool): Whether to generate a PDF report
        
    Returns:
        dict: Final state containing all agent messages and analysis
    """
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

if __name__ == "__main__":
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='Run the hedge fund trading system')
    parser.add_argument('--ticker', type=str, required=True, help='Stock ticker symbol')
    parser.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD). Defaults to 3 months before end date')
    parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD). Defaults to today')
    parser.add_argument('--show-reasoning', action='store_true', help='Show reasoning from each agent')
    parser.add_argument('--report', action='store_true', help='Generate PDF report')
    
    args = parser.parse_args()
    
    # Only create reports directory if report generation is requested
    if args.report:
        ensure_report_directory()
    
    # Validate and set dates
    if args.start_date:
        validate_date(args.start_date, "Start date")
    if args.end_date:
        validate_date(args.end_date, "End date")
        
    # Get default dates if not provided
    if not args.start_date or not args.end_date:
        default_start, default_end = get_default_dates()
        args.start_date = args.start_date or default_start
        args.end_date = args.end_date or default_end
    
    # Initialize portfolio
    portfolio = {
        "cash": 100000.0,  # $100,000 initial cash
        "stock": 0         # No initial stock position
    }
    
    try:
        # Run the analysis
        result = run_hedge_fund(
            ticker=args.ticker,
            start_date=args.start_date,
            end_date=args.end_date,
            portfolio=portfolio,
            show_reasoning=args.show_reasoning,
            generate_report=args.report
        )
        
        # Extract and display report information if generated
        if args.report:
            report_message = next(msg for msg in result["messages"] if msg.name == "report_agent")
            report_file = json.loads(report_message.content)["report_file"]
            print(f"\nPDF Report generated: {report_file}")
        
        # Extract portfolio decision
        portfolio_message = next(msg for msg in result["messages"] if msg.name == "portfolio_management")
        
        # Display results
        print("\nFinal Portfolio Decision:")
        print(portfolio_message.content)
        
    except Exception as e:
        print(f"\nError during analysis: {str(e)}")
        raise