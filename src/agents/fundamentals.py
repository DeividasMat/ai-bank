from langchain_core.messages import HumanMessage

from agents.state import AgentState, show_agent_reasoning

import json

##### Fundamental Agent #####
def fundamentals_agent(state):
    """
    Analyze fundamental data including extracted financial statements if available.
    """
    extracted_data = state.data.get("extracted_data")
    
    if extracted_data:
        # Use extracted financial data from documents
        financial_metrics = analyze_extracted_data(extracted_data)
    else:
        # Fall back to API-based data collection
        financial_metrics = collect_fundamental_data(state.data["ticker"])
    
    # Continue with analysis...

def calculate_intrinsic_value(
    free_cash_flow: float,
    growth_rate: float = 0.05,
    discount_rate: float = 0.10,
    terminal_growth_rate: float = 0.02,
    num_years: int = 5,
) -> float:
    """
    Computes the discounted cash flow (DCF) for a given company based on the current free cash flow.
    Use this function to calculate the intrinsic value of a stock.
    """
    # Estimate the future cash flows based on the growth rate
    cash_flows = [free_cash_flow * (1 + growth_rate) ** i for i in range(num_years)]

    # Calculate the present value of projected cash flows
    present_values = []
    for i in range(num_years):
        present_value = cash_flows[i] / (1 + discount_rate) ** (i + 1)
        present_values.append(present_value)

    # Calculate the terminal value
    terminal_value = cash_flows[-1] * (1 + terminal_growth_rate) / (discount_rate - terminal_growth_rate)
    terminal_present_value = terminal_value / (1 + discount_rate) ** num_years

    # Sum up the present values and terminal value
    dcf_value = sum(present_values) + terminal_present_value

    return dcf_value