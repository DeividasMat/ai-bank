import math
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from agents.state import AgentState, show_agent_reasoning
from tools.api import prices_to_df
import json
import ast

def risk_management_agent(state: AgentState):
    """Evaluates portfolio risk and PE investment risks using GPT-4."""
    show_reasoning = state["metadata"]["show_reasoning"]
    pe_analysis = state.get("pe_analysis", {})
    
    # Initialize GPT-4
    gpt4 = ChatOpenAI(model="gpt-4o", temperature=0)
    
    # Create PE Risk Analysis Prompt
    pe_risk_prompt = ChatPromptTemplate.from_template("""
    You are a senior risk management professional specializing in Private Equity investments with over 20 years of experience.
    Analyze the following PE ratios and metrics to provide a comprehensive risk assessment.

    Financial Ratios:
    {pe_ratios}

    Please provide a detailed risk analysis covering:

    1. Leverage Risk Assessment
    - Analyze Debt/EBITDA ratio implications
    - Evaluate debt sustainability
    - Assess refinancing risks
    - Provide industry-specific context for leverage levels

    2. Liquidity Risk Analysis
    - Evaluate current ratio and working capital
    - Assess cash flow adequacy
    - Identify potential liquidity constraints
    - Compare to industry standards

    3. Capital Structure Risk
    - Analyze equity ratio implications
    - Evaluate financial flexibility
    - Assess potential for financial distress
    - Compare to peer group capital structures

    4. Debt Service Risk
    - Analyze DSCR implications
    - Evaluate debt service capability
    - Identify potential debt service constraints
    - Compare to lending covenants

    5. Overall Risk Rating
    - Provide a risk rating (Low/Medium/High) for each category
    - Calculate an overall risk score (1-10)
    - List key risk factors in order of priority
    - Suggest specific risk mitigation strategies

    Format your response as JSON with the following structure:
    {
        "leverage_risk": {
            "rating": "Low/Medium/High",
            "score": 1-10,
            "analysis": "detailed analysis",
            "mitigation": ["list of strategies"]
        },
        "liquidity_risk": {...},
        "capital_structure_risk": {...},
        "debt_service_risk": {...},
        "overall_assessment": {
            "total_risk_score": 1-10,
            "key_risks": ["prioritized list"],
            "recommendations": ["specific actions"],
            "monitoring_requirements": ["list of metrics to track"]
        }
    }
    """)

    # Generate PE risk analysis
    try:
        pe_risk_analysis = gpt4.invoke(
            pe_risk_prompt.format_messages(
                pe_ratios=json.dumps(pe_analysis, indent=2)
            )
        ).content
        
        # Parse the GPT-4 response
        pe_risk_results = json.loads(pe_risk_analysis)
        
        # Calculate traditional market risk metrics
        prices_df = prices_to_df(state["data"]["prices"])
        returns = prices_df['close'].pct_change().dropna()
        volatility = returns.std() * (252 ** 0.5)
        var_95 = returns.quantile(0.05)
        max_drawdown = (prices_df['close'] / prices_df['close'].cummax() - 1).min()

        # Combine PE and market risk metrics
        risk_assessment = {
            "pe_risk_analysis": pe_risk_results,
            "market_risk_metrics": {
                "volatility": float(volatility),
                "value_at_risk_95": float(var_95),
                "max_drawdown": float(max_drawdown)
            },
            "trading_action": "hold" if pe_risk_results["overall_assessment"]["total_risk_score"] >= 7 else "proceed",
            "max_position_size": calculate_position_size(pe_risk_results["overall_assessment"]["total_risk_score"])
        }

        # Create the risk management message
        message = HumanMessage(
            content=json.dumps(risk_assessment),
            name="risk_management_agent",
        )

        if show_reasoning:
            show_agent_reasoning(risk_assessment, "Risk Management Agent")

        # Update state with risk analysis
        state["risk_analysis"] = risk_assessment
        
        return {"messages": state["messages"] + [message]}

    except Exception as e:
        print(f"Error in risk analysis: {str(e)}")
        raise e

def calculate_position_size(risk_score: float) -> float:
    """Calculate maximum position size based on risk score."""
    # Base position size of $100,000
    base_size = 100000
    
    # Risk adjustment factor (reduces position size as risk increases)
    risk_factor = max(0.1, 1 - (risk_score / 10))
    
    return base_size * risk_factor

