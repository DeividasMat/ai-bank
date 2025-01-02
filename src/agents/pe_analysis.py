from typing import Dict, Any
from pathlib import Path
import json
from langchain_openai import ChatOpenAI

class PEAnalysisAgent:
    def __init__(self):
        self.llm = ChatOpenAI(model_name="gpt-4")

    def analyze(self, financial_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze financial data and calculate ratios."""
        try:
            print("PE Agent: Starting financial analysis...")
            
            # Calculate financial ratios
            ratios = self.calculate_financial_ratios(financial_data)
            print("PE Agent: Financial ratios calculated")
            
            # Analyze with GPT-4
            analysis_prompt = f"""
            Analyze the following financial ratios and data for detailed insights:

            Financial Ratios:
            {json.dumps(ratios, indent=2)}

            Raw Financial Data:
            {json.dumps(financial_data, indent=2)}

            Please provide a comprehensive analysis covering:
            1. Profitability Analysis
            2. Liquidity Assessment
            3. Efficiency Metrics
            4. Growth Indicators
            5. Risk Factors
            6. Comparative Industry Analysis
            7. Key Financial Strengths and Weaknesses
            8. Future Financial Outlook

            Focus on both quantitative analysis and qualitative insights.
            """

            analysis = self.llm.invoke(analysis_prompt)
            print("PE Agent: GPT-4 analysis completed")

            return {
                "ratios": ratios,
                "analysis": analysis.content
            }

        except Exception as e:
            print(f"Error in PE analysis: {str(e)}")
            return {}

    def calculate_financial_ratios(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate key financial ratios from the data."""
        try:
            company_data = list(data.values())[0]  # Get the company's financial data
            
            ratios = {
                "profitability_ratios": {
                    "gross_margin": self.calculate_gross_margin(company_data),
                    "operating_margin": self.calculate_operating_margin(company_data),
                    "net_profit_margin": self.calculate_net_margin(company_data),
                    "return_on_equity": self.calculate_roe(company_data),
                    "return_on_assets": self.calculate_roa(company_data)
                },
                "liquidity_ratios": {
                    "current_ratio": self.calculate_current_ratio(company_data),
                    "quick_ratio": self.calculate_quick_ratio(company_data),
                    "cash_ratio": self.calculate_cash_ratio(company_data)
                },
                "efficiency_ratios": {
                    "asset_turnover": self.calculate_asset_turnover(company_data),
                    "inventory_turnover": self.calculate_inventory_turnover(company_data),
                    "receivables_turnover": self.calculate_receivables_turnover(company_data)
                },
                "leverage_ratios": {
                    "debt_to_equity": self.calculate_debt_to_equity(company_data),
                    "debt_to_assets": self.calculate_debt_to_assets(company_data),
                    "interest_coverage": self.calculate_interest_coverage(company_data)
                }
            }
            
            return ratios

        except Exception as e:
            print(f"Error calculating ratios: {str(e)}")
            return {}

    # Individual ratio calculations
    def calculate_gross_margin(self, data: Dict[str, Any]) -> float:
        try:
            revenue = float(data.get("revenue", 0))
            cogs = float(data.get("cost_of_goods_sold", 0))
            return (revenue - cogs) / revenue if revenue else 0
        except:
            return 0

    def calculate_operating_margin(self, data: Dict[str, Any]) -> float:
        try:
            revenue = float(data.get("revenue", 0))
            operating_income = float(data.get("operating_income", 0))
            return operating_income / revenue if revenue else 0
        except:
            return 0

    def calculate_net_margin(self, data: Dict[str, Any]) -> float:
        try:
            revenue = float(data.get("revenue", 0))
            net_income = float(data.get("net_income", 0))
            return net_income / revenue if revenue else 0
        except:
            return 0

    # Add other ratio calculation methods...
    # (Keep all other existing ratio calculation methods)

pe_analysis_agent = PEAnalysisAgent() 