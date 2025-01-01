from pathlib import Path
import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from tools.api import FinancialDataAPI  # Import the API tool

class PEAnalysisAgent:
    """Agent for calculating Private Equity ratios and metrics."""
    
    def __init__(self):
        self.base_dir = Path("document_processing")
        self.setup_logging()
        self.api = FinancialDataAPI()  # Initialize API client

    def setup_logging(self):
        """Configure logging."""
        log_file = self.base_dir / "logs" / "pe_analysis.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def load_financial_data(self, json_path: str) -> Dict[str, Any]:
        """Load financial data from JSON file or fetch from API if needed."""
        try:
            # First try to load from file
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            # Check if we have the required financial data
            if not self._has_required_metrics(data):
                # If not, try to get from API
                company = data.get('company') or data.get('ticker')
                if company:
                    self.logger.info(f"Fetching additional data from API for {company}")
                    api_data = self.api.get_financial_statements(company)
                    
                    # Merge API data with existing data
                    data = self._merge_financial_data(data, api_data)
                else:
                    self.logger.warning("No company identifier found for API lookup")
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error loading financial data: {e}")
            raise

    def _has_required_metrics(self, data: Dict[str, Any]) -> bool:
        """Check if the data has required financial metrics."""
        try:
            financial_data = data.get('financial_data', {})
            
            # Check balance sheet metrics
            balance_sheet = financial_data.get('balance_sheet', {})
            has_balance = all(key in balance_sheet for key in [
                'total_debt', 'total_assets', 'total_equity'
            ])
            
            # Check income statement metrics
            income_stmt = financial_data.get('income_statement', {})
            has_income = all(key in income_stmt for key in [
                'revenue', 'operating_income', 'ebitda', 'net_income'
            ])
            
            # Check cash flow metrics
            cash_flow = financial_data.get('cash_flow', {})
            has_cash_flow = all(key in cash_flow for key in [
                'operating_cash_flow', 'capital_expenditures'
            ])
            
            return has_balance and has_income and has_cash_flow
            
        except Exception:
            return False

    def _merge_financial_data(self, existing_data: Dict[str, Any], api_data: Dict[str, Any]) -> Dict[str, Any]:
        """Merge API data with existing data, preferring existing data when available."""
        try:
            if not existing_data.get('financial_data'):
                existing_data['financial_data'] = {}
            
            for statement_type in ['balance_sheet', 'income_statement', 'cash_flow']:
                if not existing_data['financial_data'].get(statement_type):
                    existing_data['financial_data'][statement_type] = {}
                
                # Get API data for this statement
                api_statement = api_data.get('financial_data', {}).get(statement_type, {})
                
                # Update missing metrics
                for metric, value in api_statement.items():
                    if metric not in existing_data['financial_data'][statement_type]:
                        existing_data['financial_data'][statement_type][metric] = value
            
            return existing_data
            
        except Exception as e:
            self.logger.error(f"Error merging financial data: {e}")
            return existing_data

    def calculate_pe_metrics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate key PE metrics and ratios."""
        try:
            metrics = {
                "leverage_metrics": self._calculate_leverage_metrics(data),
                "operational_metrics": self._calculate_operational_metrics(data),
                "cash_flow_metrics": self._calculate_cash_flow_metrics(data),
                "growth_metrics": self._calculate_growth_metrics(data),
                "valuation_metrics": self._calculate_valuation_metrics(data)
            }
            
            return metrics
        except Exception as e:
            self.logger.error(f"Error calculating PE metrics: {e}")
            raise

    def _calculate_leverage_metrics(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate leverage-related metrics."""
        try:
            financial_data = data.get('financial_data', {})
            balance_sheet = financial_data.get('balance_sheet', {})
            income_stmt = financial_data.get('income_statement', {})
            
            total_debt = float(balance_sheet.get('total_debt', 0) or 0)
            ebitda = float(income_stmt.get('ebitda', 0) or 0)
            total_assets = float(balance_sheet.get('total_assets', 0) or 0)
            equity = float(balance_sheet.get('total_equity', 0) or 0)
            
            return {
                "debt_to_ebitda": total_debt / ebitda if ebitda != 0 else None,
                "debt_to_equity": total_debt / equity if equity != 0 else None,
                "debt_to_assets": total_debt / total_assets if total_assets != 0 else None
            }
        except Exception as e:
            self.logger.error(f"Error in leverage metrics: {e}")
            return {}

    def _calculate_operational_metrics(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate operational metrics."""
        try:
            financial_data = data.get('financial_data', {})
            income_stmt = financial_data.get('income_statement', {})
            balance_sheet = financial_data.get('balance_sheet', {})
            
            revenue = float(income_stmt.get('revenue', 0) or 0)
            ebit = float(income_stmt.get('operating_income', 0) or 0)
            total_assets = float(balance_sheet.get('total_assets', 0) or 0)
            
            return {
                "ebit_margin": (ebit / revenue if revenue != 0 else None),
                "asset_turnover": (revenue / total_assets if total_assets != 0 else None),
                "roa": (ebit / total_assets if total_assets != 0 else None)
            }
        except Exception as e:
            self.logger.error(f"Error in operational metrics: {e}")
            return {}

    def _calculate_cash_flow_metrics(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate cash flow metrics."""
        try:
            financial_data = data.get('financial_data', {})
            cash_flow = financial_data.get('cash_flow', {})
            income_stmt = financial_data.get('income_statement', {})
            
            operating_cash_flow = float(cash_flow.get('operating_cash_flow', 0) or 0)
            capex = float(cash_flow.get('capital_expenditures', 0) or 0)
            ebitda = float(income_stmt.get('ebitda', 0) or 0)
            revenue = float(income_stmt.get('revenue', 0) or 0)
            
            fcf = operating_cash_flow - capex
            
            return {
                "fcf_yield": fcf / ebitda if ebitda != 0 else None,
                "capex_to_revenue": capex / revenue if revenue != 0 else None,
                "ocf_to_ebitda": operating_cash_flow / ebitda if ebitda != 0 else None
            }
        except Exception as e:
            self.logger.error(f"Error in cash flow metrics: {e}")
            return {}

    def _calculate_growth_metrics(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate growth metrics."""
        try:
            financial_data = data.get('financial_data', {})
            income_stmt = financial_data.get('income_statement', {})
            
            return {
                "revenue_growth": float(income_stmt.get('revenue_growth', 0) or 0),
                "ebitda_growth": float(income_stmt.get('ebitda_growth', 0) or 0),
                "net_income_growth": float(income_stmt.get('net_income_growth', 0) or 0)
            }
        except Exception as e:
            self.logger.error(f"Error in growth metrics: {e}")
            return {}

    def _calculate_valuation_metrics(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate valuation metrics."""
        try:
            financial_data = data.get('financial_data', {})
            income_stmt = financial_data.get('income_statement', {})
            balance_sheet = financial_data.get('balance_sheet', {})
            
            ebitda = float(income_stmt.get('ebitda', 0) or 0)
            enterprise_value = float(balance_sheet.get('enterprise_value', 0) or 0)
            net_income = float(income_stmt.get('net_income', 0) or 0)
            total_equity = float(balance_sheet.get('total_equity', 0) or 0)
            
            return {
                "ev_to_ebitda": enterprise_value / ebitda if ebitda != 0 else None,
                "roe": net_income / total_equity if total_equity != 0 else None
            }
        except Exception as e:
            self.logger.error(f"Error in valuation metrics: {e}")
            return {}

    def format_metrics_for_report(self, metrics: Dict[str, Any]) -> str:
        """Format metrics into a readable string for the report."""
        report_sections = []
        
        # Format each metric category
        for category, values in metrics.items():
            section = f"\n{category.replace('_', ' ').upper()}\n" + "-" * 40 + "\n"
            
            for name, value in values.items():
                if value is not None:
                    formatted_name = name.replace('_', ' ').title()
                    if isinstance(value, float):
                        if 'ratio' in name or 'margin' in name:
                            formatted_value = f"{value:.2%}"
                        else:
                            formatted_value = f"{value:,.2f}"
                    else:
                        formatted_value = str(value)
                    section += f"{formatted_name}: {formatted_value}\n"
            
            report_sections.append(section)
        
        return "\n".join(report_sections)

    def get_metrics_summary(self, metrics: Dict[str, Any]) -> str:
        """Generate a summary analysis of the metrics."""
        summary_template = """
FINANCIAL METRICS ANALYSIS
=========================

1. Leverage Analysis
-------------------
{leverage_analysis}

2. Operational Performance
-------------------------
{operational_analysis}

3. Cash Flow Analysis
--------------------
{cash_flow_analysis}

4. Growth & Valuation
--------------------
{growth_analysis}

Key Findings:
{key_findings}
"""
        
        # Analyze each category
        leverage_analysis = self._analyze_leverage(metrics.get('leverage_metrics', {}))
        operational_analysis = self._analyze_operational(metrics.get('operational_metrics', {}))
        cash_flow_analysis = self._analyze_cash_flow(metrics.get('cash_flow_metrics', {}))
        growth_analysis = self._analyze_growth_and_valuation(
            metrics.get('growth_metrics', {}),
            metrics.get('valuation_metrics', {})
        )
        key_findings = self._compile_key_findings(metrics)
        
        return summary_template.format(
            leverage_analysis=leverage_analysis,
            operational_analysis=operational_analysis,
            cash_flow_analysis=cash_flow_analysis,
            growth_analysis=growth_analysis,
            key_findings=key_findings
        )

    def _analyze_leverage(self, metrics: Dict[str, float]) -> str:
        """Analyze leverage metrics."""
        analysis = []
        
        debt_to_ebitda = metrics.get('debt_to_ebitda')
        if debt_to_ebitda is not None:
            try:
                ratio = float(debt_to_ebitda)
                if ratio > 6:
                    analysis.append(f"High leverage with Debt/EBITDA of {ratio:.2f}x")
                elif ratio > 4:
                    analysis.append(f"Moderate leverage with Debt/EBITDA of {ratio:.2f}x")
                else:
                    analysis.append(f"Conservative leverage with Debt/EBITDA of {ratio:.2f}x")
            except (TypeError, ValueError):
                pass
        
        debt_to_equity = metrics.get('debt_to_equity')
        if debt_to_equity is not None:
            try:
                ratio = float(debt_to_equity)
                analysis.append(f"Debt/Equity ratio stands at {ratio:.2f}x")
            except (TypeError, ValueError):
                pass
            
        return "\n".join(analysis) if analysis else "Insufficient leverage data"

    def _analyze_operational(self, metrics: Dict[str, float]) -> str:
        """Analyze operational metrics."""
        analysis = []
        
        ebit_margin = metrics.get('ebit_margin')
        if ebit_margin is not None:
            try:
                margin = float(ebit_margin)
                if margin > 0.20:
                    analysis.append(f"Strong operational efficiency with {margin:.1%} EBIT margin")
                elif margin > 0.10:
                    analysis.append(f"Moderate operational efficiency with {margin:.1%} EBIT margin")
                else:
                    analysis.append(f"Low operational efficiency with {margin:.1%} EBIT margin")
            except (TypeError, ValueError):
                pass
        
        roa = metrics.get('roa')
        if roa is not None:
            try:
                roa_value = float(roa)
                analysis.append(f"Return on Assets: {roa_value:.1%}")
            except (TypeError, ValueError):
                pass
            
        return "\n".join(analysis) if analysis else "Insufficient operational data"

    def _analyze_cash_flow(self, metrics: Dict[str, float]) -> str:
        """Analyze cash flow metrics."""
        analysis = []
        
        fcf_yield = metrics.get('fcf_yield')
        if fcf_yield is not None:
            try:
                yield_value = float(fcf_yield)
                analysis.append(f"Free Cash Flow Yield: {yield_value:.1%}")
            except (TypeError, ValueError):
                pass
        
        ocf_to_ebitda = metrics.get('ocf_to_ebitda')
        if ocf_to_ebitda is not None:
            try:
                ratio = float(ocf_to_ebitda)
                if ratio > 0.8:
                    analysis.append("Strong cash conversion")
                else:
                    analysis.append("Moderate cash conversion")
            except (TypeError, ValueError):
                pass
            
        return "\n".join(analysis) if analysis else "Insufficient cash flow data"

    def _analyze_growth_and_valuation(self, growth: Dict[str, float], valuation: Dict[str, float]) -> str:
        """Analyze growth and valuation metrics."""
        analysis = []
        
        revenue_growth = growth.get('revenue_growth')
        if revenue_growth is not None:
            try:
                growth_rate = float(revenue_growth)
                analysis.append(f"Revenue growth: {growth_rate:.1%}")
            except (TypeError, ValueError):
                pass
        
        ev_to_ebitda = valuation.get('ev_to_ebitda')
        if ev_to_ebitda is not None:
            try:
                multiple = float(ev_to_ebitda)
                analysis.append(f"EV/EBITDA multiple: {multiple:.1f}x")
            except (TypeError, ValueError):
                pass
            
        return "\n".join(analysis) if analysis else "Insufficient growth/valuation data"

    def _compile_key_findings(self, metrics: Dict[str, Any]) -> str:
        """Compile key findings from all metrics."""
        findings = []
        
        leverage = metrics.get('leverage_metrics', {})
        debt_to_ebitda = leverage.get('debt_to_ebitda')
        if debt_to_ebitda is not None:
            try:
                if float(debt_to_ebitda) > 6:
                    findings.append("- High leverage poses significant risk")
            except (TypeError, ValueError):
                pass
        
        operational = metrics.get('operational_metrics', {})
        ebit_margin = operational.get('ebit_margin')
        if ebit_margin is not None:
            try:
                if float(ebit_margin) > 0.20:
                    findings.append("- Strong operational performance")
            except (TypeError, ValueError):
                pass
        
        cash_flow = metrics.get('cash_flow_metrics', {})
        fcf_yield = cash_flow.get('fcf_yield')
        if fcf_yield is not None:
            try:
                if float(fcf_yield) > 0.10:
                    findings.append("- Strong free cash flow generation")
            except (TypeError, ValueError):
                pass
            
        return "\n".join(findings) if findings else "No significant findings"

    def analyze_company(self, json_path: str) -> Dict[str, Any]:
        """Analyze company using PE metrics."""
        try:
            # Load data (now with API fallback)
            data = self.load_financial_data(json_path)
            
            # Calculate metrics
            metrics = self.calculate_pe_metrics(data)
            
            # Format metrics for report
            formatted_metrics = self.format_metrics_for_report(metrics)
            metrics_summary = self.get_metrics_summary(metrics)
            
            # Combine into analysis result
            analysis = {
                "company": data.get("company", "Unknown"),
                "analysis_date": datetime.now().isoformat(),
                "metrics": metrics,
                "formatted_metrics": formatted_metrics,
                "metrics_summary": metrics_summary
            }
            
            # Save analysis results
            output_path = self.base_dir / "analysis_results" / f"pe_analysis_{datetime.now():%Y%m%d_%H%M%S}.json"
            output_path.parent.mkdir(exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(analysis, f, indent=2)
            
            self.logger.info(f"Analysis completed and saved to {output_path}")
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing company: {e}")
            return {"error": str(e)} 