import os
from typing import Dict, Any, List
import pandas as pd
import requests
import logging
from pathlib import Path

class FinancialDataAPI:
    """API client for fetching financial data."""
    
    def __init__(self):
        self.base_url = "https://financialmodelingprep.com/api/v3"
        self.api_key = os.getenv("FMP_API_KEY")
        self.setup_logging()

    def setup_logging(self):
        """Configure logging."""
        log_dir = Path("document_processing/logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / "api.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def get_financial_statements(self, ticker: str) -> Dict[str, Any]:
        """Fetch financial statements for a company."""
        try:
            financial_data = {
                "financial_data": {
                    "balance_sheet": self._get_balance_sheet(ticker),
                    "income_statement": self._get_income_statement(ticker),
                    "cash_flow": self._get_cash_flow(ticker)
                }
            }
            return financial_data
        except Exception as e:
            self.logger.error(f"Error fetching financial data for {ticker}: {e}")
            return {}

    def _get_balance_sheet(self, ticker: str) -> Dict[str, Any]:
        """Fetch balance sheet data."""
        try:
            url = f"{self.base_url}/balance-sheet-statement/{ticker}?apikey={self.api_key}&limit=1"
            response = requests.get(url)
            data = response.json()
            
            if data and isinstance(data, list):
                latest = data[0]
                return {
                    "total_debt": latest.get("totalDebt", 0),
                    "total_assets": latest.get("totalAssets", 0),
                    "total_equity": latest.get("totalStockholdersEquity", 0),
                    "enterprise_value": latest.get("enterpriseValue", 0)
                }
            return {}
        except Exception as e:
            self.logger.error(f"Error fetching balance sheet for {ticker}: {e}")
            return {}

    def _get_income_statement(self, ticker: str) -> Dict[str, Any]:
        """Fetch income statement data."""
        try:
            url = f"{self.base_url}/income-statement/{ticker}?apikey={self.api_key}&limit=2"
            response = requests.get(url)
            data = response.json()
            
            if data and isinstance(data, list) and len(data) >= 2:
                latest = data[0]
                previous = data[1]
                
                revenue = latest.get("revenue", 0)
                prev_revenue = previous.get("revenue", 0)
                ebitda = latest.get("ebitda", 0)
                prev_ebitda = previous.get("ebitda", 0)
                net_income = latest.get("netIncome", 0)
                prev_net_income = previous.get("netIncome", 0)
                
                return {
                    "revenue": revenue,
                    "operating_income": latest.get("operatingIncome", 0),
                    "ebitda": ebitda,
                    "net_income": net_income,
                    "revenue_growth": (revenue - prev_revenue) / prev_revenue if prev_revenue else 0,
                    "ebitda_growth": (ebitda - prev_ebitda) / prev_ebitda if prev_ebitda else 0,
                    "net_income_growth": (net_income - prev_net_income) / prev_net_income if prev_net_income else 0
                }
            return {}
        except Exception as e:
            self.logger.error(f"Error fetching income statement for {ticker}: {e}")
            return {}

    def _get_cash_flow(self, ticker: str) -> Dict[str, Any]:
        """Fetch cash flow data."""
        try:
            url = f"{self.base_url}/cash-flow-statement/{ticker}?apikey={self.api_key}&limit=1"
            response = requests.get(url)
            data = response.json()
            
            if data and isinstance(data, list):
                latest = data[0]
                return {
                    "operating_cash_flow": latest.get("operatingCashFlow", 0),
                    "capital_expenditures": abs(latest.get("capitalExpenditure", 0))
                }
            return {}
        except Exception as e:
            self.logger.error(f"Error fetching cash flow for {ticker}: {e}")
            return {}

def get_financial_metrics(
    ticker: str,
    report_period: str,
    period: str = 'ttm',
    limit: int = 1
) -> List[Dict[str, Any]]:
    """Fetch financial metrics from the API."""
    headers = {"X-API-KEY": os.environ.get("FINANCIAL_DATASETS_API_KEY")}
    url = (
        f"https://api.financialdatasets.ai/financial-metrics/"
        f"?ticker={ticker}"
        f"&report_period_lte={report_period}"
        f"&limit={limit}"
        f"&period={period}"
    )
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise Exception(
            f"Error fetching data: {response.status_code} - {response.text}"
        )
    data = response.json()
    financial_metrics = data.get("financial_metrics")
    if not financial_metrics:
        raise ValueError("No financial metrics returned")
    return financial_metrics

def search_line_items(
    ticker: str,
    line_items: List[str],
    period: str = 'ttm',
    limit: int = 1
) -> List[Dict[str, Any]]:
    """Fetch cash flow statements from the API."""
    headers = {"X-API-KEY": os.environ.get("FINANCIAL_DATASETS_API_KEY")}
    url = "https://api.financialdatasets.ai/financials/search/line-items"

    body = {
        "tickers": [ticker],
        "line_items": line_items,
        "period": period,
        "limit": limit
    }
    response = requests.post(url, headers=headers, json=body)
    if response.status_code != 200:
        raise Exception(
            f"Error fetching data: {response.status_code} - {response.text}"
        )
    data = response.json()
    search_results = data.get("search_results")
    if not search_results:
        raise ValueError("No search results returned")
    return search_results

def get_insider_trades(
    ticker: str,
    end_date: str,
    limit: int = 5,
) -> List[Dict[str, Any]]:
    """
    Fetch insider trades for a given ticker and date range.
    """
    headers = {"X-API-KEY": os.environ.get("FINANCIAL_DATASETS_API_KEY")}
    url = (
        f"https://api.financialdatasets.ai/insider-trades/"
        f"?ticker={ticker}"
        f"&filing_date_lte={end_date}"
        f"&limit={limit}"
    )
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise Exception(
            f"Error fetching data: {response.status_code} - {response.text}"
        )
    data = response.json()
    insider_trades = data.get("insider_trades")
    if not insider_trades:
        raise ValueError("No insider trades returned")
    return insider_trades

def get_market_cap(
    ticker: str,
) -> List[Dict[str, Any]]:
    """Fetch market cap from the API."""
    headers = {"X-API-KEY": os.environ.get("FINANCIAL_DATASETS_API_KEY")}
    url = (
        f'https://api.financialdatasets.ai/company/facts'
        f'?ticker={ticker}'
    )

    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise Exception(
            f"Error fetching data: {response.status_code} - {response.text}"
        )
    data = response.json()
    company_facts = data.get('company_facts')
    if not company_facts:
        raise ValueError("No company facts returned")
    return company_facts.get('market_cap')

def get_prices(
    ticker: str,
    start_date: str,
    end_date: str
) -> List[Dict[str, Any]]:
    """Fetch price data from the API."""
    headers = {"X-API-KEY": os.environ.get("FINANCIAL_DATASETS_API_KEY")}
    url = (
        f"https://api.financialdatasets.ai/prices/"
        f"?ticker={ticker}"
        f"&interval=day"
        f"&interval_multiplier=1"
        f"&start_date={start_date}"
        f"&end_date={end_date}"
    )
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise Exception(
            f"Error fetching data: {response.status_code} - {response.text}"
        )
    data = response.json()
    prices = data.get("prices")
    if not prices:
        raise ValueError("No price data returned")
    return prices

def prices_to_df(prices: List[Dict[str, Any]]) -> pd.DataFrame:
    """Convert prices to a DataFrame."""
    df = pd.DataFrame(prices)
    df["Date"] = pd.to_datetime(df["time"])
    df.set_index("Date", inplace=True)
    numeric_cols = ["open", "close", "high", "low", "volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.sort_index(inplace=True)
    return df

# Update the get_price_data function to use the new functions
def get_price_data(
    ticker: str,
    start_date: str,
    end_date: str
) -> pd.DataFrame:
    prices = get_prices(ticker, start_date, end_date)
    return prices_to_df(prices)
