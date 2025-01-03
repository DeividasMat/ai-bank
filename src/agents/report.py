from pathlib import Path
import json
from datetime import datetime
import logging
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from fpdf import FPDF
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from src.agents.search_agent import SearchAgent
from dotenv import load_dotenv
import os
from typing import Dict, Any, List
from agents.market_research import market_research_agent
from agents.pe_analysis import pe_analysis_agent
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
PERPLEXITY_API_KEY = os.getenv('PERPLEXITY_API_KEY')

class PEReport(FPDF):
    def __init__(self):
        super().__init__()
        self.set_margins(25, 25, 25)
        self.set_auto_page_break(auto=True, margin=25)
        
    def sanitize_text(self, text):
        """Clean text of problematic characters"""
        if not text:
            return ""
        # Replace smart quotes and other problematic characters
        replacements = {
            '"': '"',
            '"': '"',
            ''': "'",
            ''': "'",
            '–': '-',
            '—': '-',
            '…': '...',
            '\u2022': '-',  # Replace bullet with dash
            '\u2019': "'",
            '\u2018': "'",
            '\u201C': '"',
            '\u201D': '"',
            '\u2013': '-',
            '\u2014': '-',
            '\u2026': '...'
        }
        result = text
        for old, new in replacements.items():
            result = result.replace(old, new)
        return result
        
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.set_text_color(50, 50, 50)
        self.cell(0, 10, 'Private Equity Investment Analysis', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    def chapter_title(self, title):
        title = self.sanitize_text(title)
        self.set_font('Arial', 'B', 16)
        self.set_text_color(31, 73, 125)
        self.cell(0, 10, title, 0, 1, 'L')
        self.set_draw_color(31, 73, 125)
        self.line(25, self.get_y(), 185, self.get_y())
        self.ln(10)

    def add_section(self, title, content):
        """Enhanced section with better formatting"""
        title = self.sanitize_text(title)
        content = self.sanitize_text(content)
        
        if title:
            self.set_font('Arial', 'B', 14)
            self.set_text_color(31, 73, 125)
            self.cell(0, 10, title, ln=True)
            self.set_draw_color(200, 200, 200)
            self.line(25, self.get_y(), 185, self.get_y())
            self.ln(5)
        
        self.set_font('Arial', '', 11)
        self.set_text_color(50, 50, 50)
        
        paragraphs = content.split('\n')
        for paragraph in paragraphs:
            if not paragraph.strip():
                self.ln(5)
                continue
            
            # Handle bullet points with simple dash
            if paragraph.strip().startswith('•') or paragraph.strip().startswith('-'):
                self.cell(10, 7, '-', 0, 0)
                paragraph = paragraph.strip()[1:].strip()
            
            words = paragraph.split()
            line = ''
            for word in words:
                test_line = line + ' ' + word if line else word
                if self.get_string_width(test_line) < (self.w - 60):
                    line = test_line
                else:
                    self.cell(0, 7, line.strip(), ln=True)
                    line = word
            if line:
                self.cell(0, 7, line.strip(), ln=True)
            self.ln(5)

def clean_json_text(data):
    """Recursively clean text in JSON data"""
    if isinstance(data, dict):
        return {k: clean_json_text(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [clean_json_text(x) for x in data]
    elif isinstance(data, str):
        # Replace problematic characters
        replacements = {
            '"': '"',
            '"': '"',
            ''': "'",
            ''': "'",
            '–': '-',
            '—': '-',
            '…': '...',
        }
        text = data
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text
    return data

def get_company_background(gpt4, company_name):
    """Get comprehensive company background using GPT-4"""
    background_prompt = ChatPromptTemplate.from_template("""
    Provide a comprehensive background analysis for {company} including:
    1. Company Overview
    2. Business Model
    3. Market Position
    4. Competitive Advantages
    5. Management Team Analysis
    6. Industry Analysis
    7. Historical Performance
    
    Format the response in clear sections with professional language suitable for a PE report.
    """)
    
    response = gpt4.invoke(background_prompt.format_messages(company=company_name))
    return response.content

def analyze_financial_data(gpt4, data):
    """Get detailed financial analysis using GPT-4"""
    analysis_prompt = ChatPromptTemplate.from_template("""
    Analyze this financial data and provide a detailed PE investment analysis:
    {data}
    
    Include:
    1. Executive Summary
    2. Key Financial Metrics Analysis
    3. Growth Trends
    4. Profitability Analysis
    5. Balance Sheet Strength
    6. Cash Flow Analysis
    7. Working Capital Management
    8. Investment Considerations
    9. Risk Factors
    10. Valuation Insights
    11. Investment Recommendation
    
    Format as a professional PE report with clear sections and bullet points where appropriate.
    """)
    
    response = gpt4.invoke(analysis_prompt.format_messages(data=json.dumps(data, indent=2)))
    return response.content

def create_visualizations(data, metrics_dir):
    """Create professional visualizations from financial data"""
    plots = {}
    
    try:
        # Set professional style for plots
        plt.style.use('bmh')  # Using a built-in style that works well for financial data
        
        # Configure Seaborn style
        sns.set_theme(style="whitegrid")
        sns.set_palette("deep")
        
        # Time Series Plot
        if isinstance(data, dict):
            for key, values in data.items():
                if isinstance(values, dict) and any(str(k).isdigit() for k in values.keys()):
                    # Create figure
                    fig, ax = plt.subplots(figsize=(12, 6))
                    
                    # Prepare data
                    years = [k for k in values.keys() if str(k).isdigit()]
                    years.sort()  # Sort years
                    numeric_values = [float(values[year]) if isinstance(values[year], (int, float)) else 0 
                                    for year in years]
                    
                    # Create plot
                    ax.plot(years, numeric_values, marker='o', linewidth=2, markersize=8)
                    ax.set_title(f'{key.replace("_", " ").title()} Trend', pad=20, fontsize=14)
                    ax.set_xlabel('Year', fontsize=12)
                    ax.set_ylabel('Value (USD)', fontsize=12)
                    
                    # Add value labels
                    for x, y in zip(years, numeric_values):
                        ax.annotate(f'${y:,.0f}', 
                                  (x, y), 
                                  textcoords="offset points", 
                                  xytext=(0,10), 
                                  ha='center')
                    
                    # Customize grid
                    ax.grid(True, linestyle='--', alpha=0.7)
                    
                    # Rotate x-axis labels
                    plt.xticks(rotation=45)
                    
                    # Adjust layout
                    plt.tight_layout()
                    
                    # Save plot
                    plot_path = metrics_dir / f'{key}_trend.png'
                    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    plots[f'{key}_trend'] = plot_path

                # Create distribution plots for nested dictionaries
                elif isinstance(values, dict) and any(isinstance(v, (int, float)) for v in values.values()):
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    # Prepare data
                    categories = list(values.keys())
                    amounts = [float(v) if isinstance(v, (int, float)) else 0 for v in values.values()]
                    
                    # Create bar plot
                    bars = ax.bar(categories, amounts)
                    ax.set_title(f'{key.replace("_", " ").title()} Distribution', pad=20, fontsize=14)
                    ax.set_xlabel('Category', fontsize=12)
                    ax.set_ylabel('Value (USD)', fontsize=12)
                    
                    # Add value labels
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                               f'${height:,.0f}',
                               ha='center', va='bottom')
                    
                    # Rotate x-axis labels if needed
                    plt.xticks(rotation=45, ha='right')
                    
                    # Adjust layout
                    plt.tight_layout()
                    
                    # Save plot
                    plot_path = metrics_dir / f'{key}_distribution.png'
                    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    plots[f'{key}_dist'] = plot_path

    except Exception as e:
        logger.error(f"Error creating visualizations: {str(e)}")
        raise e
    
    return plots

def find_company_name(gpt4, data):
    """Dedicated function to find company name in JSON data"""
    name_prompt = ChatPromptTemplate.from_template("""
    Find the company name in this JSON data. Return ONLY the name, without any additional text.
    Example response format: "Netflix, Inc."

    JSON data:
    {raw_data}
    """)
    
    try:
        response = gpt4.invoke(name_prompt.format_messages(raw_data=json.dumps(data, indent=2)))
        company_name = response.content.strip().strip('"').strip("'")
        # Remove any extra text that might have been added
        if "the company name" in company_name.lower():
            company_name = company_name.split('"')[1]
        print(f"Found company name: {company_name}")
        return company_name
    except Exception as e:
        print(f"Error finding company name: {str(e)}")
        return "Unknown Company"

def format_text_for_pdf(text, max_chars=75):
    """Format text to prevent PDF rendering issues"""
    if not text:
        return ""
    
    # Remove any problematic characters
    text = text.replace('\t', ' ').replace('\r', '')
    
    # Split into paragraphs
    paragraphs = text.split('\n')
    formatted_paragraphs = []
    
    for paragraph in paragraphs:
        words = paragraph.split()
        lines = []
        current_line = []
        current_length = 0
        
        for word in words:
            word_length = len(word)
            if current_length + word_length + 1 <= max_chars:
                current_line.append(word)
                current_length += word_length + 1
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
                current_length = word_length
        
        if current_line:
            lines.append(' '.join(current_line))
        
        formatted_paragraphs.append('\n'.join(lines))
    
    return '\n\n'.join(formatted_paragraphs)

def analyze_json_structure(gpt4, data):
    """Use GPT-4 to analyze JSON structure and extract key information"""
    # First, get the company name
    company_name = find_company_name(gpt4, data)
    
    structure_prompt = ChatPromptTemplate.from_template("""
    Analyze this financial data for {company_name}:

    {raw_data}
    
    Create a structured analysis with:
    1. All numerical values found (with their context and meaning)
    2. Any time series data (year-over-year changes)
    3. Geographical or segment breakdowns
    4. Calculate these ratios if data available:
       - Growth rates
       - Regional distribution percentages
       - Year-over-year changes
       - Market share metrics
       - Asset utilization
    
    Respond with this exact JSON structure:
    {{
        "company_name": "{company_name}",
        "numerical_data": {{
            "key_metrics": {{
                "metric_name": {{
                    "value": "number",
                    "context": "explanation"
                }}
            }},
            "time_series": {{
                "series_name": {{
                    "years": [],
                    "values": [],
                    "trend": "description"
                }}
            }},
            "geographical_data": {{}}
        }},
        "calculated_ratios": {{
            "ratio_name": {{
                "value": "number",
                "formula": "explanation",
                "interpretation": "meaning"
            }}
        }}
    }}
    """)
    
    try:
        response = gpt4.invoke(structure_prompt.format_messages(
            company_name=company_name,
            raw_data=json.dumps(data, indent=2)
        ))
        
        # Extract JSON from response
        content = response.content
        start = content.find('{')
        end = content.rfind('}') + 1
        if start >= 0 and end > start:
            json_str = content[start:end]
            analyzed_data = json.loads(json_str)
            analyzed_data["company_name"] = company_name  # Ensure we use the found name
            return analyzed_data
        else:
            return {
                "company_name": company_name,
                "numerical_data": {},
                "calculated_ratios": {}
            }
    except Exception as e:
        print(f"Error in analyze_json_structure: {str(e)}")
        return {
            "company_name": company_name,
            "numerical_data": {},
            "calculated_ratios": {}
        }

def generate_financial_insights(gpt4, analyzed_data):
    """Generate comprehensive financial insights using GPT-4"""
    insights_prompt = ChatPromptTemplate.from_template("""
    Analyze this financial data and provide insights:

    {analyzed_data}
    
    Please provide:
    1. Key Performance Indicators analysis
    2. Financial strength assessment
    3. Growth analysis
    4. Risk factors
    5. Industry comparison
    6. Investment considerations
    
    Focus on private equity investment relevance.
    Highlight any patterns or potential issues.
    """)
    
    try:
        response = gpt4.invoke(insights_prompt.format_messages(
            analyzed_data=json.dumps(analyzed_data, indent=2)
        ))
        return response.content
    except Exception as e:
        print(f"Error generating insights: {str(e)}")
        return "Unable to generate financial insights."

def generate_executive_summary(gpt4, data, company_name):
    """Generate an executive summary using GPT-4"""
    summary_prompt = ChatPromptTemplate.from_template("""
    Create a detailed executive summary for {company_name} based on this data:
    {data}
    
    Include:
    1. Company Overview
    2. Market Position
    3. Key Investment Highlights
    4. Major Risk Factors
    5. Growth Opportunities
    6. Competitive Advantages
    
    Format with simple dashes (-) for bullet points, not special characters.
    Be specific, data-driven, and analytical.
    """)
    
    response = gpt4.invoke(summary_prompt.format_messages(
        company_name=company_name,
        data=json.dumps(data, indent=2)
    ))
    return response.content

def generate_market_analysis(gpt4, data):
    """Generate comprehensive market analysis"""
    market_prompt = ChatPromptTemplate.from_template("""
    Provide a detailed market analysis based on this data:
    {data}
    
    Include:
    1. Industry Overview & Size
    2. Market Growth Trends
    3. Competitive Landscape
    4. Market Share Analysis
    5. Industry Challenges & Opportunities
    6. Regulatory Environment
    7. Future Market Projections
    
    Use simple dashes (-) for bullet points, not special characters.
    Focus on quantitative data and specific insights.
    """)
    
    response = gpt4.invoke(market_prompt.format_messages(data=json.dumps(data, indent=2)))
    return response.content

def generate_financial_deep_dive(gpt4, data):
    """Generate detailed financial analysis"""
    financial_prompt = ChatPromptTemplate.from_template("""
    Perform a comprehensive financial analysis:
    {data}
    
    Include:
    1. Revenue Analysis
       - Growth trends
       - Revenue streams
       - Geographic breakdown
    
    2. Profitability Analysis
       - Margin trends
       - Cost structure
       - Operational efficiency
    
    3. Balance Sheet Strength
       - Asset composition
       - Debt levels
       - Working capital
    
    4. Cash Flow Analysis
       - Operating cash flow
       - Capital expenditure
       - Free cash flow
    
    5. Key Financial Metrics
       - ROE, ROA, ROIC
       - Debt/EBITDA
       - Interest coverage
    
    6. Peer Comparison
       - Relative valuation
       - Performance benchmarking
    
    Be specific with numbers and provide interpretation for each metric.
    """)
    
    response = gpt4.invoke(financial_prompt.format_messages(data=json.dumps(data, indent=2)))
    return response.content

def report_agent(state):
    """Generate enhanced PE investment report"""
    print("Starting enhanced PE report generation...")
    
    try:
        # Initialize GPT-4
        gpt4 = ChatOpenAI(model="gpt-4", temperature=0)
        
        # Setup directories
        reports_dir = Path("reports")
        metrics_dir = reports_dir / "metrics"
        reports_dir.mkdir(exist_ok=True)
        metrics_dir.mkdir(exist_ok=True)
        
        # Get data
        json_files = list(Path("document_processing/extracted_data").glob("*.json"))
        if not json_files:
            print("No JSON files found!")
            return state
            
        latest_file = max(json_files, key=lambda x: x.stat().st_mtime)
        print(f"\nAnalyzing file: {latest_file}")
        
        with open(latest_file, 'r') as f:
            data = json.load(f)
        
        # Clean data before processing
        data = clean_json_text(data)
        
        # Get company name
        company_name = find_company_name(gpt4, data)
        print(f"Identified company: {company_name}")
        
        # Analyze data structure
        analyzed_data = analyze_json_structure(gpt4, data)
        
        # Generate analyses
        print("Generating executive summary...")
        executive_summary = generate_executive_summary(gpt4, analyzed_data, company_name)
        
        print("Generating market analysis...")
        market_analysis = generate_market_analysis(gpt4, analyzed_data)
        
        print("Generating financial deep dive...")
        financial_deep_dive = generate_financial_deep_dive(gpt4, analyzed_data)
        
        # Create PDF Report
        pdf = PEReport()
        
        # Cover Page
        pdf.add_page()
        pdf.set_fill_color(31, 73, 125)  # Dark blue
        pdf.rect(0, 0, 210, 30, 'F')  # Top banner
        pdf.rect(0, 277, 210, 20, 'F')  # Bottom banner
        
        pdf.ln(60)
        pdf.set_font('Arial', 'B', 24)
        pdf.set_text_color(31, 73, 125)
        pdf.cell(0, 15, company_name, ln=True, align='C')
        pdf.ln(10)
        pdf.set_font('Arial', 'B', 18)
        pdf.cell(0, 10, 'Investment Analysis Report', ln=True, align='C')
        pdf.ln(10)
        pdf.set_font('Arial', '', 12)
        pdf.cell(0, 10, f'Generated: {datetime.now().strftime("%B %d, %Y")}', ln=True, align='C')
        
        # Table of Contents
        pdf.add_page()
        pdf.chapter_title("Table of Contents")
        sections = [
            "1. Executive Summary",
            "2. Market Analysis",
            "3. Financial Analysis",
            "4. Key Metrics & Ratios",
            "5. Investment Considerations",
            "6. Risk Factors"
        ]
        pdf.ln(10)
        for section in sections:
            pdf.set_font('Arial', '', 12)
            pdf.cell(0, 10, section, ln=True)
            pdf.set_draw_color(200, 200, 200)
            pdf.line(pdf.get_x() + 25, pdf.get_y() - 2, 185, pdf.get_y() - 2)
            pdf.ln(5)
        
        # Executive Summary
        pdf.add_page()
        pdf.chapter_title("1. Executive Summary")
        pdf.add_section("", executive_summary)
        
        # Market Analysis
        pdf.add_page()
        pdf.chapter_title("2. Market Analysis")
        pdf.add_section("", market_analysis)
        
        # Financial Analysis
        pdf.add_page()
        pdf.chapter_title("3. Financial Analysis")
        pdf.add_section("", financial_deep_dive)
        
        # Key Metrics & Ratios
        pdf.add_page()
        pdf.chapter_title("4. Key Metrics & Ratios")
        if analyzed_data.get("numerical_data", {}).get("key_metrics"):
            for metric_name, metric_data in analyzed_data["numerical_data"]["key_metrics"].items():
                metric_text = f"{metric_name}: {metric_data.get('value', 'N/A')}"
                if metric_data.get('context'):
                    metric_text += f"\n{metric_data['context']}"
                pdf.add_section("", metric_text)
        
        # Investment Considerations
        pdf.add_page()
        pdf.chapter_title("5. Investment Considerations")
        if analyzed_data.get("calculated_ratios"):
            for ratio_name, ratio_data in analyzed_data["calculated_ratios"].items():
                ratio_text = f"{ratio_name}: {ratio_data.get('value', 'N/A')}"
                if ratio_data.get('formula'):
                    ratio_text += f"\nFormula: {ratio_data['formula']}"
                if ratio_data.get('interpretation'):
                    ratio_text += f"\nInterpretation: {ratio_data['interpretation']}"
                pdf.add_section("", ratio_text)
        
        # Risk Factors
        pdf.add_page()
        pdf.chapter_title("6. Risk Factors")
        pdf.add_section("", "Key risks and mitigation strategies identified from the analysis:")
        
        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_company_name = company_name.replace(', Inc.', '').replace(' ', '_')
        report_path = reports_dir / f"PE_Analysis_{safe_company_name}_{timestamp}.pdf"
        pdf.output(str(report_path))
        
        print(f"\nEnhanced PE Analysis Report generated at: {report_path}")
        
    except Exception as e:
        print(f"Error generating report: {str(e)}")
        raise e
        
    return state 

class Report:
    def __init__(self):
        self.llm = ChatOpenAI(model_name="gpt-4")
        self.report_dir = Path("document_processing/reports")
        self.report_dir.mkdir(parents=True, exist_ok=True)

    def __call__(self, financial_data: Dict[str, Any]) -> Dict[str, Any]:
        return self.generate(financial_data)

    def find_company_name_with_gpt4(self, financial_data: Dict[str, Any]) -> str:
        """Extract company name from financial data using GPT-4."""
        try:
            print("\nExtracting company name using GPT-4...")
            
            messages = [
                {
                    "role": "system",
                    "content": """You are a financial data analyzer. Find the actual company name in this financial data.
                              Return ONLY the company name, no additional text or explanation."""
                },
                {
                    "role": "user",
                    "content": f"""Find and return ONLY the company name from this financial data:
                            {json.dumps(financial_data, indent=2)}"""
                }
            ]
            
            # Format messages for LangChain
            from langchain_core.messages import SystemMessage, HumanMessage
            formatted_messages = [
                SystemMessage(content=messages[0]["content"]),
                HumanMessage(content=messages[1]["content"])
            ]
            
            response = self.llm.invoke(formatted_messages)
            company_name = response.content.strip()
            
            if not company_name or company_name.lower() in ["company name", "unknown", "none", "company"]:
                raise ValueError(f"Invalid company name extracted: {company_name}")
            
            print(f"Found company name: {company_name}")
            return company_name
            
        except Exception as e:
            print(f"Error extracting company name: {e}")
            raise

    def generate(self, financial_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate complete report and save as PDF."""
        try:
            print("\nStarting data collection and analysis phase...")
            
            # Phase 1: Load Financial Data
            print("\nPhase 1: Loading financial data...")
            extracted_data_dir = Path("document_processing/extracted_data")
            json_files = list(extracted_data_dir.glob("*.json"))
            if not json_files:
                raise FileNotFoundError("No financial data files found in document_processing/extracted_data/")
            
            latest_file = max(json_files, key=lambda x: x.stat().st_mtime)
            print(f"Found latest financial data: {latest_file}")
            
            with open(latest_file, 'r') as f:
                financial_data = json.load(f)
            
            # Extract company name using GPT-4
            company_name = self.find_company_name_with_gpt4(financial_data)
            print(f"\nAnalyzing data for: {company_name}")
            
            # Phase 2: Market Research with Perplexity
            print(f"\nPhase 2: Starting market research for {company_name}...")
            print("Initiating Perplexity article search...")
            market_research_agent.search_articles(company_name)
            print("✓ Perplexity article search complete")
            
            # Phase 3: Analyze Latest Raw Response
            print("\nPhase 3: Analyzing latest Perplexity response...")
            raw_responses_dir = Path("document_processing/articles/raw_responses")
            raw_response_files = list(raw_responses_dir.glob("raw_response_*.txt"))
            
            if not raw_response_files:
                raise FileNotFoundError("No Perplexity responses found")
            
            latest_response = max(raw_response_files, key=lambda x: x.stat().st_mtime)
            print(f"Found latest response file: {latest_response}")
            
            with open(latest_response, 'r') as f:
                raw_response = f.read()
            
            print("\nAnalyzing market research with GPT-4...")
            market_analysis = self.analyze_perplexity_response(raw_response, company_name)
            print("✓ Market research analysis complete")
            
            # Phase 4: Financial Analysis
            print("\nPhase 4: Performing financial analysis...")
            pe_data = pe_analysis_agent.analyze(financial_data)
            print("✓ Financial analysis complete")
            
            # Phase 5: Generate Final Report
            print("\nPhase 5: Generating comprehensive report...")
            report_data = {
                "company_name": company_name,
                "financial_data": financial_data,
                "pe_analysis": pe_data,
                "market_analysis": {
                    "analysis": market_analysis.content,
                    "source": latest_response.name,
                    "analyzed_date": datetime.now().isoformat()
                },
                "report_date": datetime.now().isoformat()
            }
            
            final_report = self.generate_report_content(report_data)
            
            # Phase 6: Save Report
            print("\nPhase 6: Saving final report...")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_dir = Path("reports")
            report_path = report_dir / f"{company_name}_report_{timestamp}.pdf"
            self.save_as_pdf(final_report, report_path)
            
            print(f"\n✓ Report generation complete")
            print(f"✓ Report saved to: {report_path}")
            return final_report

        except Exception as e:
            print(f"Error generating report: {e}")
            raise

    def generate_with_market_research(self, financial_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate report after market research is complete."""
        try:
            company_name = list(financial_data.keys())[0].replace('_', ' ')
            
            # Phase 1: Load Financial Analysis
            print("\nPhase 1: Loading financial analysis...")
            pe_data = pe_analysis_agent.analyze(financial_data)
            
            # Phase 2: Load and Analyze Perplexity Data
            print("\nPhase 2: Loading Perplexity research data...")
            raw_responses_dir = Path("document_processing/articles/raw_responses")
            raw_response_files = list(raw_responses_dir.glob("raw_response_*.txt"))
            
            if not raw_response_files:
                raise FileNotFoundError("No Perplexity responses found. Please run article search first.")
            
            latest_response = max(raw_response_files, key=lambda x: x.stat().st_mtime)
            print(f"Found latest Perplexity response: {latest_response}")
            
            with open(latest_response, 'r') as f:
                raw_response = f.read()
            
            # Analyze Perplexity data
            print("Analyzing market research data with citations...")
            market_analysis = self.analyze_perplexity_response(raw_response, company_name)
            print("✓ Market research analysis complete")
            
            # Phase 3: Generate Final Report
            print("\nPhase 3: Generating comprehensive report...")
            report_data = {
                "company_name": company_name,
                "financial_data": financial_data,
                "pe_analysis": pe_data,
                "market_analysis": {
                    "analysis": market_analysis.content,
                    "source": latest_response.name,
                    "analyzed_date": datetime.now().isoformat()
                },
                "report_date": datetime.now().isoformat()
            }
            
            final_report = self.generate_report_content(report_data)
            
            # Phase 4: Save Report
            print("\nPhase 4: Saving final report...")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_dir = Path("reports")
            report_path = report_dir / f"{company_name}_report_{timestamp}.pdf"
            self.save_as_pdf(final_report, report_path)
            
            print(f"\n✓ Report generation complete")
            print(f"✓ Report saved to: {report_path}")
            return final_report

        except Exception as e:
            print(f"Error generating report: {e}")
            raise

    def generate_report_content(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate report content using LangChain."""
        try:
            # Create prompt template with emphasis on market research and citations
            template = """
            Generate a comprehensive financial report for {company_name}.

            Financial Analysis:
            {pe_analysis}

            Market Research and News Analysis:
            {market_analysis}

            Please structure the report with the following sections:
            1. Executive Summary
            2. Financial Analysis
               - Key Financial Metrics
               - Ratio Analysis
               - Performance Trends
            3. Market Analysis & News
               - Key Market Developments (with citations)
               - Competitive Landscape
               - Industry Trends
               - News Analysis with Sources
            4. Risk Assessment
               - Financial Risks
               - Market Risks
               - Regulatory Risks (cite relevant news)
            5. Future Outlook
            6. References & Citations
               - List all news sources used

            Important:
            - Include specific citations for market research claims
            - Reference specific articles when discussing market developments
            - Use financial data to support analysis
            - Maintain professional tone and format
            """

            prompt = ChatPromptTemplate.from_template(template)
            chain = prompt | self.llm

            # Generate report with citations
            report_content = chain.invoke({
                "company_name": data["company_name"],
                "pe_analysis": json.dumps(data["pe_analysis"], indent=2),
                "market_analysis": json.dumps(data["market_analysis"], indent=2)  # Use market_analysis directly
            })

            return {
                "content": report_content.content,
                "metadata": {
                    "company": data["company_name"],
                    "date": data["report_date"],
                    "sources": data["market_analysis"].get("source", "")  # Get source from market_analysis
                }
            }

        except Exception as e:
            print(f"Error generating report content: {e}")
            raise

    def save_as_pdf(self, report_content: Dict[str, Any], output_path: Path) -> None:
        """Save report as PDF."""
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            
            # Create PDF document
            doc = SimpleDocTemplate(str(output_path), pagesize=letter)
            styles = getSampleStyleSheet()
            story = []

            # Add title
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                spaceAfter=30
            )
            title = Paragraph(f"Financial Report: {report_content['metadata']['company']}", title_style)
            story.append(title)
            story.append(Spacer(1, 12))

            # Add date
            date_style = ParagraphStyle(
                'Date',
                parent=styles['Normal'],
                fontSize=12,
                spaceAfter=20
            )
            date = Paragraph(f"Generated on: {report_content['metadata']['date']}", date_style)
            story.append(date)
            story.append(Spacer(1, 20))

            # Add content
            content_style = ParagraphStyle(
                'Content',
                parent=styles['Normal'],
                fontSize=12,
                leading=14,
                spaceAfter=12
            )

            # Split content into paragraphs and add to story
            paragraphs = report_content['content'].split('\n')
            for para in paragraphs:
                if para.strip():
                    p = Paragraph(para, content_style)
                    story.append(p)
                    story.append(Spacer(1, 12))

            # Build PDF
            doc.build(story)
            print(f"PDF report saved successfully to: {output_path}")

        except Exception as e:
            print(f"Error saving PDF: {e}")
            raise

    def analyze_perplexity_response(self, raw_response: str, company_name: str) -> Dict[str, Any]:
        """Analyze Perplexity raw response with GPT-4 and extract citations."""
        analysis_prompt = f"""
        Analyze this Perplexity API response about {company_name} and provide a detailed analysis with citations.
        
        Raw Response:
        {raw_response}
        
        Please provide:
        1. Key Findings:
           - Major developments
           - Market trends
           - Company updates
           - Industry insights
        
        2. Detailed Analysis:
           - Market position
           - Competitive landscape
           - Growth opportunities
           - Risk factors
        
        3. Citations:
           For each key point, include:
           - Source name
           - Date (if available)
           - Brief quote or summary
           - URL (if provided)
        
        Format citations as: [Source Name, Date] Quote/Summary
        
        Ensure all significant claims are supported by specific citations from the response.
        """
        
        return self.llm.invoke(analysis_prompt)

# Keep existing report agent instance
report_agent = Report() 