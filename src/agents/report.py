from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from langchain_core.messages import HumanMessage
from agents.state import AgentState
import json
import ast

def report_agent(state: AgentState):
    """Generates a PDF report from the analysis and metrics."""
    
    # Extract messages from other agents
    technical_message = next(msg for msg in state["messages"] if msg.name == "technical_analyst_agent")
    fundamentals_message = next(msg for msg in state["messages"] if msg.name == "fundamentals_agent")
    sentiment_message = next(msg for msg in state["messages"] if msg.name == "sentiment_agent")
    risk_message = next(msg for msg in state["messages"] if msg.name == "risk_management_agent")
    portfolio_message = next(msg for msg in state["messages"] if msg.name == "portfolio_management")

    # Parse messages safely
    try:
        technical_data = json.loads(technical_message.content)
    except json.JSONDecodeError:
        technical_data = ast.literal_eval(technical_message.content)
        
    try:
        fundamental_data = json.loads(fundamentals_message.content)
    except json.JSONDecodeError:
        fundamental_data = ast.literal_eval(fundamentals_message.content)
        
    try:
        sentiment_data = json.loads(sentiment_message.content)
    except json.JSONDecodeError:
        sentiment_data = ast.literal_eval(sentiment_message.content)
        
    try:
        risk_data = json.loads(risk_message.content)
    except json.JSONDecodeError:
        risk_data = ast.literal_eval(risk_message.content)

    # Generate PDF report
    filename = generate_pdf_report(
        state["data"]["ticker"],
        {
            "technical": technical_data,
            "fundamental": fundamental_data,
            "sentiment": sentiment_data,
            "risk": risk_data,
            "portfolio": portfolio_message.content
        },
        state["data"]
    )

    message = HumanMessage(
        content=json.dumps({"report_file": filename}),
        name="report_agent"
    )

    return {"messages": state["messages"] + [message]}

def generate_pdf_report(ticker: str, analysis: dict, data: dict) -> str:
    """Creates a PDF report with all analysis and metrics."""
    
    filename = f"reports/{ticker}_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    doc = SimpleDocTemplate(filename, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30
    )
    story.append(Paragraph(f"Investment Analysis Report - {ticker}", title_style))
    story.append(Paragraph(f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles["Normal"]))
    story.append(Spacer(1, 20))

    # Technical Analysis Section
    story.append(Paragraph("Technical Analysis", styles["Heading2"]))
    tech_analysis = analysis["technical"]
    story.append(Paragraph(f"Signal: {tech_analysis['signal']}", styles["Normal"]))
    story.append(Paragraph(f"Confidence: {tech_analysis['confidence']}", styles["Normal"]))
    
    if 'metrics' in tech_analysis:
        # Create table for technical metrics
        tech_data = [[k, str(v)] for k, v in tech_analysis["metrics"].items()]
        tech_table = Table(
            [["Metric", "Value"]] + tech_data,
            colWidths=[200, 300]
        )
        tech_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(tech_table)
    story.append(Spacer(1, 20))

    # Fundamental Analysis Section
    story.append(Paragraph("Fundamental Analysis", styles["Heading2"]))
    fund_analysis = analysis["fundamental"]
    story.append(Paragraph(f"Signal: {fund_analysis['signal']}", styles["Normal"]))
    story.append(Paragraph(f"Confidence: {fund_analysis['confidence']}", styles["Normal"]))
    
    if 'reasoning' in fund_analysis:
        story.append(Paragraph("Analysis Details:", styles["Heading3"]))
        for category, details in fund_analysis['reasoning'].items():
            story.append(Paragraph(f"{category}:", styles["Heading4"]))
            if isinstance(details, dict):
                for key, value in details.items():
                    story.append(Paragraph(f"{key}: {value}", styles["Normal"]))
            else:
                story.append(Paragraph(str(details), styles["Normal"]))
    story.append(Spacer(1, 20))

    # Sentiment Analysis Section
    story.append(Paragraph("Sentiment Analysis", styles["Heading2"]))
    sentiment_analysis = analysis["sentiment"]
    story.append(Paragraph(f"Signal: {sentiment_analysis['signal']}", styles["Normal"]))
    story.append(Paragraph(f"Confidence: {sentiment_analysis['confidence']}", styles["Normal"]))
    if 'reasoning' in sentiment_analysis:
        story.append(Paragraph(f"Reasoning: {sentiment_analysis['reasoning']}", styles["Normal"]))
    story.append(Spacer(1, 20))
    
    # Risk Analysis Section
    story.append(Paragraph("Risk Analysis", styles["Heading2"]))
    risk_analysis = analysis["risk"]
    story.append(Paragraph(f"Risk Score: {risk_analysis['risk_score']}/10", styles["Normal"]))
    story.append(Paragraph(f"Trading Action: {risk_analysis['trading_action']}", styles["Normal"]))
    story.append(Paragraph(f"Max Position Size: ${risk_analysis['max_position_size']:,.2f}", styles["Normal"]))
    
    if 'risk_metrics' in risk_analysis:
        story.append(Paragraph("Risk Metrics:", styles["Heading3"]))
        risk_data = [[k.replace('_', ' ').title(), f"{float(v):.2%}" if isinstance(v, (float, int)) else str(v)] 
                    for k, v in risk_analysis['risk_metrics'].items()]
        risk_table = Table(
            [["Metric", "Value"]] + risk_data,
            colWidths=[200, 300]
        )
        risk_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(risk_table)
    story.append(Spacer(1, 20))
    
    # Portfolio Management Decision
    story.append(Paragraph("Portfolio Management Decision", styles["Heading2"]))
    try:
        portfolio_decision = json.loads(analysis["portfolio"])
        story.append(Paragraph(f"Action: {portfolio_decision['action']}", styles["Normal"]))
        story.append(Paragraph(f"Quantity: {portfolio_decision['quantity']}", styles["Normal"]))
        if 'reasoning' in portfolio_decision:
            story.append(Paragraph(f"Reasoning: {portfolio_decision['reasoning']}", styles["Normal"]))
    except:
        story.append(Paragraph(analysis["portfolio"], styles["Normal"]))

    # Build and save the PDF
    doc.build(story)
    return filename 