from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
from main import run_hedge_fund
from fastapi.responses import FileResponse
import json
import os
from tools.setup import ensure_report_directory

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Add your Vercel URL here when deployed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TickerRequest(BaseModel):
    ticker: str

@app.post("/api/analyze")
async def analyze_stock(request: TickerRequest):
    ensure_report_directory()
    try:
        # Default portfolio
        portfolio = {
            "cash": 100000.0,
            "stock": 0
        }
        
        # Get current date
        end_date = datetime.now().strftime('%Y-%m-%d')
        
        # Calculate start date (3 months ago)
        start_date = (datetime.now().replace(month=datetime.now().month - 3 if datetime.now().month > 3 
                     else datetime.now().month + 9, 
                     year=datetime.now().year if datetime.now().month > 3 
                     else datetime.now().year - 1)).strftime('%Y-%m-%d')

        result = run_hedge_fund(
            ticker=request.ticker,
            start_date=start_date,
            end_date=end_date,
            portfolio=portfolio,
            show_reasoning=True
        )
        
        # Get the report filename from the report agent's message
        report_message = next(msg for msg in result["messages"] if msg.name == "report_agent")
        report_file = json.loads(report_message.content)["report_file"]
        
        return {
            "status": "success",
            "data": result,
            "report_url": f"/reports/{os.path.basename(report_file)}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/reports/{filename}")
async def get_report(filename: str):
    report_path = f"reports/{filename}"
    if os.path.exists(report_path):
        return FileResponse(report_path, media_type="application/pdf")
    raise HTTPException(status_code=404, detail="Report not found") 