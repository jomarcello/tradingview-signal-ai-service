import os
import logging
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_KEY"))

SYSTEM_PROMPT = """You are a trading signal formatter. Format the provided trading signal details into a clear, easy-to-read message.

Your message should follow this format:

 *New Trading Signal* 

*Instrument:* [instrument]
*Action:* [BUY/SELL] / 

*Entry Price:* [price]
*Stop Loss:* [price] 
*Take Profit:* [price] 

*Timeframe:* [timeframe]
*Strategy:* [strategy]

-------------------

*Risk Management:*
• Position size: 1-2% max
• Use proper stop loss
• Follow your trading plan

-------------------

 *SigmaPips AI Verdict:*
[2-3 lines explaining why this trade setup looks promising, focusing on technical aspects and risk/reward ratio]

Remember:
- Keep it concise and professional
"""

ANALYSIS_PROMPT = """You are a trading signal analyzer. Analyze the provided trading signal and provide a brief but insightful verdict.

Focus on:
1. Technical aspects of the setup
2. Risk/reward ratio
3. Key levels and potential market behavior

Your verdict should be 2-3 lines long and professional.
"""

class SignalRequest(BaseModel):
    instrument: str
    direction: str
    entry_price: str
    stop_loss: str
    take_profit: str
    timeframe: Optional[str] = None
    strategy: Optional[str] = None
    timestamp: Optional[str] = None

@app.post("/analyze-signal")
async def analyze_signal(request: SignalRequest):
    """Analyze a trading signal using AI"""
    try:
        # Calculate risk/reward ratio
        entry = float(request.entry_price)
        sl = float(request.stop_loss)
        tp = float(request.take_profit)
        
        risk = abs(entry - sl)
        reward = abs(tp - entry)
        rr_ratio = round(reward / risk, 2) if risk > 0 else 0
        
        # Create analysis prompt
        prompt = f"""Signal Details:
- Instrument: {request.instrument}
- Direction: {request.direction.upper()}
- Entry: {request.entry_price}
- Stop Loss: {request.stop_loss}
- Take Profit: {request.take_profit}
- Risk/Reward Ratio: {rr_ratio}
- Timeframe: {request.timeframe or 'Not specified'}
- Strategy: {request.strategy or 'Not specified'}

Please analyze this trade setup and provide your verdict."""

        # Get AI analysis
        response = client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[
                {"role": "system", "content": ANALYSIS_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=150
        )
        
        verdict = response.choices[0].message.content.strip()
        
        return {
            "status": "success",
            "verdict": verdict,
            "risk_reward_ratio": rr_ratio
        }
        
    except Exception as e:
        logger.error(f"Error analyzing signal: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error analyzing signal: {str(e)}")

@app.post("/format-signal")
async def format_signal(request: SignalRequest):
    """Format a trading signal into a clear message using OpenAI"""
    try:
        # Create the prompt
        prompt = f"""Please format this trading signal:
- Instrument: {request.instrument}
- Direction: {request.direction.upper()}
- Entry Price: {request.entry_price}
- Stop Loss: {request.stop_loss}
- Take Profit: {request.take_profit}
- Timeframe: {request.timeframe or 'Not specified'}
- Strategy: {request.strategy or 'Not specified'}"""

        # Get AI response
        response = client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        
        formatted_message = response.choices[0].message.content.strip()
        
        return {
            "status": "success",
            "formatted_message": formatted_message
        }
        
    except Exception as e:
        logger.error(f"Error formatting signal: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error formatting signal: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
