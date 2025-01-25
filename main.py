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

🚨 *New Trading Signal* 🚨

*Instrument:* [instrument]
*Action:* [BUY/SELL] 📈/📉

*Entry Price:* [price]
*Stop Loss:* [price] 🛑
*Take Profit:* [price] 🎯

*Timeframe:* [timeframe]
*Strategy:* [strategy]

-------------------

*Risk Management:*
• Position size: 1-2% max
• Use proper stop loss
• Follow your trading plan

-------------------

🤖 *SigmaPips AI Verdict:*
[2-3 lines explaining why this trade setup looks promising, focusing on technical aspects and risk/reward ratio]

Remember:
- Keep it concise and professional
- Use emojis sparingly
- Format numbers with 4 decimals for forex (e.g., 1.0950)
- Add the AI verdict only at the end"""

class SignalRequest(BaseModel):
    instrument: str
    direction: str
    entry_price: float
    stop_loss: float
    take_profit: float
    timeframe: Optional[str]
    strategy: Optional[str]

@app.post("/format-signal")
def format_signal(request: SignalRequest):
    """Format a trading signal into a clear message using OpenAI."""
    try:
        # Create a prompt for the signal formatting
        prompt = f"""Format this trading signal into a clear, professional message for subscribers.
        Include all relevant information and add emoji where appropriate.
        
        Signal Data:
        - Instrument: {request.instrument}
        - Direction: {request.direction}
        - Entry Price: {request.entry_price}
        - Stop Loss: {request.stop_loss}
        - Take Profit: {request.take_profit}
        - Timeframe: {request.timeframe or "Unknown"}
        - Strategy: {request.strategy or "Custom Strategy"}
        
        Format the message to be engaging and easy to read."""

        # Call OpenAI API
        response = client.chat.completions.create(
            model="gpt-4-0125-preview",
            messages=[{
                "role": "system",
                "content": SYSTEM_PROMPT
            }, {
                "role": "user",
                "content": prompt
            }],
            temperature=0.7,
            max_tokens=500
        )
        
        formatted_message = response.choices[0].message.content
        logger.info(f"Successfully formatted signal for {request.instrument}")
        
        return {"formatted_message": formatted_message}
            
    except Exception as e:
        logger.error(f"Error formatting signal: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
