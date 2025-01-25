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
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class SignalRequest(BaseModel):
    instrument: str
    direction: str
    entry_price: float
    stop_loss: float
    take_profit: float
    timeframe: Optional[str]
    strategy: Optional[str]

@app.post("/format-signal")
async def format_signal(request: SignalRequest):
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
        response = await client.chat.completions.create(
            model="gpt-4-0125-preview",
            messages=[{
                "role": "system",
                "content": "You are a professional trading signal formatter. Your job is to take raw trading signals and format them into clear, engaging messages for subscribers."
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
