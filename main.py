import os
import json
import logging
from typing import List, Dict, Any, Optional
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

class NewsRequest(BaseModel):
    instrument: str
    articles: List[Dict[str, Any]]

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

@app.post("/analyze-news")
async def analyze_news(request: NewsRequest):
    """Analyze news articles and provide sentiment analysis."""
    try:
        # Create a prompt for news analysis
        articles_text = "\n\n".join([
            f"Title: {article.get('title', 'No Title')}\n"
            f"Content: {article.get('content', 'No Content')}\n"
            f"Source: {article.get('source', 'Unknown')}\n"
            f"Date: {article.get('date', 'Unknown')}"
            for article in request.articles
        ])
        
        prompt = f"""Analyze these news articles about {request.instrument} and provide:
        1. A concise summary of the key points
        2. Overall market sentiment (Bullish/Bearish/Neutral)
        3. Key factors influencing the market
        4. Potential impact on trading decisions
        
        News Articles:
        {articles_text}
        
        Format your response in a clear, structured way with emoji where appropriate."""

        # Call OpenAI API for analysis
        analysis_response = await client.chat.completions.create(
            model="gpt-4-0125-preview",
            messages=[{
                "role": "system",
                "content": "You are a financial news analyst specializing in market sentiment analysis. Your job is to analyze news articles and provide clear, actionable insights for traders."
            }, {
                "role": "user",
                "content": prompt
            }],
            temperature=0.5,
            max_tokens=1000
        )
        
        # Get a specific trading verdict
        verdict_prompt = f"""Based on the news analysis, provide a clear trading verdict for {request.instrument}.
        Previous analysis: {analysis_response.choices[0].message.content}
        
        Format your response as a JSON with these fields:
        - verdict: (STRONG_BUY, BUY, NEUTRAL, SELL, STRONG_SELL)
        - confidence: (percentage between 0-100)
        - key_reason: (brief explanation)"""
        
        verdict_response = await client.chat.completions.create(
            model="gpt-4-0125-preview",
            messages=[{
                "role": "system",
                "content": "You are a trading advisor that provides clear, decisive trading verdicts based on news analysis."
            }, {
                "role": "user",
                "content": verdict_prompt
            }],
            temperature=0.3,
            max_tokens=200
        )
        
        logger.info(f"Successfully analyzed news for {request.instrument}")
        
        return {
            "analysis": analysis_response.choices[0].message.content,
            "verdict": json.loads(verdict_response.choices[0].message.content)
        }
            
    except Exception as e:
        logger.error(f"Error analyzing news: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/get-market-context")
async def get_market_context(instrument: str):
    """Get broader market context and potential correlations."""
    try:
        prompt = f"""Provide a brief market context analysis for {instrument}. Consider:
        1. Related instruments and their performance
        2. Key market drivers
        3. Important technical levels
        4. Upcoming economic events that might impact the instrument
        
        Format your response in a clear, concise way."""

        response = await client.chat.completions.create(
            model="gpt-4-0125-preview",
            messages=[{
                "role": "system",
                "content": "You are a market analyst providing context and correlation analysis for trading instruments."
            }, {
                "role": "user",
                "content": prompt
            }],
            temperature=0.5,
            max_tokens=500
        )
        
        logger.info(f"Successfully got market context for {instrument}")
        return {"market_context": response.choices[0].message.content}
            
    except Exception as e:
        logger.error(f"Error getting market context: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
