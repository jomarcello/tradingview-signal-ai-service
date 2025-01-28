import os
import logging
from typing import Optional, List, Dict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
import httpx

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = """You are a trading signal formatter. Format the provided trading signal details into a clear, easy-to-read message using Telegram-native formatting.

Your message should follow this exact format:

**[INSTRUMENT] [ACTION] Signal** ([TIMEFRAME])

Entry Price: [PRICE]
Take Profit: [TP]
Stop Loss: [SL]

Strategy: [STRATEGY]

-------------------

Risk Management:
• Position size: 1-2% max
• Use proper stop loss
• Follow your trading plan

-------------------

SigmaPips AI Verdict:
[2-3 lines explaining why this trade setup looks promising, focusing on technical aspects and risk/reward ratio]"""

ANALYSIS_PROMPT = """You are a trading signal analyzer. Analyze the provided trading signal and provide a brief but insightful verdict.

Focus on:
1. Technical aspects of the setup
2. Risk/reward ratio
3. Key levels and potential market behavior

Your verdict should be 2-3 lines long and professional.
"""

MARKET_SENTIMENT_PROMPT = """You are a forex market analyst. Analyze the provided news articles and create a comprehensive market sentiment analysis for the given trading instrument.

Your analysis should follow this exact format:

Market Impact Analysis
• ECB's latest decision: [Key points from recent central bank decisions]
• Market implications: [How this affects the currency pair]
• Current trend: [Current market behavior and notable changes]

Market Sentiment
• Direction: [Bullish/Bearish towards the instrument]
• Strength: [Strong/Moderate/Weak]
• Key driver: [Main factor driving the sentiment]

Trading Implications
• Short-term outlook: [What to expect in the next few hours/days]
• Risk assessment: [Current market risks]
• Key levels: [Important support/resistance levels to watch]

Risk Factors
• [List 2-3 key events or factors that could impact the trade]"""

class SignalRequest(BaseModel):
    instrument: str
    direction: str
    entry_price: str
    stop_loss: str
    take_profit: str
    timeframe: Optional[str] = None
    strategy: Optional[str] = None
    timestamp: Optional[str] = None

class NewsRequest(BaseModel):
    instrument: str
    articles: List[Dict[str, str]]

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
        
        # Remove any markdown code blocks and 'html' text
        formatted_message = response.choices[0].message.content.strip()
        formatted_message = formatted_message.replace("```html", "").replace("```", "").strip()
        
        # Ensure all HTML tags are properly closed
        open_tags = []
        parts = formatted_message.split("<")
        result = [parts[0]]  # Add text before first tag
        
        for part in parts[1:]:
            if not part:
                continue
                
            # Handle closing tags
            if part.startswith("/"):
                tag = part[1:].split(">")[0]
                if open_tags and open_tags[-1] == tag:
                    open_tags.pop()
                result.append("<" + part)
            # Handle opening tags
            else:
                tag = part.split(">")[0].split()[0]
                if tag in ["b", "i", "u", "s", "code", "pre"]:  # Safe HTML tags
                    open_tags.append(tag)
                result.append("<" + part)
        
        # Close any remaining open tags
        for tag in reversed(open_tags):
            result.append(f"</{tag}>")
            
        formatted_message = "".join(result)
        
        return {
            "status": "success",
            "formatted_message": formatted_message
        }
        
    except Exception as e:
        logger.error(f"Error formatting signal: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error formatting signal: {str(e)}")

@app.post("/analyze-news")
async def analyze_news(request: NewsRequest):
    """Analyze news articles and generate market sentiment"""
    try:
        # If no articles provided, fetch them
        if not request.articles:
            articles = await fetch_news_articles(request.instrument)
            if not articles:
                raise HTTPException(status_code=400, detail="Could not fetch news articles")
        else:
            articles = request.articles

        # Format articles for analysis
        articles_text = "\n\n".join([
            f"Title: {article.get('title', '')}\n"
            f"Content: {article.get('content', '')}\n"
            f"Date: {article.get('date', '')}"
            for article in articles[:3]  # Only analyze last 3 articles
        ])

        # Get AI analysis
        response = client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[
                {"role": "system", "content": MARKET_SENTIMENT_PROMPT},
                {"role": "user", "content": f"Analyze these news articles for {request.instrument}:\n\n{articles_text}"}
            ],
            temperature=0.7,
            max_tokens=500
        )

        analysis = response.choices[0].message.content

        return {
            "status": "success",
            "instrument": request.instrument,
            "sentiment": analysis
        }

    except Exception as e:
        logger.error(f"Error analyzing news: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def fetch_news_articles(instrument: str) -> List[Dict[str, str]]:
    """Fetch recent news articles for the given instrument"""
    try:
        # Extract currencies from instrument (e.g., EURUSD -> EUR,USD)
        base = instrument[:3]
        quote = instrument[3:] if len(instrument) > 3 else None

        # Use a news API to fetch articles
        async with httpx.AsyncClient() as client:
            # Example using marketaux.com API (you'll need an API key)
            api_key = os.getenv("MARKETAUX_API_KEY")
            params = {
                "api_token": api_key,
                "symbols": f"{base},{quote}" if quote else base,
                "limit": 3,
                "language": "en"
            }
            response = await client.get("https://api.marketaux.com/v1/news/all", params=params)
            
            if response.status_code == 200:
                data = response.json()
                return [{
                    "title": article["title"],
                    "content": article["description"],
                    "date": article["published_at"]
                } for article in data["data"]]
            else:
                logger.error(f"News API error: {response.text}")
                return []

    except Exception as e:
        logger.error(f"Error fetching news: {str(e)}")
        return []

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
