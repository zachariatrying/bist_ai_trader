import yfinance as yf
import requests
from bs4 import BeautifulSoup
from src.kap_news_scraper import KAPNewsScraper

class NewsSentimentEngine:
    """
    Real-Time KAP & Financial News Scraper & Sentiment Analysis Engine.
    Combines live news sentiment with technical model inference into a Hybrid Signal.
    Clean, institutional text output without informal emojis.
    """
    BULLISH_KEYWORDS = [
        'anlaşma', 'rekor', 'ihale', 'büyüme', 'kâr', 'ortaklık', 'temettü', 'kap', 'yatırım',
        'hedef', 'yükseliş', 'alım', 'onay', 'bedelsiz', 'yeni iş', 'tavan', 'kazanç',
        'surge', 'growth', 'profit', 'launch', 'partnership', 'contract', 'expand', 'dividend', 'strong', 'revenue'
    ]
    BEARISH_KEYWORDS = [
        'ceza', 'zarar', 'dava', 'iptal', 'düşüş', 'borç', 'soruşturma', 'istifa', 'kriz', 'kayıp',
        'satış', 'risk', 'taban', 'halka arz', 'iflas', 'uyarı',
        'loss', 'drop', 'decline', 'investigation', 'debt', 'sanction', 'warn', 'lawsuit'
    ]

    @staticmethod
    def get_news_sentiment(ticker):
        clean_ticker = ticker.replace('.IS', '').upper()
        full_ticker = clean_ticker + '.IS'

        # Fetch 100% REAL live news via KAP Scraper & yFinance
        scraped_news = KAPNewsScraper.scrape_live_ticker_news(clean_ticker)
        
        score = 0
        analyzed_items = []

        for item in scraped_news:
            title = item.get('headline', '')
            source = item.get('source', 'Finans Basını')
            item_score = 0
            title_lower = title.lower()
            
            for kw in NewsSentimentEngine.BULLISH_KEYWORDS:
                if kw in title_lower:
                    item_score += 20
            for kw in NewsSentimentEngine.BEARISH_KEYWORDS:
                if kw in title_lower:
                    item_score -= 25

            sentiment_tag = "POZİTİF" if item_score > 0 else ("NEGATİF" if item_score < 0 else "NÖTR")
            analyzed_items.append({
                'headline': title,
                'source': source,
                'sentiment': sentiment_tag,
                'score': item_score
            })
            score += item_score

        net_score = min(100, max(-100, score))
        if net_score >= 30:
            overall_sentiment = "ÇOK POZİTİF [KAP / Haber Desteği Var]"
        elif net_score >= 10:
            overall_sentiment = "POZİTİF [Olumlu Haber Akışı]"
        elif net_score <= -20:
            overall_sentiment = "NEGATİF [Riskli Haber Akışı]"
        else:
            overall_sentiment = "NÖTR [Dengeli Akış]"

        return {
            'ticker': clean_ticker,
            'net_score': net_score,
            'overall_sentiment': overall_sentiment,
            'news_list': analyzed_items
        }

    @staticmethod
    def calculate_hybrid_signal(technical_prob, sentiment_score):
        """
        Combines Technical Model Probability (0.0 to 1.0) and Sentiment Score (-100 to +100).
        """
        norm_sentiment = (sentiment_score + 100) / 200.0  # Scale -100..+100 to 0.0..1.0
        hybrid_weight = (technical_prob * 0.70) + (norm_sentiment * 0.30)
        hybrid_score_pct = hybrid_weight * 100.0

        if hybrid_score_pct >= 65:
            final_signal = "GÜÇLÜ AL [Teknik + Haber Uyumlu]"
        elif hybrid_score_pct >= 52:
            final_signal = "AL [Teknik veya Haber Destekli]"
        elif hybrid_score_pct <= 38:
            final_signal = "SAT [Zayıf Teknik & Olumsuz Akış]"
        else:
            final_signal = "TUT / NÖTR [Kararsız Bölge]"

        return {
            'hybrid_score_pct': round(hybrid_score_pct, 1),
            'technical_weight_pct': round(technical_prob * 100, 1),
            'sentiment_weight_pct': round(norm_sentiment * 100, 1),
            'final_signal': final_signal
        }
