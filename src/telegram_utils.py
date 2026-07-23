import requests
import json
import os

CONFIG_FILE = "telegram_config.json"

def load_tg_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_tg_config(token, chat_id):
    with open(CONFIG_FILE, 'w') as f:
        json.dump({"token": token, "chat_id": chat_id}, f)
    return True

def send_telegram_alert(message):
    """Sends a formatted HTML message using the configured Telegram bot."""
    config = load_tg_config()
    token = config.get("token")
    chat_id = config.get("chat_id")
    
    if not token or not chat_id:
        return False, "Telegram token veya Chat ID ayarları eksik."
        
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "HTML",
        "disable_web_page_preview": True
    }
    try:
        resp = requests.post(url, json=payload, timeout=10)
        return resp.status_code == 200, resp.json()
    except Exception as e:
        return False, str(e)

def format_and_send_pattern_alert(ticker, pattern_name, price, target_price=None):
    """Formats and dispatches geometric pattern detection alerts."""
    msg = f"<b>🚨 BorsaNeuron Pattern Alert</b>\n\n"
    msg += f"<b>Hisse:</b> #{ticker}\n"
    msg += f"<b>Formasyon:</b> {pattern_name}\n"
    msg += f"<b>Mevcut Fiyat:</b> {price:.2f} TL\n"
    if target_price:
        msg += f"<b>Hedef Fiyat:</b> {target_price:.2f} TL\n"
    msg += f"\n<i>⚡ BorsaNeuron AI Automation System</i>"
    return send_telegram_alert(msg)

def format_and_send_signal_alert(ticker, signal_name, hybrid_score, win_rate=None):
    """Formats and dispatches Hybrid AI Buy/Sell signal alerts."""
    msg = f"<b>🎯 BorsaNeuron Hybrid AI Signal</b>\n\n"
    msg += f"<b>Hisse:</b> #{ticker}\n"
    msg += f"<b>Sinyal:</b> {signal_name}\n"
    msg += f"<b>Hibrit Skor:</b> %{hybrid_score}\n"
    if win_rate:
        msg += f"<b>Tarihsel Başarı (%Win-Rate):</b> %{win_rate:.1f}\n"
    msg += f"\n<i>📊 BorsaNeuron Decision Workstation</i>"
    return send_telegram_alert(msg)
