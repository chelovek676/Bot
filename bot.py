import re
from datetime import datetime
import requests
import sqlite3
import spacy

API_KEY = "0c969c7b82a3f98ed1b475fd8b734ade"

nlp = spacy.load("ru_core_news_sm")

def init_db():
    conn = sqlite3.connect("bot.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_name TEXT,
            user_message TEXT,
            bot_response TEXT,
            timestamp TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def save_user(name):
    conn = sqlite3.connect("bot.db")
    cursor = conn.cursor()
    cursor.execute("INSERT OR REPLACE INTO users (name) VALUES (?)", (name,))
    conn.commit()
    conn.close()

def log_message_db(user_name, user_message, bot_response):
    conn = sqlite3.connect("bot.db")
    cursor = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("""
        INSERT INTO messages (user_name, user_message, bot_response, timestamp)
        VALUES (?, ?, ?, ?)
    """, (user_name, user_message, bot_response, timestamp))
    conn.commit()
    conn.close()

def get_weather_simple(city):
    url = "http://api.weatherstack.com/current"
    params = {"access_key": API_KEY, "query": city, "units": "m"}
    
    try:
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        if data.get("success") is False or "error" in data:
            return "Ошибка API погоды"
        
        current = data.get("current", {})
        if not current:
            return f"Нет данных для '{city}'"
        
        temperature = current.get("temperature", "N/A")
        wind_speed = current.get("wind_speed", 0)
        return f"{city}: {temperature}°C, ветер {wind_speed} м/с"
    except Exception:
        return "Ошибка соединения с погодным сервисом"

def _get_weather(city):
    return get_weather_simple(city)

def handle_greetings(match, user_name=None):
    return "Здравствуй!"

def handle_farewell(match, user_name=None):
    return "Пока!"

def handle_addition(match, user_name=None):
    try:
        a = float(match.group(1))
        b = float(match.group(2))
        return f"= {a + b}"
    except:
        return "Ошибка"

def handle_time(match=None, user_name=None):
    return datetime.now().strftime("%H:%M:%S")

patterns = [
    (re.compile(r"^(привет|здравствуй|добрый день)$", re.IGNORECASE), handle_greetings),
    (re.compile(r"^(чао|пока|до свидания)$", re.IGNORECASE), handle_farewell),
    (re.compile(r"^погода в ([а-яА-Яa-zA-Z\-]+)", re.IGNORECASE), None),  
    (re.compile(r"^(\d+)\s*\+\s*(\d+)$"), handle_addition),
    (re.compile(r"^(время|который час)", re.IGNORECASE), handle_time),
]

def handle_message(text, user_name="аноним"):
    doc = nlp(text)
    city = None

    for ent in doc.ents:
        if ent.label_ in ["GPE", "LOC"]:
            city = ent[0].lemma_  
            break

    if any(token.lemma_.lower() == "погода" for token in doc) and city:
        response = _get_weather(city)
        log_message_db(user_name, text, response)
        return response

    for pattern, handler in patterns:
        match = pattern.search(text)
        if match and handler is not None:
            response = handler(match, user_name)
            log_message_db(user_name, text, response)
            return response

   
    response = "Че, попутал"
    log_message_db(user_name, text, response)
    return response

class ChatBot:
    def __init__(self):
        init_db()
    
    def process(self, message, user_name):
        return handle_message(message, user_name)
    
def main():
    init_db()
    bot = ChatBot()
    
    user_name = input("Как тебя зовут? ").strip() or "аноним"
    print(f"\nПривет, {user_name}!")
    print("Команды: привет, пока, погода в [город], 5+3, время")
    print("-" * 40)
    
    while True:
        try:
            user_input = input(f"{user_name}: ").strip()
            if user_input.lower() in ["выход", "exit", "quit"]:
                print(f"Бот: Пока, {user_name}!")
                break
            
            response = bot.process(user_input, user_name)
            print(f"Бот: {response}")
            
        except KeyboardInterrupt:
            print(f"\nБот: Пока, {user_name}!")
            break
        except Exception as e:
            print(f"Бот: Ошибка - {e}")

if __name__ == "main":
    main()