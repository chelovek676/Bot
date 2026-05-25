import re
from datetime import datetime
import requests
import sqlite3

API_KEY = "0c969c7b82a3f98ed1b475fd8b734ade"

def init_db():
    conn = sqlite3.connect("bot.db")
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE
        )
    """)

    conn.commit()
    conn.close()

def save_user(name):
    conn = sqlite3.connect("bot.db")
    cursor = conn.cursor()

    cursor.execute("INSERT OR REPLACE INTO users (name) VALUES (?)",
                   (name,))

    conn.commit()
    conn.close()

def get_user(name):
    conn = sqlite3.connect("bot.db")
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM users WHERE name=?", (name,))
    result = cursor.fetchone()
    conn.close()

    return result[0] if result else None

def log_message_db(user_name, user_message, bot_response):
    conn = sqlite3.connect("bot.db")
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_name TEXT,
            user_message TEXT,
            bot_response TEXT,
            timestamp TIMESTAMP
        )
    """)
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    cursor.execute("""
        INSERT INTO messages (user_name, user_message, bot_response, timestamp)
        VALUES (?, ?, ?, ?)
    """, (user_name, user_message, bot_response, timestamp))
    
    conn.commit()
    conn.close()

def get_weather_simple(city):
    url = "http://api.weatherstack.com/current"
    
    params = {
        "access_key": API_KEY,
        "query": city,
        "units": "m"
    }
    
    try:
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        
        data = response.json()
        
        if data.get("success") is False:
            return "Ошибка соединения с погодным сервисом."
        
        current = data.get("current", {})
        
        if not current:
            return "Ошибка соединения с погодным сервисом."
        
        temperature = current.get("temperature", "N/A")
        wind_speed = current.get("wind_speed", 0)
        
        return f"{city}: {temperature}°C, ветер {wind_speed} м/с"
        
    except requests.RequestException:
        return "Ошибка соединения с погодным сервисом."

def handle_weather(match, user_name):
    city = match.group(1).strip()
    return get_weather_simple(city)

def handle_greeting(match, user_name):
    return f"Здравствуй, {user_name}!"

def handle_farewell(match, user_name):
    return f"Пока, {user_name}!"

def handle_addition(match, user_name):
    try:
        a = float(match.group(1))
        b = float(match.group(2))
        return f"= {a + b}"
    except:
        return "Ошибка"

def handle_time(match, user_name):
    return datetime.now().strftime("%H:%M:%S")

patterns = [
    (re.compile(r"^(привет|здравствуй)$", re.IGNORECASE), handle_greeting),
    (re.compile(r"^(пока|до свидания)$", re.IGNORECASE), handle_farewell),
    (re.compile(r"погода в ([а-яА-Яa-zA-Z\- ]+)", re.IGNORECASE), handle_weather),
    (re.compile(r"погода ([а-яА-Яa-zA-Z\- ]+)", re.IGNORECASE), handle_weather),
    (re.compile(r"(\d+)\s*\+\s*(\d+)"), handle_addition),
    (re.compile(r"(время|который час)", re.IGNORECASE), handle_time),
]

class ChatBot:
    def init(self):
        self.patterns = patterns
        init_db()

    def process(self, message: str, user_name: str):
        message = message.strip()
        
        save_user(user_name)
        
        for pattern, handler in self.patterns:
            match = pattern.search(message)
            if match:
                response = handler(match, user_name)
                log_message_db(user_name, message, response)
                return response
        
        response = "чё, попутал?"
        log_message_db(user_name, message, response)
        return response

def main():
    bot = ChatBot()
    
    user_name = input("Как тебя зовут? ").strip()
   
    if not user_name:
        user_name = "аноним"
    
    print(f"\nПривет, {user_name}!")
    print("Команды: привет, пока, время, погода [город], 5+3")
    print("-" * 30)
    
    while True:
        try:
            user_input = input(f"{user_name}: ")
            
            if user_input.lower() in ["выход", "exit"]:
                print(f"Бот: Пока, {user_name}!")
                break
                
            response = bot.process(user_input, user_name)
            print("Бот:", response)
            
        except KeyboardInterrupt:
            print(f"\nБот: Пока, {user_name}!")
            break
        except Exception as e:
            print(f"Бот: Ошибка - {e}")

if __name__ == "main":
    main()