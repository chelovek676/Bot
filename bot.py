import re
import joblib
import sqlite3
from datetime import datetime
from collections import defaultdict
import spacy
import numpy as np


pipeline = joblib.load("intent_model_embeddings.pkl")  
label_encoder = joblib.load("label_encoder.pkl")        
nlp = joblib.load("nlp_model_embeddings.pkl")         



class DialogState:
    START = "START"
    WAIT_CITY = "WAIT_CITY"

user_states = defaultdict(lambda: DialogState.START)

def get_state(user_id):
    return user_states[user_id]

def set_state(user_id, state):
    user_states[user_id] = state

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

API_KEY = "0c969c7b82a3f98ed1b475fd8b734ade"

def get_weather_simple(city):
    import requests
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
    except Exception as e:
        return f"Ошибка соединения: {e}"

def predict_intent(text):

    doc = nlp(text)
    vectors = [token.vector for token in doc if token.has_vector and token.vector_norm != 0]
    
    if vectors:
        text_vector = np.mean(vectors, axis=0).reshape(1, -1)
    else:
        text_vector = np.zeros((1, nlp.vocab.vectors_length))
    
    probabilities = pipeline.predict_proba(text_vector)[0]
    intent_idx = pipeline.predict(text_vector)[0]
    intent = label_encoder.inverse_transform([intent_idx])[0]
    confidence = max(probabilities)
    
    return intent, confidence

def extract_city(text):
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ in ["GPE", "LOC"]:
            return ent.text.strip()
    return None

def handle_greeting(user_name=None):
    return f"Здравствуй, {user_name}!" if user_name else "Здравствуйте!"

def handle_farewell(user_name=None):
    return f"До свидания, {user_name}!" if user_name else "До свидания!"

def handle_addition(text):
    match = re.search(r"(\d+)\s*\+\s*(\d+)", text)
    if match:
        a, b = float(match.group(1)), float(match.group(2))
        return f"{a} + {b} = {a + b}"
    return "Не удалось вычислить"

def handle_time():
    return f"Сейчас: {datetime.now().strftime('%H:%M:%S')}"

def handle_weather(text, user_id):
    city = extract_city(text)
    if city:
        return get_weather_simple(city)
    else:
        set_state(user_id, DialogState.WAIT_CITY)
        return "В каком городе вас интересует погода?"

def handle_message(text, user_name="аноним"):
    user_id = user_name
    state = get_state(user_id)
    
    if state == DialogState.WAIT_CITY:
        city = text.strip()
        set_state(user_id, DialogState.START)
        response = get_weather_simple(city)
        log_message_db(user_name, text, response)
        return response
    
    intent, confidence = predict_intent(text)
    
    if confidence < 0.5:
        response = "Не уверен, что понял. Можете перефразировать?"
        log_message_db(user_name, text, response)
        return response
    
    if intent == "greeting":
        response = handle_greeting(user_name)
    elif intent == "farewell":
        response = handle_farewell(user_name)
    elif intent == "weather":
        response = handle_weather(text, user_id)
    elif intent == "addition":
        response = handle_addition(text)
    elif intent == "time":
        response = handle_time()
    else:
        response = "Не понял запрос. Попробуйте: привет, погода в [город], 5+3, время"
    
    log_message_db(user_name, text, response)
    return response

class ChatBot:
    def __init__(self):
        init_db()
        save_user("system")
    
    def process(self, message, user_name):
        return handle_message(message, user_name)

def main():
    init_db()
    bot = ChatBot()
    
    user_name = input("Как тебя зовут? ").strip() or "аноним"
    save_user(user_name)
    
    print(f"\nПривет, {user_name}!")
    print("   - привет / здравствуй — приветствие")
    print("   - пока / до свидания — прощание") 
    print("   - погода в [город] — прогноз погоды")
    print("   - 5+3 — калькулятор")
    print("   - время / который час — текущее время")
    print("   - выход / quit — завершить")
    print("-" * 50)
    
    while True:
        try:
            user_input = input(f"\n{user_name}: ").strip()
            if user_input.lower() in ["выход", "exit", "quit", "q"]:
                print(f"Бот: Пока, {user_name}!")
                break
            
            if not user_input:
                continue
                
            response = bot.process(user_input, user_name)
            print(f"Бот: {response}")
            
        except KeyboardInterrupt:
            print(f"\nБот: Пока, {user_name}!")
            break
        except Exception as e:
            print(f"Бот: Ошибка — {e}")

if __name__ == "__main__":
    main()