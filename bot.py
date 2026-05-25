import re
import joblib
import sqlite3
import torch
import os
from datetime import datetime
from collections import defaultdict
import spacy
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = "./bert_intent_model"

if not os.path.exists(MODEL_PATH):
    print(f"Ошибка: Модель не найдена в {MODEL_PATH}")
    print("Запустите сначала: python train_bert_intent.py")
    exit()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Загрузка модели на {DEVICE}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(DEVICE)
model.eval()

metadata = joblib.load(f"{MODEL_PATH}/metadata.pkl")
id2label = metadata["id2label"]
max_length = metadata.get("max_length", 64)

nlp_ner = spacy.load("ru_core_news_sm")

bert_cache = {}

def bert_vector(text):
    inputs = tokenizer(
        text, 
        return_tensors="pt", 
        truncation=True, 
        max_length=max_length,
        padding=True
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1)[0]
    predicted_class = torch.argmax(logits, dim=1).item()
    
    intent = id2label[predicted_class]
    confidence = probabilities[predicted_class].item()
    
    return intent, confidence

def cached_bert_vector(text):
    if text not in bert_cache:
        bert_cache[text] = bert_vector(text)
    return bert_cache[text]

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
    return cached_bert_vector(text)

def extract_city(text):
    doc = nlp_ner(text)
    for ent in doc.ents:
        if ent.label_ in ["GPE", "LOC"]:
            return ent.text.strip()
    return None

def greeting_skill(user_name=None):
    return f"Здравствуй, {user_name}!" if user_name else "Здравствуйте!"

def farewell_skill(user_name=None):
    return f"До свидания, {user_name}!" if user_name else "До свидания!"

def addition_skill(text):
    match = re.search(r"(\d+)\s*[+плюс]\s*(\d+)", text, re.IGNORECASE)
    if match:
        a, b = float(match.group(1)), float(match.group(2))
        return f"{a} + {b} = {a + b}"
    return "Не удалось вычислить"

def time_skill():
    return f"Сейчас: {datetime.now().strftime('%H:%M:%S')}"

def weather_skill(text, user_id):
    city = extract_city(text)
    if city:
        return get_weather_simple(city)
    else:
        set_state(user_id, DialogState.WAIT_CITY)
        return "В каком городе вас интересует погода?"

def smalltalk_skill(text):
    text_lower = text.lower()
    
    if any(w in text_lower for w in ["как дела", "как ты", "как жизнь", "как поживаешь"]):
        return "Спасибо, всё отлично! А у вас как?"
    
    if any(w in text_lower for w in ["спасибо", "благодарю"]):
        return "Всегда рад помочь!"
    
    if any(w in text_lower for w in ["расскажи анекдот", "пошути", "развлеки"]):
        return "Еврей нашёл кошелек, а там мало"

    if any(w in text_lower for w in ["понял", "ок", "хорошо", "ладно", "ясно", "ага", "угу"]):
        return "Отлично! Чем ещё могу помочь?"
    
    return "Понял вас. Чем ещё могу быть полезен?"

def fallback():
    return "Не понял запрос. Попробуйте: привет, погода в [город], 5+3, время, какое сегодня число, как дела"

def route_intent(intent, text, user_id, user_name=None):
    if intent == "weather":
        return weather_skill(text, user_id)
    
    elif intent == "time":
        return time_skill()
    
    elif intent == "greeting":
        return greeting_skill(user_name)
    
    elif intent == "farewell":
        return farewell_skill(user_name)
    
    elif intent == "addition":
        return addition_skill(text)
    
    elif intent == "smalltalk":
        return smalltalk_skill(text)
    
    else:
        return fallback()

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
    
    response = route_intent(intent, text, user_id, user_name)
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
    
    print(f"Привет, {user_name}!")
    print("   - привет / здравствуй — приветствие")
    print("   - пока / до свидания — прощание") 
    print("   - погода в [город], дождь, зонт — прогноз погоды")
    print("   - 5+3, сколько будет 2+2 — калькулятор")
    print("   - время / который час — текущее время")
    print("   - какое сегодня число / дата — сегодняшняя дата")
    print("   - как дела / расскажи анекдот — поболтать")
    print("   - выход / quit — завершить")
    print("-" * 50)
    
    while True:
        try:
            user_input = input(f"{user_name}: ").strip()
            if user_input.lower() in ["выход", "exit", "quit", "q"]:
                print(f"Бот: Пока, {user_name}!")
                break
            
            if not user_input:
                continue
                
            response = bot.process(user_input, user_name)
            print(f"Бот: {response}")
            
        except KeyboardInterrupt:
            print(f"Бот: Пока, {user_name}!")
            break
        except Exception as e:
            print(f"Бот: Ошибка — {e}")

if __name__ == "__main__":
    main()