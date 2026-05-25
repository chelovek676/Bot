import re
from datetime import datetime

def handle_greeting(match):
    return "Здравствуйте! Чем могу помочь?"

def handle_farewell(match):
    return "До свидания!"

def handle_weather(match):
    city = match.group(1).strip()
    return f"Погода в городе {city}: солнечно (демо-режим)."

def handle_addition(match):
    try:
        a = float(match.group(1))
        b = float(match.group(2))
        return f"Результат: {a + b}"
    except ValueError:
        return "Ошибка при вычислении. Пожалуйста, введите корректные числа."

def handle_time(match):
    current_time = datetime.now().strftime("%H:%M:%S")
    return f"Сейчас {current_time}"

def log_message(user_message, bot_response):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("chat_log.txt", "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] Пользователь: {user_message} | Бот: {bot_response}\n")

patterns = [
    (re.compile(r"^(привет|здравствуй|добрый день)$", re.IGNORECASE), handle_greeting),
    (re.compile(r"^(пока|до свидания)$", re.IGNORECASE), handle_farewell),
    (re.compile(r"погода в ([а-яА-Яa-zA-Z\- ]+)", re.IGNORECASE), handle_weather),
    (re.compile(r"(\d+)\s*\+\s*(\d+)"), handle_addition),
    (re.compile(r"(который час|сколько времени|текущее время|время)", re.IGNORECASE), handle_time),
]

class ChatBot:
    def __init__(self):
        self.patterns = patterns
        print("Бот инициализирован. Логирование включено.")

    def process(self, message: str):
        message = message.strip()
        
        for pattern, handler in self.patterns:
            match = pattern.search(message)
            if match:
                response = handler(match)
                log_message(message, response)
                return response
        
        response = "Я не понимаю запрос."
        log_message(message, response)
        return response

def main():
    bot = ChatBot()
    print("Чат-бот запущен. Команды:")
    print("- Приветствие: 'привет', 'здравствуй', 'добрый день'")
    print("- Прощание: 'пока', 'до свидания'")
    print("- Погода: 'погода в Москва'")
    print("- Калькулятор: '5 + 3'")
    print("- Время: 'который час'")
    print("- Выход: 'выход' или Ctrl+C")
    print("-" * 40)
    
    try:
        while True:
            user_input = input("Вы: ")
            
            if user_input.lower() in ["выход", "exit", "quit"]:
                print("Бот: До свидания!")
                break
                
            response = bot.process(user_input)
            print("Бот:", response)
            
    except KeyboardInterrupt:
        print("\nБот: Работа завершена.")

if __name__ == "__main__":
    main()