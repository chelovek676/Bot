import os
import re
import hashlib
import threading
import asyncio
import edge_tts
import playsound

CACHE_DIR = "./tts_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

VOICE = "ru-RU-SvetlanaNeural"  

_tts_cache = {}

def normalize_text(text):
    text = re.sub(r'\b(\d+)\s*\+\s*(\d+)\b', lambda m: f"{int(m.group(1))} плюс {int(m.group(2))}", text)
    text = re.sub(r'\b(\d+)\s*-\s*(\d+)\b', lambda m: f"{int(m.group(1))} минус {int(m.group(2))}", text)
    text = re.sub(r'(\d+):(\d+)', lambda m: f"{int(m.group(1))} часов {int(m.group(2))} минут", text)
    text = re.sub(r'(\d{1,2})\.(\d{1,2})\.(\d{4})', lambda m: f"{int(m.group(1))} {int(m.group(2))} {int(m.group(3))} года", text)
    text = re.sub(r'\b(\d+)\s*(градус[аов]?|°)\b', r'\1 градусов', text)
    text = re.sub(r'\b(\d+)\s*(м/с)\b', r'\1 метров в секунду', text)
    return text

def _get_cache_path(text):
    text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
    return os.path.join(CACHE_DIR, f"{text_hash}.mp3")

async def _generate_audio(text, path):
    communicate = edge_tts.Communicate(text, VOICE)
    await communicate.save(path)

def _speak_sync(text):
    clean = normalize_text(text)
    path = _get_cache_path(clean)
    
    if clean not in _tts_cache:
        if os.path.exists(path):
            _tts_cache[clean] = path
        else:
            try:
                asyncio.run(_generate_audio(clean, path))
                _tts_cache[clean] = path
            except Exception as e:
                print(f"Ошибка TTS: {e}")
                return False
    
    try:
        playsound.playsound(_tts_cache[clean])
        return True
    except Exception as e:
        print(f"Ошибка воспроизведения: {e}")
        return False

def speak_async(text):
    thread = threading.Thread(target=_speak_sync, args=(text,), daemon=True)
    thread.start()
    return thread

def speak_sync(text):
    return _speak_sync(text)

def wait_for_tts():
    pass

def shutdown_tts():
    pass

def clear_cache():
    for f in os.listdir(CACHE_DIR):
        if f.endswith('.mp3'):
            os.remove(os.path.join(CACHE_DIR, f))
    _tts_cache.clear()
    print("Кэш TTS очищен")

def get_cache_stats():
    total_size = sum(
        os.path.getsize(os.path.join(CACHE_DIR, f)) 
        for f in os.listdir(CACHE_DIR) 
        if f.endswith('.mp3')
    )
    return {"entries": len(_tts_cache), "size_mb": round(total_size / (1024 * 1024), 2)}