import whisper
import sounddevice as sd
import scipy.io.wavfile as wavfile
import numpy as np
import os
import tempfile
import re
import time

MODEL_SIZE = "base"
SAMPLE_RATE = 16000
RECORD_SECONDS = 5

print(f"Загрузка модели Whisper ({MODEL_SIZE})...")
model = whisper.load_model(MODEL_SIZE)

def clean_text(text):
    if not text: return ""
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'[^\w\s.,?!-]', '', text, flags=re.UNICODE)
    return text

def record_audio(duration=RECORD_SECONDS):
    print(f"Запись {duration} сек... Говорите!")
    recording = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait()
    print("Запись завершена.")


    tmp_path = os.path.join(tempfile.gettempdir(), f"whisper_rec_{int(time.time())}.wav")
    wavfile.write(tmp_path, SAMPLE_RATE, (recording * 32767).astype(np.int16))
    return tmp_path

def speech_to_text(duration=RECORD_SECONDS, language="ru"):
    try:
        audio_path = record_audio(duration)
        

        if not os.path.exists(audio_path):
            raise FileNotFoundError("Аудиофайл не был создан")
            
        result = model.transcribe(audio_path, language=language, fp16=False)
        text = clean_text(result["text"])
        

        os.unlink(audio_path)
        return text
        
    except Exception as e:
        print(f"Ошибка Whisper: {e}")
        try:
            if 'audio_path' in locals() and os.path.exists(audio_path):
                os.unlink(audio_path)
        except: pass
        return ""

def list_mics():
    print("\n Доступные устройства записи:")
    print(sd.query_devices(kind='input'))
    print("-" * 50)

def listen_once():
    return speech_to_text()