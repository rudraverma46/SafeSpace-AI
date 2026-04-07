# main.py - Your Local FastAPI Backend Server (REFACTORED LOGIC)
import base64
import re
import uvicorn
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import cv2
import librosa
import json
import os
import requests
from pathlib import Path
import uuid

# --- Configuration ---
BASE_DIR = Path(__file__).parent.resolve()
MODELS_DIR = BASE_DIR / "saved_models"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")

# --- App Initialization ---
app = FastAPI()

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Model Loading ---
try:
    facial_model = tf.keras.models.load_model(MODELS_DIR / "facial_model.h5")
    audio_model = tf.keras.models.load_model(MODELS_DIR / "audio_model.h5")
    dass21_model = tf.keras.models.load_model(MODELS_DIR / "dass211_model.h5")
    physio_model = tf.keras.models.load_model(MODELS_DIR / "physio_model.h5")
    dass21_scaler = joblib.load(MODELS_DIR / "dass211_scaler.pkl")
    physio_scaler = joblib.load(MODELS_DIR / 'physio_scaler.pkl')
    print("✅ All models and scalers loaded successfully.")
except Exception as e:
    print(f"❌ ERROR loading models/scalers: {e}")
    facial_model = audio_model = dass21_model = physio_model = None
    dass21_scaler = physio_scaler = None

# --- Transcription Function ---
def transcribe_audio_with_deepgram(audio_file_path):
    if not DEEPGRAM_API_KEY:
        print("⚠️ Deepgram API key not found. Skipping transcription.")
        return "[Transcription Disabled: API key not set]"
    try:
        with open(audio_file_path, 'rb') as audio:
            headers = {'Authorization': f'Token {DEEPGRAM_API_KEY}', 'Content-Type': 'audio/wav'}
            url = "https://api.deepgram.com/v1/listen?model=nova-2&smart_format=true"
            response = requests.post(url, headers=headers, data=audio)
            response.raise_for_status()
            result = response.json()
            return result['results']['channels'][0]['alternatives'][0]['transcript']
    except Exception as e:
        print(f"❌ Deepgram Transcription Error: {e}")
        return "[Transcription failed due to an error]"

# --- Helper & Prediction Functions ---

# 🗑️ REMOVED: The confusing get_stress_confidence function is no longer needed.
def generate_deepgram_tts(text):
    if not DEEPGRAM_API_KEY:
        return None
    
    # Remove markdown like ** or * so the AI doesn't read them out loud
    clean_text = re.sub(r'[*_#]', '', text)
    
    url = "https://api.deepgram.com/v1/speak?model=aura-asteria-en"
    headers = {
        "Authorization": f"Token {DEEPGRAM_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {"text": clean_text}
    
    try:
        # Request the audio file from Deepgram
        response = requests.post(url, headers=headers, json=payload, timeout=15)
        response.raise_for_status()
        
        # Convert the raw audio file into a Base64 text string to send to the frontend
        audio_b64 = base64.b64encode(response.content).decode('utf-8')
        return audio_b64
    except Exception as e:
        print(f"❌ Deepgram TTS Error: {e}")
        return None
def agreement_fusion(confidences):
    valid_confidences = [c for c in confidences if c != 0.5]
    if not valid_confidences: return 0.5
    if len(valid_confidences) == 1: return valid_confidences[0]
    M = len(valid_confidences)
    agree_scores = [sum(1 - abs(valid_confidences[i] - valid_confidences[j]) for j in range(M) if i != j) / (M - 1) for i in range(M)]
    sum_agree = sum(agree_scores)
    if sum_agree < 1e-9: return np.mean(valid_confidences)
    return float(np.sum(np.array(agree_scores) * np.array(valid_confidences)) / sum_agree)

# ✅ CHANGED: All predict functions now return a single float (stress probability).
def predict_facial(photo_path):
    if not facial_model: return 0.5
    try:
        img = cv2.imread(str(photo_path))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (48, 48))
        input_img = (resized / 255.0).reshape(1, 48, 48, 1)
        prediction = facial_model.predict(input_img, verbose=0)
        # The model outputs [P(Not Stressed), P(Stressed)], so we always return the second value.
        return float(prediction[0][1])
    except Exception as e:
        print(f"Facial prediction error: {e}"); return 0.5

def predict_audio(audio_file):
    if not audio_model: return 0.5
    try:
        y, sr = librosa.load(audio_file, sr=22050)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=38)
        target_length = 98
        if mfccs.shape[1] < target_length:
            mfccs = np.pad(mfccs, ((0, 0), (0, target_length - mfccs.shape[1])))
        else:
            mfccs = mfccs[:, :target_length]
        mfccs = np.expand_dims(mfccs, axis=(0, -1))
        # The model directly outputs the stress probability.
        prediction = audio_model.predict(mfccs, verbose=0)[0][0]
        return float(prediction)
    except Exception as e:
        print(f"Audio prediction error: {e}"); return 0.5

def predict_dass21(q_responses):
    if not all([dass21_model, dass21_scaler]): return 0.5
    try:
        X = np.array([float(r) for r in q_responses]).reshape(1, -1)
        X_scaled = dass21_scaler.transform(X)
        # The model directly outputs the stress probability.
        pred_prob = dass21_model.predict(X_scaled, verbose=0)[0][0]
        return float(pred_prob)
    except Exception as e:
        print(f"DASS-21 prediction error: {e}"); return 0.5

def predict_physio_from_line(line):
    if not all([physio_model, physio_scaler]): return 0.5
    try:
        data = json.loads(line)
        if all(value == 0 for value in data.values()):
            print("🧠 Detected dummy physiological data. Returning neutral 0.5 score.")
            return 0.5
        
        feature_names = ["eda_raw", "bvp_ir_raw", "temp_c", "acc_x_raw", "acc_y_raw", "acc_z_raw"]
        input_df = pd.DataFrame([[float(data.get(k, 0)) for k in feature_names]], columns=feature_names)
        input_scaled = physio_scaler.transform(input_df)
        # The model directly outputs the stress probability.
        prediction = physio_model.predict(input_scaled, verbose=0)[0][0]
        return float(prediction)
    except Exception as e:
        print(f"Physio prediction error: {e}"); return 0.5

def get_llm_suggestion(stress_score_percent, user_paragraph):
    # 1. Check if the key exists
    if not GROQ_API_KEY: 
        return "AI Coach is unavailable: API key not configured."
    
    # 2. Prevent empty inputs (which cause 400 errors)
    if not user_paragraph or user_paragraph.startswith("["):
        user_paragraph = "The user provided no spoken context for this session."

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    prompt = f"""
[INSTRUCTION]
You are a compassionate AI mental health coach trained in:
- Clinical psychology
- Cognitive behavioral therapy (CBT)
- Positive psychology
- Stress resilience coaching

The user has provided:
- Stress Score: {stress_score_percent}%
- Paragraph: "{user_paragraph}"

Your job is to analyze both the numerical stress score and the context of the user's written input.

=============================
STEP 1: Emotional Tone Analysis
=============================
Evaluate the emotional tone of the paragraph using keywords and inferred feeling:
- Positive/growth (e.g. excited, focused, determined, calm, challenged) → potential **Eustress**
- Negative/emotional overload (e.g. hopeless, drowning, burned out, can’t focus, exhausted) → **Distress**
- If unsure, choose based on emotional weight and wording.

=============================
STEP 2: Stress Classification Logic
=============================
Combine the context (tone) with the score for final stress classification.

Use these rules:
- If score < 20% and tone is calm → **No Stress**
- If score 21–39% and tone is motivated/excited → **Eustress**
- If score 21–39% and tone is overwhelmed or exhausted → **Mild Distress**
- If score 40–60% and signs of burnout/mental fatigue → **Mild Distress**
- If score 61–79% → **Moderate Distress** unless clearly optimistic
- If score ≥ 80% and tone includes words like “hopeless,” “drowning,” or “can’t take it anymore” → **Severe Distress**
- If conflicting tone and score → Favor **tone context**

=============================
STEP 3: Final Output
=============================
Respond in the following format:

1. **Likely causes of stress:** Based on the user's paragraph.
2. **Emotional tone:** 1-sentence mood interpretation.
3. **Stress classification:** Example: "Eustress – motivation-driven" or "Moderate Distress – emotional overload"
4. **Personalized suggestions (3–5 tips):**
    - These must fit the user's exact situation
    - Avoid generic health advice unless relevant (e.g. sleep, hydration)
    - If symptoms indicate need for help, suggest professional support
5. **Motivational closing line:** Encourage the user with warmth and empathy.

=============================
Tone & Style Requirements:
=============================
- Use second-person voice ("you are…", "you can…")
- Keep the total response under 300 words
- Maintain an empathetic, non-judgmental tone
- Avoid robotic or overly clinical phrasing

Respond only with the analysis. Do not repeat the instruction or user input.
[/INSTRUCTION]
"""
    
    # 3. Cleanly format the JSON payload
    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7
    }

    try:
        # 4. Make the request using the json= parameter
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        
        # 5. Catch exact API errors for debugging
        if response.status_code != 200:
            print(f"❌ Groq API Error ({response.status_code}): {response.text}")
            response.raise_for_status()
            
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        print(f"❌ Python Error in get_llm_suggestion: {e}")
        return f"Could not reach the AI coach. Please try again later."


@app.post("/analyze")
async def analyze_stress(
    dass_data: str = Form(...),
    physio_data: str = Form(...),
    image_file: UploadFile = File(...),
    audio_file: UploadFile = File(...)
):
    unique_id = uuid.uuid4()
    image_path = f"/tmp/{unique_id}_image.jpg"
    audio_path = f"/tmp/{unique_id}_audio.wav"

    with open(image_path, "wb") as f: f.write(await image_file.read())
    with open(audio_path, "wb") as f: f.write(await audio_file.read())
    
    transcribed_text = transcribe_audio_with_deepgram(audio_path)
    
    # ✅ CHANGED: Directly get the stress scores from each model.
    facial_score = predict_facial(image_path)
    audio_score = predict_audio(audio_path)
    survey_score = predict_dass21(json.loads(dass_data)['responses'])
    physio_score = predict_physio_from_line(physio_data)
    
    # ✅ CHANGED: The list of confidences is now a clean list of scores.
    confidences = [facial_score, audio_score, survey_score, physio_score]
    
    fused_score = agreement_fusion(confidences)
    stress_score_percent = round(fused_score * 100)
    overall_label = "Stressed" if fused_score >= 0.5 else "Not Stressed"
    
    llm_suggestion = get_llm_suggestion(stress_score_percent, transcribed_text)
    
    # --- NEW CODE: Generate the audio ---
    coach_audio_b64 = generate_deepgram_tts(llm_suggestion)
    
    os.remove(image_path)
    os.remove(audio_path)
    
    # --- CHANGED: Add the audio to the dictionary ---
    return { 
        "stress_level": overall_label, 
        "stress_score_percent": stress_score_percent, 
        "transcribed_text": transcribed_text, 
        "llm_suggestion": llm_suggestion,
        "coach_audio_b64": coach_audio_b64 
    }

@app.get("/")
def read_root():
    return {"status": "Safe Space Server is running."}

