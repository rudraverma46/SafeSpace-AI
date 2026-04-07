# 🧠 SafeSpace AI

SafeSpace AI is a multimodal **Stress Detection and Analysis System** that combines physiological signals, facial expressions, voice input, and psychological assessment to provide intelligent mental health insights and personalized suggestions.

- 📱 **Download Android App:** [Click Here](https://github.com/rudraverma46/SafeSpace-AI/releases/download/v1.0/safespaceai.apk)
- 🌐 **Live Demo:** [https://safespace-ai.netlify.app/](https://safespace-ai.netlify.app/)
- 📧 **Email:** [rudraverma4682@gmail.com](mailto:rudraverma4682@gmail.com)
- 💼 **LinkedIn:** [Rudra Verma](https://www.linkedin.com/in/rudra-verma-b03330324/)
- 🎥 **YouTube:** [Watch Project Demo](https://www.youtube.com/watch?v=_-fJuH7gFWA)

---

## 🚀 Overview

SafeSpace AI is designed as a **privacy-first digital mental wellness system** that enables users to perform a guided “check-in” using multiple data sources.

The system analyzes:

* Facial expressions
* Voice recordings
* Physiological signals (EDA, HRV, Temperature, Accelerometer)
* DASS-21 psychological questionnaire

These inputs are fused using intelligent algorithms to generate:

* Stress score
* Stress classification (eustress/distress)
* Personalized AI-driven suggestions

---

## ✨ Key Features

### 🧠 Multimodal Stress Detection

* Combines multiple inputs for higher accuracy
* Reduces bias compared to single-source systems

### 📡 Physiological Signal Integration

* Real-time sensor data (ESP32-based)
* EDA, HRV, temperature, motion tracking

### 📸 Facial Expression Analysis

* CNN-based emotion detection model
* Detects stress-related facial cues

### 🎤 Voice Emotion Analysis

* Audio processing using MFCC features
* Deep learning model for stress inference

### 📝 Psychological Assessment (DASS-21)

* Standardized questionnaire
* Provides mental health context

### 🤖 AI Coach (LLM Integration)

* Context-aware suggestions
* Classifies stress into:

  * No Stress
  * Eustress (positive)
  * Distress (negative)

### 🔊 Voice Feedback (TTS)

* AI-generated spoken suggestions using Deepgram

### 🌐 Web-Based Interface

* Clean, modern UI deployed on Netlify
* Fully responsive and interactive

---

## 🏗️ System Architecture

```
User Input →
    ├── Facial Image → CNN Model
    ├── Audio Input → Audio Model
    ├── DASS-21 → ML Model
    ├── Sensor Data → Physiological Model
            ↓
    Multimodal Fusion Algorithm
            ↓
    Stress Score + Classification
            ↓
    LLM (Groq API)
            ↓
    Personalized Suggestions + Voice Output
```

---

## 🛠️ Tech Stack

### Frontend

* HTML, CSS, JavaScript
* Responsive UI
* Web APIs (Camera, Mic, Bluetooth)

### Backend

* FastAPI (Python)
* REST API for processing

### Machine Learning

* TensorFlow / Keras
* OpenCV (image processing)
* Librosa (audio processing)
* Scikit-learn (scaling & preprocessing)

### AI & APIs

* Groq API (LLM)
* Deepgram (Speech-to-Text + Text-to-Speech)

### Hardware

* ESP32 (sensor interface)
* Physiological sensors (EDA, HRV, Temp, ACC)

### Deployment

* Frontend: Netlify
* Backend: Hugging Face Spaces

---

## ⚙️ Setup Instructions

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/SafeSpace-AI.git
cd SafeSpace-AI
```

### 2. Create Environment File

```bash
GROQ_API_KEY=your_api_key
DEEPGRAM_API_KEY=your_api_key
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run Backend

```bash
uvicorn main:app --reload
```

---


## 📦 Models

Trained models are not included in this repository due to size constraints and ongoing research work.

The models and detailed implementation will be released after the publication of the associated research paper.


📌 Model release: Coming soon after publication

---

## 🔐 Security & Privacy

* No API keys stored in repository
* Uses environment variables
* No user data stored permanently
* Designed as a privacy-first system

Modern AI systems emphasize privacy and secure handling of user data, especially in sensitive applications like mental health. ([Netlify Docs][1])

---

## 🎯 Use Cases

* Mental health monitoring
* Student stress analysis
* Workplace well-being systems
* Smart healthcare applications

---

## 📌 Project Type

Research Project

---

## 🙏 Acknowledgements

* Open-source ML libraries
* Deepgram API
* Groq LLM API

---

## 📬 Contact

For queries or collaboration:
- 📧 **Email:** [rudraverma4682@gmail.com](mailto:rudraverma4682@gmail.com)
- 💼 **LinkedIn:** [Rudra Verma](https://www.linkedin.com/in/rudra-verma-b03330324/)
---

## ⭐ Future Improvements

* Mobile app integration
* Real-time wearable sync
* Improved model accuracy
* Personalized long-term tracking

---

