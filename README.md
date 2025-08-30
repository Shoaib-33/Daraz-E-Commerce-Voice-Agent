# 🛒 Daraz Voice Assistant (RAG + FAISS + LiveKit)

A **real-time e-commerce voice assistant** built with  
[LlamaIndex](https://github.com/jerryjliu/llama_index) (RAG),  
[FAISS](https://github.com/facebookresearch/faiss) (vector search),  
[LiveKit](https://github.com/livekit), and [Gemini](https://ai.google/) LLMs.  

# Demo
Click below to watch a sample demo
[![Demo Video](https://img.youtube.com/vi/ahAg7CgEJxg/0.jpg)](https://www.youtube.com/watch?v=ahAg7CgEJxg)


The agent can:
- 🔎 Search & recommend products
- 🛍️ Manage a shopping cart (add, remove, clear, show)
- ⚖️ Compare products
- ✅ Place orders with spoken confirmation
- 📦 Track existing orders
- 🗣️ Converse naturally with **low-latency speech-to-speech** (Deepgram STT + Cartesia TTS)

---

## ✨ Features

- **Fast product retrieval** with FAISS vector store  
- **Voice-first experience** using LiveKit Agents  
- **Cart & order flow** with checkout confirmation  
- **RAG-powered Q&A** over product CSVs  
- **Lightweight order state machine** for safe confirmations  

---

## ⚡ Setup

### 1. Clone & Install
```bash
git clone https://github.com/yourusername/daraz-voice-assistant.git
cd daraz-voice-assistant

# Create virtual env
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
```
### 2. Create an `.env` with the follows:
```bash

GOOGLE_API_KEY=your_google_api_key
DEEPGRAM_API_KEY=your_deepgram_api_key
CARTESIA_API_KEY=your_cartesia_api_key
```
### 3. Run using 
``` bash
python app.py console
```
