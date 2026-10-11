# 🤖 Lisa: Offline WhatsApp AI Assistant (RAG Pipeline)

An advanced, offline-first Retrieval-Augmented Generation (RAG) AI assistant integrated directly into WhatsApp Web. This project reads your exported WhatsApp chats, processes them into a local vector database, and uses a Large Language Model (LLM) to answer queries contextually right inside your WhatsApp chats.

Built for the **Next_Techz** community.

## 🌟 Key Features
*   **100% Local Vector DB:** Uses ChromaDB to store embeddings locally, ensuring your private chats never go to a cloud database[cite: 1, 2].
*   **Blazing Fast Inference:** Uses Groq API with the `openai/gpt-oss-120b` model for ultra-fast, zero-cost LLM reasoning[cite: 1, 2].
*   **Two-Stage Retrieval:** Prevents AI hallucination by strictly filtering exact dates and friend names before running vector similarity searches.
*   **Smart Typing Guard:** Playwright DOM observer ensures the bot waits for you to finish typing before it triggers a search.
*   **Robust Offline Sync:** Automatically scrapes and injects new background messages into the database while the bot is active.

## 🛠️ Tech Stack
*   **Python 3.9+**
*   **LangChain** (RAG Pipeline & Text Chunking)
*   **ChromaDB** (Local Vector Database)[cite: 1, 2]
*   **Groq API** (LLM Inference)[cite: 1, 2]
*   **HuggingFace Embeddings** (`all-MiniLM-L6-v2`)[cite: 1, 2]
*   **Playwright** (Headless Browser Automation)

---

## 🚀 Step-by-Step Installation

### 1. Clone the Repository
```bash
git clone https://github.com/nilj07/WhatsApp-AI-Assistant-RAG.git
cd WhatsApp-AI-Assistant-RAG
