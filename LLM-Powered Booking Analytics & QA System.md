**LLM-Powered Booking Analytics & QA System:**

This project is a **FastAPI-based** system that processes **hotel booking data**, extracts **insights**, and enables **Retrieval-Augmented Generation (RAG)** for answering booking-related questions.

The system supports:

1) **Analytics**: Revenue trends, cancellation rates, geographical distribution, and lead time distribution.  
2)  **Question Answering (QA)**: Uses **LLM (Google Gemini via OpenRouter)** and **ChromaDB for embeddings** to answer user queries based on booking data.  
3) **API Development**: Exposes endpoints to request analytics and query booking details.

## **Features**

* **Data Processing & Storage**:

  * Loads hotel booking data into a **SQLite database**.

  * Converts relevant booking records into **vector embeddings (ChromaDB)**.

* **Analytics Module**:

  * Computes **revenue trends, cancellation rates, and lead time distributions**.

  * Uses SQL-based queries to fetch insights from the database.

* **Retrieval-Augmented QA System**:

  * Uses **LLM (Google Gemini)** to answer booking-related queries.

  * Implements **ChromaDB** to store and retrieve relevant data efficiently.

* **FastAPI Endpoints**:

  * **`POST /analytics`** → Returns booking analytics.

  * **`POST /ask`** → Answers booking-related questions.

**Install dependencies to run the tttt.py app:**

pip install fastapi uvicorn pandas sqlite3 matplotlib seaborn chromadb openai langchain

**From your folder where the app is downloaded go to terminal, run python \<file name\>**

**Then open the link [http://127.0.0.1:8000/](http://127.0.0.1:8000/)**

**Go to [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)** 

**There you will see api end points like ASK and ANALYTICS**   
**Use those endpoints to use the app.**

This LLM-Powered Booking Analytics & QA System efficiently processes hotel booking data, extracts key insights, and enhances question-answering capabilities using Retrieval-Augmented Generation (RAG). By leveraging FastAPI, SQLite, ChromaDB, and Google Gemini and hugging face for sentence transformers.

I am using openrouter to access the LLMs, but it keeps hitting the limit so you need to keep changing the LLM I am using. 

