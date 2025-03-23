from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from langchain_core.tools import tool
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings

app = FastAPI()
import sqlite3
import pandas as pd

class Question(BaseModel):
    question: str

from langchain_community.chat_models import ChatOpenAI
class ChatOpenRouter(ChatOpenAI):
    openai_api_base: str
    openai_api_key: str
    model_name: str

    def __init__(self,
                 model_name: str,
                 openai_api_key: str,
                 openai_api_base: str = "https://openrouter.ai/api/v1",
                 **kwargs):
        openai_api_key = openai_api_key
        super().__init__(openai_api_base=openai_api_base,
                         openai_api_key=openai_api_key,
                         model_name=model_name, **kwargs)
chat = ChatOpenRouter(model_name="google/gemini-2.0-flash-lite-preview-02-05:free", openai_api_key="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")

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent

from langchain_chroma import Chroma

vector_store = Chroma(
    embedding_function=embeddings,
    persist_directory="./chroma_langchain",  # Where to save data locally, remove if not necessary
)

@tool
def get_booking_details(query):
    """
    Executes an SQL query on the 'booking_data.db' SQLite database and returns the result as a Pandas DataFrame.
    Use this function to respond to any question about the booking data.

    Parameters:
    query (str): The SQL query to execute.

    Returns:
    pd.DataFrame: A DataFrame containing the query results.
    """
    # Step 1: Connect to SQLite Database (Creates 'booking_data.db' if it doesn’t exist)
    conn = sqlite3.connect("booking_data.db")
    cursor = conn.cursor()

    # Step 2: Execute the query and fetch results into a Pandas DataFrame
    data = pd.read_sql(query, conn)

    # Step 3: Close the database connection
    conn.close()

    # Step 4: Return the query result as a DataFrame
    return data

def get_analytics(query):
    """
    Performs a similarity search using the vector store and returns the top-k most relevant results.

    Parameters:
    query (str): The search query for retrieving relevant results.

    Returns:
    list: A list of the top-k most similar results from the vector store.
    """
    # Step 1: Perform a similarity search in the vector store
    data = vector_store.similarity_search(query, k=3)
    prompt = "Answer the User question :{query} on the basis of the context provided: {data}"
    answer = chat.invoke(prompt)

    # Step 2: Return the top 3 most relevant results
    return answer.content

prompt = PromptTemplate.from_template('''
Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Columns in DB: ['hotel', 'is_canceled', 'lead_time', 'arrival_date_year',
       'arrival_date_month', 'arrival_date_week_number',
       'arrival_date_day_of_month', 'stays_in_weekend_nights',
       'stays_in_week_nights', 'adults', 'children', 'babies', 'meal',
       'country', 'market_segment', 'distribution_channel',
       'is_repeated_guest', 'previous_cancellations',
       'previous_bookings_not_canceled', 'reserved_room_type',
       'assigned_room_type', 'booking_changes', 'deposit_type', 'agent',
       'company', 'days_in_waiting_list', 'customer_type', 'adr',
       'required_car_parking_spaces', 'total_of_special_requests',
       'reservation_status', 'reservation_status_date', 'total_nights',
       'total_guests', 'revenue']
DB Name: booking
Pass the sql statement as string object to the function.
Begin!

Question: {input}
Thought:{agent_scratchpad}
''')
def doc_qa(question):
    
    tools = [get_booking_details]

    agent = create_react_agent(chat, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, handle_parsing_errors=True)

    answer = agent_executor.invoke({"input": question})

    
    return answer['output']


@app.post("/ask")
async def ask_question(question: Question):
    if not question.question:
        raise HTTPException(status_code=400, detail="No question provided")

    answer = doc_qa(question.question)
    return JSONResponse(content={"answer": answer})

@app.post("/analytics")
async def analytics_question(question: Question):
    if not question.question:
        raise HTTPException(status_code=400, detail="No question provided")

    answer = get_analytics(question.question)
    return JSONResponse(content={"answer": answer})



if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="debug")
