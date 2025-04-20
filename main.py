import cassio
from dotenv import load_dotenv
import os
import openai
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Cassandra
from langchain_openai import OpenAIEmbeddings
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic.v1 import BaseModel, Field
from typing import Literal, List
from typing_extensions import TypedDict
from langgraph.graph import END, StateGraph, START
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.schema import Document
from pprint import pprint
import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

# Load environment variables
load_dotenv()

# Set USER_AGENT for Wikipedia API
os.environ["WIKIPEDIA_USER_AGENT"] = "MyLangChainApp/1.0 (contact@example.com)"

# Initialize Wikipedia tool with custom user agent
wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(user_agent=os.getenv("WIKIPEDIA_USER_AGENT")))

# Define the routing model
class RouteQuery(BaseModel):
    datasource: Literal["vectorstore", "wiki_search", "google_search"] = Field(...)

openai_api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(openai_api_key=openai_api_key, model="gpt-4o-mini")

# 👇 Use function_calling explicitly to fix Pydantic V1 schema warning
structured_llm_router = llm.with_structured_output(RouteQuery, method="function_calling")

system = """You are expert at routing a user question to a vectorstore, wikipedia or Google search.
The vectorstore contains documents related to agents, prompt engineering and adversarial attacks.
Use the vectorstore for questions on these topics. Otherwise, use wiki-search or google search"""
route_prompt = ChatPromptTemplate.from_messages([
    ("system", system),
    ("human", "{question}"),
])
question_router = route_prompt | structured_llm_router

# Connect to Cassandra (Astra DB)
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_ID = os.getenv("ASTRA_DB_ID")

try:
    cassio.init(token=ASTRA_DB_APPLICATION_TOKEN, database_id=ASTRA_DB_ID)
    print("✅ Successfully connected to Astra DB using CassIO.")
except Exception as e:
    print("❌ Failed to connect:", str(e))

# Set OpenAI API key
openai.api_key = openai_api_key

# Load and split documents
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

docs = [WebBaseLoader(url).load() for url in urls]
doc_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=500, chunk_overlap=0
)
docs_split = text_splitter.split_documents(doc_list)

# Initialize embedding and vector store
embeddings = OpenAIEmbeddings()
astra_vector_store = Cassandra(
    embedding=embeddings,
    session=None,
    keyspace=None,
    table_name="multi_agent_demo"
)
astra_vector_store.add_documents(docs_split)
print(f"Inserted {len(docs_split)} headlines.")

# Create index
astra_vector_index = VectorStoreIndexWrapper(vectorstore=astra_vector_store)
retriever = astra_vector_index.vectorstore.as_retriever()

# Google search with SerpAPI
def serpapi_search(question: str, api_key: str):
    params = {
        "q": question,
        "api_key": api_key,
        "engine": "google"
    }
    try:
        response = requests.get("https://serpapi.com/search", params=params)
        response.raise_for_status()
        data = response.json()
        organic_results = data.get("organic_results", [])
        summaries = [item["snippet"] for item in organic_results[:3] if "snippet" in item]
        if not summaries:
            top_stories = data.get("top_stories", [])
            summaries = [f"Top story: {item['link']}" for item in top_stories[:3] if "link" in item]
        if not summaries:
            related_searches = data.get("related_searches", [])
            summaries = [f"Related search: {item['link']}" for item in related_searches[:3] if "link" in item]
        return summaries
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err} - {response.text}")
    except Exception as err:
        print(f"An error occurred: {err}")
    return []

# Search agent using Google search
SERPAPI_KEY = os.getenv("SERP_API_KEY")
def google_search_agent(state):
    print("---Google Search---")
    question = state["question"]
    search_results = serpapi_search(question, SERPAPI_KEY)
    return {"documents": search_results, "question": question}

# Wikipedia search node
def wiki_search(state):
    print("---Wikipedia---")
    question = state["question"]
    docs = wiki.invoke({"query": question})
    wiki_results = docs.get("summary") if isinstance(docs, dict) else docs
    return {"documents": wiki_results or "No Wikipedia summary found.", "question": question}

# RAG vectorstore retrieval node (🔧 FIX: Convert Document objects to strings)
def retrieve(state):
    print("---Retrieve---")
    question = state["question"]
    documents = retriever.invoke(question)
    document_texts = [doc.page_content for doc in documents]
    state["documents"] = document_texts
    return {"documents": document_texts, "question": question}

# Routing node
def route_question(state):
    print("---Route Question---")
    question = state["question"]
    source = question_router.invoke({"question": question})
    if source.datasource == "wiki_search":
        print("---Route question to Wiki SEARCH---")
        return "wiki_search"
    elif source.datasource == "vectorstore":
        print("---Route question to RAG---")
        return "vectorstore"
    elif source.datasource == "google_search":
        print("--Route question to Google Search")
        return "google_search"

# Define graph state
def define_graph():
    class GraphState(TypedDict):
        question: str
        generation: str
        documents: List[str]
    workflow = StateGraph(GraphState)
    workflow.add_node("wiki_search", wiki_search)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("google_search", google_search_agent)
    workflow.add_conditional_edges(
        START,
        route_question,
        {
            "wiki_search": "wiki_search",
            "vectorstore": "retrieve",
            "google_search": "google_search",
        },
    )
    workflow.add_edge("retrieve", END)
    workflow.add_edge("wiki_search", END)
    workflow.add_edge("google_search", END)
    return workflow.compile()

# FastAPI app setup
app = FastAPI()

@app.get("/")
def get_started():
    return {
        "message": (
            "Welcome to the Multi-Agent RAG System!\n\n"
            "This application intelligently routes your questions to the most suitable data source:\n"
            "- If your query relates to topics like AI agents, prompt engineering, or adversarial attacks, it will search a specialized vector database.\n"
            "- If the topic is broader or factual, it may pull results from Wikipedia.\n"
            "- For trending or recent information, the system will use Google search via SerpAPI.\n\n"
            "Simply post your question to `/route_question/` and the system will do the rest!"
        )
    }

@app.post("/route_question/")
def route_question_endpoint(question: str):
    try:
        # Run the app
        app_output = define_graph().stream({"question": question})

        # Return the output of the first node in the graph as a response
        result = []
        for output in app_output:
            for key, value in output.items():
                result.append({"Node": key, "Documents": value.get("documents", "No documents found")})

        return JSONResponse(content=result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
