import cassio
from dotenv import load_dotenv
import os
import openai
from langchain_community.document_loaders import WebBaseLoader  # Updated import
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Cassandra  # Updated import
from langchain_openai import OpenAIEmbeddings  # Updated import
from langchain.indexes.vectorstore import VectorStoreIndexWrapper

load_dotenv()
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_ID = os.getenv("ASTRA_DB_ID")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Ensure this is set in your .env file

try:
    cassio.init(
        token=ASTRA_DB_APPLICATION_TOKEN,
        database_id=ASTRA_DB_ID
    )
    print("✅ Successfully connected to Astra DB using CassIO.")
except Exception as e:
    print("❌ Failed to connect:", str(e))

# Set your OpenAI API key
openai.api_key = OPENAI_API_KEY

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

# Load documents from the URLs
docs = [WebBaseLoader(url).load() for url in urls]
doc_list = [item for sublist in docs for item in sublist]

# Split documents into smaller chunks
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=500, chunk_overlap=0
)
docs_split = text_splitter.split_documents(doc_list)

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings()

# Initialize Cassandra vector store
astra_vector_store = Cassandra(
    embedding=embeddings,
    session=None,
    keyspace=None,
    table_name="multi_agent_demo"
)

# Add documents to the vector store
astra_vector_store.add_documents(docs_split)
print(f"Inserted {len(docs_split)} headlines.")

# Create the index
astra_vector_index = VectorStoreIndexWrapper(vectorstore=astra_vector_store)

