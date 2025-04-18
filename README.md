Multi-Agent RAG with Astra DB & OpenAI Embeddings
This project integrates Astra DB with LangChain and OpenAI embeddings to implement a Retrieval-Augmented Generation (RAG) system.

Functionality
Document Loading: Fetches web documents from provided URLs.

Text Splitting: Breaks documents into smaller chunks for efficient processing.

Embeddings Generation: Uses OpenAI to generate embeddings for the document chunks.

Vector Store Integration: Stores the generated embeddings in Astra DB for efficient retrieval.

Search Functionality: Enables searching of stored embeddings to retrieve relevant information.

Key Features
Efficient web scraping and document loading.

Seamless integration with Astra DB to store and retrieve embeddings.

Leverages OpenAI's embeddings for advanced search capabilities.

Supports RAG-style question answering by retrieving relevant chunks of text from stored embeddings.

Setup & Usage
Set up the environment: Ensure the necessary API keys and credentials are configured.

Run the script: Execute the script to load documents, generate embeddings, and store them in Astra DB.

Search capability: Query the vector store to retrieve relevant information based on the embeddings.