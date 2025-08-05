from llama_index import VectorStoreIndex, ServiceContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores import ChromaVectorStore

# Load the existing Chroma vector store
vector_store = ChromaVectorStore(persist_dir="./vector_store")
embed_model = HuggingFaceEmbedding(model_name="intfloat/multilingual-e5-large")
service_context = ServiceContext.from_defaults(embed_model=embed_model, llm=None)

# Load your existing index
index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store,
    service_context=service_context
)

# Create a query engine
query_engine = index.as_query_engine(similarity_top_k=3)

# Ask a question
response = query_engine.query("What does the Saudi Arabia Transparency Report say about data protection?")
print(response)
