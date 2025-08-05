import streamlit as st
import os
import tempfile
import shutil
import fitz  # PyMuPDF
import textwrap
import requests
import json
from typing import List, Dict, Tuple
import time

# LlamaIndex imports
try:
    from llama_index import Document, VectorStoreIndex, ServiceContext, StorageContext, load_index_from_storage
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
except ImportError:
    st.error("Please install required packages: pip install llama-index pymupdf streamlit")
    st.stop()

# Configuration
VECTOR_STORE_DIR = "./vector_store"
MAX_CHUNK_SIZE = 500

class SimpleGroqChat:
    """Simple Groq API wrapper for chat completions."""
    
    def __init__(self, api_key: str, model: str = "llama3-8b-8192"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
    
    def chat(self, messages: list, temperature: float = 0.1, max_tokens: int = 1000) -> str:
        """Send chat completion request to Groq."""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            response = requests.post(self.base_url, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            return result["choices"][0]["message"]["content"]
            
        except requests.exceptions.RequestException as e:
            return f"Network error: {e}"
        except KeyError as e:
            return f"API response error: {e}"
        except Exception as e:
            return f"Unexpected error: {e}"

class PDFProcessor:
    """PDF processing and text extraction."""
    
    @staticmethod
    def extract_pdf_text_with_headings(pdf_path: str, file_name: str) -> List[Dict]:
        """Extract text with heading detection from PDF."""
        doc = fitz.open(pdf_path)
    
        pages_data = []

        for page_num, page in enumerate(doc, start=1):
            page_dict = page.get_text("dict")
            current_heading = None
            page_chunks = []

            for block in page_dict["blocks"]:
                for line in block.get("lines", []):
                    text_line = " ".join(span["text"] for span in line["spans"]).strip()
                    if not text_line:
                        continue

                    # Detect heading: short line + larger font
                    avg_font_size = sum(span["size"] for span in line["spans"]) / len(line["spans"])
                    if len(text_line.split()) <= 7 and avg_font_size > 12:
                        current_heading = text_line
                        continue

                    # Append line with current heading
                    page_chunks.append({
                        "file_name": file_name, 
                        "page_number": page_num,
                        "heading": current_heading,
                        "text": text_line
                    })

            if page_chunks:
                pages_data.extend(page_chunks)

        doc.close()
        return pages_data
    
    @staticmethod
    def create_documents(pages_data: List[Dict]) -> List[Document]:
        """Create LlamaIndex Documents with metadata including headings."""
        documents = []
        buffer = ""
        current_metadata = None

        for entry in pages_data:
            buffer += " " + entry["text"]
            current_metadata = {
                "file_name": entry["file_name"],
                "page_number": entry["page_number"],
                "heading": entry["heading"]
            }

            if len(buffer) >= MAX_CHUNK_SIZE:
                documents.append(Document(text=buffer.strip(), extra_info=current_metadata))
                buffer = ""

        # Add any remaining buffer
        if buffer.strip() and current_metadata:
            documents.append(Document(text=buffer.strip(), extra_info=current_metadata))

        return documents

class RAGSystem:
    """Complete RAG system with Groq integration."""
    
    def __init__(self, groq_api_key: str):
        self.groq_chat = SimpleGroqChat(groq_api_key)
        self.index = None
        self.retriever = None
        
        # Available models
        self.available_models = {
            "llama3-8b": "llama3-8b-8192",
            "llama3-70b": "llama3-70b-8192", 
            "llama2-70b": "llama2-70b-4096",
            "mixtral": "mixtral-8x7b-32768",
            "gemma": "gemma-7b-it"
        }
    
    def build_index(self, documents: List[Document]) -> bool:
        """Build and persist vector index."""
        try:
            embed_model = HuggingFaceEmbedding(model_name="intfloat/multilingual-e5-large")
            service_context = ServiceContext.from_defaults(embed_model=embed_model, llm=None)
            
            self.index = VectorStoreIndex.from_documents(
                documents, 
                service_context=service_context
            )
            
            # Persist the index
            self.index.storage_context.persist(persist_dir=VECTOR_STORE_DIR)
            self.retriever = self.index.as_retriever(similarity_top_k=5)
            
            return True
            
        except Exception as e:
            st.error(f"Error building index: {e}")
            return False
    
    def load_existing_index(self) -> bool:
        """Load existing vector index."""
        try:
            if not os.path.exists(VECTOR_STORE_DIR):
                return False
                
            embed_model = HuggingFaceEmbedding(model_name="intfloat/multilingual-e5-large")
            storage_context = StorageContext.from_defaults(persist_dir=VECTOR_STORE_DIR)
            service_context = ServiceContext.from_defaults(embed_model=embed_model, llm=None)
            
            self.index = load_index_from_storage(storage_context, service_context=service_context)
            self.retriever = self.index.as_retriever(similarity_top_k=5)
            
            return True
            
        except Exception as e:
            st.error(f"Error loading index: {e}")
            return False
    
    def retrieve_context(self, query: str, top_k: int = 5) -> Tuple[str, List[Dict]]:
        """Retrieve relevant context from the index."""
        try:
            if not self.retriever:
                return "No index available", []
                
            self.retriever = self.index.as_retriever(similarity_top_k=top_k)
            nodes = self.retriever.retrieve(query)
            
            contexts = []
            sources = []
            
            for i, node in enumerate(nodes):
                context_text = node.node.text
                metadata = node.node.extra_info if hasattr(node.node, 'extra_info') else {}
                score = getattr(node, 'score', 0)
                
                contexts.append(context_text)
                sources.append({
                    "id": i + 1,
                    "text": context_text[:300] + "..." if len(context_text) > 300 else context_text,
                    "score": round(score, 3),
                    "metadata": metadata
                })
            
            return "\n\n".join(contexts), sources
            
        except Exception as e:
            return f"Error retrieving context: {e}", []
    
    def generate_answer(self, query: str, context: str) -> str:
        """Generate answer using Groq with retrieved context."""
        system_prompt = """You are a helpful assistant that answers questions based on the provided context. 
Use the context to provide accurate, detailed answers. If the context doesn't contain enough information 
to answer the question, say so clearly. Always cite specific parts of the context when possible."""
        
        user_prompt = f"""Context information:
{context}

Question: {query}

Please provide a comprehensive answer based on the context above."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        return self.groq_chat.chat(messages)
    
    def query(self, question: str, top_k: int = 5) -> Dict:
        """Complete RAG query with context retrieval and answer generation."""
        try:
            context, sources = self.retrieve_context(question, top_k)
            
            if not context or "Error" in context:
                return {
                    "answer": "Could not retrieve relevant context from documents.",
                    "sources": [],
                    "model": self.groq_chat.model
                }
            
            answer = self.generate_answer(question, context)
            
            return {
                "answer": answer,
                "sources": sources,
                "model": self.groq_chat.model,
                "context_used": len(context.split())
            }
            
        except Exception as e:
            return {
                "answer": f"Error during query: {e}",
                "sources": [],
                "model": self.groq_chat.model
            }
    
    def switch_model(self, model_key: str):
        """Switch to a different Groq model."""
        if model_key in self.available_models:
            self.groq_chat.model = self.available_models[model_key]
            return True
        return False
    
    def test_api(self) -> bool:
        """Test if Groq API is working."""
        try:
            test_response = self.groq_chat.chat([
                {"role": "user", "content": "Hello! Please respond with 'API test successful'."}
            ])
            return "API test successful" in test_response
        except:
            return False

def main():
    st.set_page_config(
        page_title="PDF RAG System with Groq",
        page_icon="🦙",
        layout="wide"
    )
    
    st.title("🦙 PDF RAG System with Groq")
    st.markdown("Upload PDFs, build a knowledge base, and ask questions!")
    
    # Initialize session state
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = None
    if 'index_built' not in st.session_state:
        st.session_state.index_built = False
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key input
        api_key = st.text_input(
            "🔑 Groq API Key", 
            type="password",
            help="Get your API key from https://console.groq.com"
        )
        
        if api_key:
            if st.session_state.rag_system is None:
                st.session_state.rag_system = RAGSystem(api_key)
            
            # Test API
            if st.button("🧪 Test API"):
                with st.spinner("Testing API..."):
                    if st.session_state.rag_system.test_api():
                        st.success("✅ API working!")
                    else:
                        st.error("❌ API test failed")
            
            # Model selection
            st.subheader("🤖 Model Selection")
            model_options = list(st.session_state.rag_system.available_models.keys())
            selected_model = st.selectbox("Choose model:", model_options)
            
            if st.button("Switch Model"):
                if st.session_state.rag_system.switch_model(selected_model):
                    st.success(f"✅ Switched to {selected_model}")
                else:
                    st.error("❌ Failed to switch model")
            
            st.markdown("---")
            
            # Load existing index
            if st.button("📂 Load Existing Index"):
                with st.spinner("Loading index..."):
                    if st.session_state.rag_system.load_existing_index():
                        st.session_state.index_built = True
                        st.success("✅ Index loaded!")
                    else:
                        st.warning("No existing index found")
            
            # Clear index
            if st.button("🗑️ Clear Index"):
                if os.path.exists(VECTOR_STORE_DIR):
                    shutil.rmtree(VECTOR_STORE_DIR)
                    st.session_state.index_built = False
                    st.session_state.rag_system.index = None
                    st.success("✅ Index cleared!")
    
    # Main content area
    if not api_key:
        st.warning("⚠️ Please enter your Groq API key in the sidebar to continue.")
        return
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📤 Upload & Process", "💬 Ask Questions", "📊 System Info"])
    
    with tab1:
        st.header("📤 Upload PDF Documents")
        
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type="pdf",
            accept_multiple_files=True,
            help="Upload one or more PDF files to build your knowledge base"
        )
        
        if uploaded_files:
            if st.button("🔄 Process Documents", type="primary"):
                all_pages = []
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Process each uploaded file
                for i, uploaded_file in enumerate(uploaded_files):
                    original_name = uploaded_file.name  # ✅ this is the real name

                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.read())
                        tmp_path = tmp_file.name

              
                    
                    try:
                        # Extract text from PDF
                          # Pass original_name to your PDFProcessor
                        pages_data = PDFProcessor.extract_pdf_text_with_headings(tmp_path, file_name=original_name)
                        all_pages.extend(pages_data)
                        
                        progress_bar.progress((i + 1) / len(uploaded_files))
                        
                    finally:
                        # Clean up temporary file
                        os.unlink(tmp_path)
                
                if all_pages:
                    status_text.text("Creating documents and building index...")
                    
                    # Create documents
                    documents = PDFProcessor.create_documents(all_pages)
                    
                    st.info(f"📄 Extracted {len(all_pages)} text segments from {len(uploaded_files)} PDFs")
                    st.info(f"📦 Created {len(documents)} document chunks")
                    
                    # Build index
                    with st.spinner("Building vector index..."):
                        if st.session_state.rag_system.build_index(documents):
                            st.session_state.index_built = True
                            st.success("✅ Index built successfully!")
                            status_text.text("Ready for questions!")
                        else:
                            st.error("❌ Failed to build index")
                else:
                    st.error("❌ No text extracted from PDFs")
    
    with tab2:
        st.header("💬 Ask Questions")
        
        if not st.session_state.index_built:
            st.warning("⚠️ Please upload and process documents first, or load an existing index.")
            return
        
        # Query input
        question = st.text_input(
            "🔍 Ask a question about your documents:",
            placeholder="What is the main topic discussed in the documents?"
        )
        
        # Query settings
        col1, col2 = st.columns([3, 1])
        with col2:
            top_k = st.slider("📊 Number of sources", 3, 10, 5)
        
        if question and st.button("🚀 Get Answer", type="primary"):
            with st.spinner("Searching and generating answer..."):
                result = st.session_state.rag_system.query(question, top_k)
                
                # Display answer
                st.subheader("🎯 Answer")
                st.write(result['answer'])
                
                # Display sources
                if result.get('sources'):
                    st.subheader(f"📚 Sources ({len(result['sources'])} found)")
                    
                    for source in result['sources']:
                        with st.expander(f"📄 {source['metadata'].get('file_name', 'Unknown')} - Page {source['metadata'].get('page_number', '?')} (Score: {source['score']})"):
                            if source['metadata'].get('heading'):
                                st.markdown(f"**Heading:** {source['metadata']['heading']}")
                            st.text(source['text'])
                
                # Display metadata
                st.markdown("---")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("🤖 Model Used", result['model'])
                with col2:
                    st.metric("📝 Context Words", result.get('context_used', 'Unknown'))
    
    with tab3:
        st.header("📊 System Information")
        
        # System status
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔧 System Status")
            st.write(f"**Index Built:** {'✅ Yes' if st.session_state.index_built else '❌ No'}")
            st.write(f"**Vector Store Directory:** `{VECTOR_STORE_DIR}`")
            st.write(f"**Max Chunk Size:** {MAX_CHUNK_SIZE} characters")
            
            if st.session_state.rag_system:
                st.write(f"**Current Model:** {st.session_state.rag_system.groq_chat.model}")
        
        with col2:
            st.subheader("🤖 Available Models")
            if st.session_state.rag_system:
                for key, model in st.session_state.rag_system.available_models.items():
                    current = " (current)" if model == st.session_state.rag_system.groq_chat.model else ""
                    st.write(f"**{key}:** {model}{current}")
        
        # Index information
        if os.path.exists(VECTOR_STORE_DIR):
            st.subheader("💾 Index Information")
            index_files = os.listdir(VECTOR_STORE_DIR)
            st.write(f"**Index files:** {len(index_files)}")
            
            # Calculate total size
            total_size = sum(
                os.path.getsize(os.path.join(VECTOR_STORE_DIR, f)) 
                for f in index_files
            )
            st.write(f"**Total size:** {total_size / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
    main()