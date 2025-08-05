import streamlit as st
import os
import tempfile
import shutil
import fitz  # PyMuPDF
import textwrap
import requests
import json
from typing import List, Dict, Tuple, Optional
import time
import hashlib
from dotenv import load_dotenv
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from langdetect import detect, DetectorFactory
import re
import uuid

# Set seed for consistent language detection
DetectorFactory.seed = 0

# Load environment variables from .env file
load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
COLLECTION_NAME = "multilingual_pdf_rag"

# LlamaIndex imports
try:
    from llama_index import Document, VectorStoreIndex, ServiceContext, StorageContext
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
except ImportError:
    st.error("Please install required packages: pip install llama-index pymupdf streamlit qdrant-client python-dotenv")
    st.stop()

# Configuration
MAX_CHUNK_SIZE = 800  # Increased for better context
CHUNK_OVERLAP = 100   # Overlap between chunks
EMBEDDING_DIMENSION = 1024  # For multilingual-e5-large model

class LanguageHandler:
    """Enhanced multilingual handling with better language detection and processing."""
    
    LANGUAGE_NAMES = {
        'ar': 'Arabic',
        'en': 'English',
        'fr': 'French',
        'es': 'Spanish',
        'de': 'German',
        'it': 'Italian',
        'pt': 'Portuguese',
        'ru': 'Russian',
        'zh': 'Chinese',
        'ja': 'Japanese',
        'ko': 'Korean'
    }
    
    # Enhanced Arabic script detection
    ARABIC_PATTERN = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]')
    
    @classmethod
    def detect_language_robust(cls, text: str) -> str:
        """Enhanced language detection with multiple fallback strategies."""
        if not text or not text.strip():
            return 'en'
        
        # Clean text
        text = ' '.join(text.strip().split())
        
        # Arabic script detection (most reliable for Arabic)
        arabic_chars = len(cls.ARABIC_PATTERN.findall(text))
        total_chars = len([c for c in text if c.isalpha()])
        
        if total_chars > 0:
            arabic_ratio = arabic_chars / total_chars
            
            # Strong Arabic indicator
            if arabic_ratio > 0.3:
                return 'ar'
        
        # Common Arabic words check
        arabic_indicators = [
            'في', 'من', 'إلى', 'على', 'هذا', 'هذه', 'التي', 'الذي', 'كيف', 'ماذا', 'أين',
            'البيانات', 'الشخصية', 'المعالجة', 'الحماية', 'القانون', 'المبادئ', 'الحقوق'
        ]
        
        if any(word in text for word in arabic_indicators):
            return 'ar'
        
        # Use langdetect for other languages
        try:
            detected = detect(text)
            if detected in cls.LANGUAGE_NAMES:
                return detected
        except:
            pass
        
        return 'en'  # Default fallback
    
    @classmethod
    def get_simple_prompt(cls, query_lang: str) -> Dict[str, str]:
        """Generate simple, effective prompts based on query language."""
        
        if query_lang == 'ar':
            # Arabic - simple and direct
            system_prompt = """
أنت مساعد قانوني ذكي.
أجب فقط بالنقاط الموجودة في النصوص المسترجعة.
لا تضف أي معلومات غير موجودة في النص.
إذا لم تجد الإجابة في النصوص، قل: "لا توجد معلومات متاحة في النصوص."
"""

            user_template = """المعلومات المتوفرة:
{context}

السؤال: {query}

الإجابة:"""
                
        else:
            # English and other languages - simple and direct
            system_prompt = """You are a helpful assistant that answers questions based on the provided information.
Answer clearly and accurately in the same language as the question."""

            user_template = """Available information:
{context}

Question: {query}

Answer:"""
        
        return {
            "system_prompt": system_prompt,
            "user_template": user_template
        }

class DocumentProcessor:
    """Enhanced document processing with better language separation and metadata."""
    
    @staticmethod
    def extract_pdf_content(pdf_path: str, original_filename: str) -> List[Dict]:
        """Extract content with enhanced language detection and metadata."""
        doc = fitz.open(pdf_path)
        content_blocks = []
        
        for page_num, page in enumerate(doc, start=1):
            # Extract text with formatting information
            text_dict = page.get_text("dict")
            page_text = ""
            
            # Extract text blocks while preserving structure
            for block in text_dict.get("blocks", []):
                if "lines" in block:
                    block_text = ""
                    for line in block["lines"]:
                        line_text = " ".join(
                            span["text"] for span in line.get("spans", [])
                        ).strip()
                        if line_text:
                            block_text += line_text + " "
                    
                    if block_text.strip():
                        page_text += block_text.strip() + "\n\n"
            
            if page_text.strip():
                # Detect language for the entire page
                page_lang = LanguageHandler.detect_language_robust(page_text)
                
                content_blocks.append({
                    "file_name": original_filename,
                    "page_number": page_num,
                    "text": page_text.strip(),
                    "language": page_lang,
                    "char_count": len(page_text.strip())
                })
        
        doc.close()
        return content_blocks
    
    @staticmethod
    def create_smart_chunks(content_blocks: List[Dict]) -> List[Document]:
        """Create intelligent chunks with language awareness and proper metadata."""
        documents = []
        
        for block in content_blocks:
            text = block["text"]
            language = block["language"]
            
            # Split text into sentences for better chunking
            sentences = DocumentProcessor._split_into_sentences(text, language)
            
            # Create chunks with overlap
            chunks = DocumentProcessor._create_overlapping_chunks(
                sentences, MAX_CHUNK_SIZE, CHUNK_OVERLAP
            )
            
            for i, chunk in enumerate(chunks):
                if len(chunk.strip()) < 50:  # Skip very short chunks
                    continue
              
                # Create comprehensive metadata
                metadata = {
                    "file_name": block["file_name"],
                    "page_number": block["page_number"],
                    "language": language,
                    "chunk_id": f"{block['file_name']}_p{block['page_number']}_c{i}",
                    "chunk_index": i,
                    "total_chunks_in_page": len(chunks),
                    "char_count": len(chunk),
                    "content_hash": hashlib.md5(chunk.encode()).hexdigest()[:8]
                }
                
                documents.append(Document(text=chunk, extra_info=metadata))
        
        return documents
    
    @staticmethod
    def _split_into_sentences(text: str, language: str) -> List[str]:
        """Split text into sentences based on language."""
        if language == 'ar':
            # Arabic sentence endings
            delimiters = ['. ', '? ', '! ', '؟ ', '؛ ', '، ']
        else:
            # English and other languages
            delimiters = ['. ', '? ', '! ', '; ']
        
        sentences = [text]
        for delimiter in delimiters:
            new_sentences = []
            for sentence in sentences:
                new_sentences.extend(sentence.split(delimiter))
            sentences = new_sentences
        
        # Clean and filter sentences
        return [s.strip() for s in sentences if len(s.strip()) > 20]
    
    @staticmethod
    def _create_overlapping_chunks(sentences: List[str], max_size: int, overlap: int) -> List[str]:
        """Create overlapping chunks from sentences."""
        if not sentences:
            return []
        
        chunks = []
        current_chunk = ""
        sentence_buffer = []
        
        for sentence in sentences:
            # Add sentence to buffer
            sentence_buffer.append(sentence)
            test_chunk = " ".join(sentence_buffer)
            
            if len(test_chunk) > max_size:
                if current_chunk:
                    chunks.append(current_chunk)
                
                # Start new chunk with overlap
                overlap_sentences = sentence_buffer[-2:] if len(sentence_buffer) > 1 else sentence_buffer
                current_chunk = " ".join(overlap_sentences)
                sentence_buffer = overlap_sentences
            else:
                current_chunk = test_chunk
        
        # Add final chunk
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks

class EnhancedQdrantManager:
    """Enhanced Qdrant management with better error handling and operations."""
    
    def __init__(self, url: str, api_key: str, collection_name: str):
        self.url = url
        self.api_key = api_key
        self.collection_name = collection_name
        self.client = None
    
    def connect(self) -> bool:
        """Establish connection with comprehensive error handling."""
        try:
            if not self.url or not self.api_key:
                st.error("❌ Qdrant URL and API key must be provided")
                return False
            
            self.client = QdrantClient(url=self.url, api_key=self.api_key)
            
            # Test connection
            collections = self.client.get_collections()
            st.success(f"✅ Connected to Qdrant! Found {len(collections.collections)} collections")
            return True
            
        except Exception as e:
            st.error(f"❌ Failed to connect to Qdrant: {str(e)}")
            return False
    
    def setup_collection(self) -> bool:
        """Setup collection with proper configuration."""
        try:
            # Delete existing collection if it exists
            if self.collection_exists():
                st.info(f"🔄 Recreating collection '{self.collection_name}' for fresh start")
                self.client.delete_collection(self.collection_name)
            
            # Create new collection
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=EMBEDDING_DIMENSION,
                    distance=Distance.COSINE
                )
            )
            
            st.success(f"✅ Collection '{self.collection_name}' created successfully")
            return True
            
        except Exception as e:
            st.error(f"❌ Failed to setup collection: {str(e)}")
            return False
    
    def collection_exists(self) -> bool:
        """Check if collection exists."""
        try:
            collections = self.client.get_collections()
            return any(col.name == self.collection_name for col in collections.collections)
        except:
            return False
    
    def get_collection_stats(self) -> Dict:
        """Get detailed collection statistics."""
        try:
            if not self.collection_exists():
                return {"exists": False}
            
            info = self.client.get_collection(self.collection_name)
            
            # Get language distribution
            points = self.client.scroll(
                collection_name=self.collection_name,
                limit=1000,
                with_payload=True
            )[0]
            
            language_stats = {}
            for point in points:
                lang = point.payload.get('language', 'unknown')
                language_stats[lang] = language_stats.get(lang, 0) + 1
            
            return {
                "exists": True,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "status": info.status,
                "language_distribution": language_stats
            }
            
        except Exception as e:
            return {"exists": False, "error": str(e)}

class EnhancedRAGSystem:
    """Enhanced RAG system with superior multilingual handling."""
    
    def __init__(self, groq_api_key: str):
        self.groq_api_key = groq_api_key
        self.qdrant_manager = EnhancedQdrantManager(QDRANT_URL, QDRANT_API_KEY, COLLECTION_NAME)
        self.index = None
        self.retriever = None
        
        # Available models
        self.available_models = {
            "llama3-8b": "llama3-8b-8192",
            "llama3-70b": "llama3-70b-8192", 
            "mixtral": "mixtral-8x7b-32768",
            "gemma": "gemma-7b-it"
        }
        self.current_model = "llama3-70b-8192"
    
    def initialize_system(self) -> bool:
        """Initialize the complete RAG system."""
        return self.qdrant_manager.connect()
    
    def build_index(self, documents: List[Document]) -> bool:
        """Build index with enhanced error handling."""
        try:
            # Setup Qdrant collection
            if not self.qdrant_manager.setup_collection():
                return False
            
            # Initialize embeddings
            embed_model = HuggingFaceEmbedding(model_name="intfloat/multilingual-e5-large")
            
            service_context = ServiceContext.from_defaults(
                embed_model=embed_model, 
                llm=None,
                chunk_size=MAX_CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP
            )
            
            # Create vector store
            vector_store = QdrantVectorStore(
                client=self.qdrant_manager.client,
                collection_name=COLLECTION_NAME
            )
            
            # Build index
            self.index = VectorStoreIndex.from_documents(
                documents,
                service_context=service_context,
                vector_store=vector_store,
                show_progress=True
            )
            
            self.retriever = self.index.as_retriever(similarity_top_k=5)
            
            st.success(f"✅ Index built successfully with {len(documents)} documents")
            return True
            
        except Exception as e:
            st.error(f"❌ Error building index: {str(e)}")
            return False
    
    def load_existing_index(self) -> bool:
        """Load existing index from Qdrant."""
        try:
            if not self.qdrant_manager.connect():
                return False
            
            stats = self.qdrant_manager.get_collection_stats()
            if not stats.get("exists") or stats.get("points_count", 0) == 0:
                st.warning("No existing data found in Qdrant")
                return False
            
            # Initialize embeddings
            embed_model = HuggingFaceEmbedding(
                model_name="intfloat/multilingual-e5-large",
                cache_folder="./embeddings_cache"
            )
            
            service_context = ServiceContext.from_defaults(
                embed_model=embed_model,
                llm=None
            )
            
            # Create vector store
            vector_store = QdrantVectorStore(
                client=self.qdrant_manager.client,
                collection_name=COLLECTION_NAME
            )
            
            # Load index
            self.index = VectorStoreIndex.from_vector_store(
                vector_store,
                service_context=service_context
            )
            
            self.retriever = self.index.as_retriever(similarity_top_k=5)
            
            st.success(f"✅ Loaded existing index with {stats['points_count']} documents")
            return True
            
        except Exception as e:
            st.error(f"❌ Error loading index: {str(e)}")
            return False
    
    def query_with_language_awareness(self, question: str, top_k: int = 5) -> Dict:
        """Simplified query with natural language handling."""
        try:
            if not self.retriever:
                return {"error": "No index available"}
            
            # Detect query language
            query_lang = LanguageHandler.detect_language_robust(question)
            
            # Retrieve documents
            self.retriever = self.index.as_retriever(similarity_top_k=top_k)
            nodes = self.retriever.retrieve(question)
            
            if not nodes:
                return {
                    "answer": "No relevant documents found." if query_lang == 'en' else "لم يتم العثور على وثائق ذات صلة.",
                    "query_language": query_lang,
                    "sources": []
                }
            
            # Process retrieved documents
            contexts = []
            sources = []
            
            for i, node in enumerate(nodes):
                text = node.node.text
                metadata = node.node.extra_info or {}
                score = getattr(node, 'score', 0)
                
                contexts.append(text)
                sources.append({
                    "id": i + 1,
                    "text": text if len(text) <= 500 else text[:500] + "...",
                    "score": round(score, 3),
                    "metadata": metadata,
                    "language": metadata.get('language', 'unknown')
                })
            
            # Generate answer with simplified approach
            combined_context = "\n\n".join(contexts)
            answer = self._generate_simple_answer(question, combined_context, query_lang)
            
            return {
                "answer": answer,
                "query_language": query_lang,
                "sources": sources,
                "model": self.current_model
            }
            
        except Exception as e:
            return {
                "error": f"Query error: {str(e)}",
                "query_language": "unknown",
                "sources": []
            }
    
    def _generate_simple_answer(self, query: str, context: str, query_lang: str) -> str:
        """Generate answer using simple, effective prompts."""
        try:
            # Get simple prompts
            prompts = LanguageHandler.get_simple_prompt(query_lang)
            
            # Format the user message
            user_message = prompts["user_template"].format(
                context=context,
                query=query
            )
            
            # Prepare messages
            messages = [
                {"role": "system", "content": prompts["system_prompt"]},
                {"role": "user", "content": user_message}
            ]
            
            # Call Groq API
            headers = {
                "Authorization": f"Bearer {self.groq_api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": self.current_model,
                "messages": messages,
                "temperature": 0.1,
                "max_tokens": 2000
            }
            
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=60
            )
            
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
            
        except Exception as e:
            return f"Error generating answer: {str(e)}"
    
    def get_system_stats(self) -> Dict:
        """Get comprehensive system statistics."""
        stats = self.qdrant_manager.get_collection_stats()
        stats["current_model"] = self.current_model
        stats["available_models"] = list(self.available_models.keys())
        return stats

def main():
    st.set_page_config(
        page_title="Enhanced Multilingual PDF RAG",
        page_icon="🌍",
        layout="wide"
    )
    
    st.title("🌍 Enhanced Multilingual PDF RAG System")
    st.markdown("Advanced multilingual document processing with language-aware retrieval")
    
    # Environment check
    if not all([QDRANT_URL, QDRANT_API_KEY, GROQ_API_KEY]):
        st.error("❌ Please set QDRANT_URL, QDRANT_API_KEY, and GROQ_API_KEY in your .env file")
        st.stop()
    
    # Initialize session state
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = EnhancedRAGSystem(GROQ_API_KEY)
    if 'index_ready' not in st.session_state:
        st.session_state.index_ready = False
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ System Control")
        
        # Initialize system
        if st.button("🚀 Initialize System"):
            with st.spinner("Initializing..."):
                if st.session_state.rag_system.initialize_system():
                    st.success("✅ System initialized!")
        
        # Load existing index
        if st.button("📂 Load Existing Index"):
            with st.spinner("Loading..."):
                if st.session_state.rag_system.load_existing_index():
                    st.session_state.index_ready = True
                    st.success("✅ Index loaded!")
        
        # System stats
        if st.button("📊 System Stats"):
            stats = st.session_state.rag_system.get_system_stats()
            st.json(stats)
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["📤 Upload & Process", "💬 Query", "📊 Analytics"])
    
    with tab1:
        st.header("📤 Document Upload & Processing")
        
        uploaded_files = st.file_uploader(
            "Upload PDF files",
            type="pdf",
            accept_multiple_files=True,
            help="Upload multilingual PDFs (Arabic/English supported)"
        )
        
        if uploaded_files and st.button("🔄 Process Documents", type="primary"):
            all_documents = []
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"Processing {uploaded_file.name}...")
                
                # Save temporarily with original filename
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_path = tmp_file.name
                
                try:
                    # Extract content with original filename
                    content_blocks = DocumentProcessor.extract_pdf_content(
                        tmp_path, uploaded_file.name
                    )
                    
                    # Create smart chunks
                    documents = DocumentProcessor.create_smart_chunks(content_blocks)
                    all_documents.extend(documents)
                    
                    # Show processing stats
                    lang_stats = {}
                    for block in content_blocks:
                        lang = block['language']
                        lang_stats[lang] = lang_stats.get(lang, 0) + 1
                    
                    st.info(f"📄 {uploaded_file.name}: {len(content_blocks)} pages, {len(documents)} chunks")
                    st.write("Languages detected:", lang_stats)
                    
                finally:
                    os.unlink(tmp_path)
                
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            # Build index
            if all_documents:
                status_text.text("Building multilingual vector index...")
                
                with st.spinner("Building index in Qdrant..."):
                    if st.session_state.rag_system.build_index(all_documents):
                        st.session_state.index_ready = True
                        st.success(f"✅ Successfully processed {len(all_documents)} document chunks!")
                    else:
                        st.error("❌ Failed to build index")
    
    with tab2:
        st.header("💬 Intelligent Query System")
        
        if not st.session_state.index_ready:
            st.warning("⚠️ Please upload documents or load existing index first")
            return
        
        # Query interface
        col1, col2 = st.columns([3, 1])
        
        with col1:
            question = st.text_area(
                "Ask your question:",
                placeholder="What are the main principles of personal data processing?\nما هي المبادئ الأساسية لمعالجة البيانات الشخصية؟",
                height=100
            )
        
        with col2:
            top_k = st.slider("Sources to retrieve", 3, 10, 5)
            
        if question and st.button("🔍 Search", type="primary"):
            with st.spinner("Processing your question..."):
                result = st.session_state.rag_system.query_with_language_awareness(
                    question, top_k
                )
                
                if "error" in result:
                    st.error(result["error"])
                else:
                    # Display results
                    col1, col2 = st.columns(2)
                    with col1:
                        st.info(f"🔍 Query Language: {result['query_language']}")
                    with col2:
                        st.info(f"🤖 Model: {result['model']}")
                    
                    # Answer
                    st.subheader("🎯 Answer")
                    st.write(result['answer'])
                    
                    # Sources
                    if result.get('sources'):
                        st.subheader(f"📚 Sources ({len(result['sources'])})")
                        
                        for source in result['sources']:
                            metadata = source['metadata']
                            title = f"📄 {metadata.get('file_name', 'Unknown')} - Page {metadata.get('page_number', '?')}"
                            title += f" ({source['language']}) - Score: {source['score']}"
                            
                            with st.expander(title):
                                st.write("**Chunk ID:**", metadata.get('chunk_id', 'Unknown'))
                                st.write("**Language:**", source['language'])
                                st.write("**Content:**", source['text'])
    
    with tab3:
        st.header("📊 System Analytics")
        
        if st.button("🔄 Refresh Stats"):
            stats = st.session_state.rag_system.get_system_stats()
            
            if stats.get("exists"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Total Documents", stats.get('points_count', 0))
                    st.metric("Vector Count", stats.get('vectors_count', 0))
                    st.metric("Current Model", stats.get('current_model', 'Unknown'))
                
                with col2:
                    st.subheader("Language Distribution")
                    lang_dist = stats.get('language_distribution', {})
                    for lang, count in lang_dist.items():
                        lang_name = LanguageHandler.LANGUAGE_NAMES.get(lang, lang)
                        st.write(f"**{lang_name}:** {count} documents")
                
                # Language distribution chart
                if lang_dist:
                    st.subheader("📈 Language Distribution Chart")
                    chart_data = {
                        "Language": [LanguageHandler.LANGUAGE_NAMES.get(k, k) for k in lang_dist.keys()],
                        "Count": list(lang_dist.values())
                    }
                    st.bar_chart(chart_data)
            else:
                st.warning("No data found in the system")

if __name__ == "__main__":
    main()