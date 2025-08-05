import json
import uuid
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import hashlib
import re
import random
import time
import os

# LlamaIndex imports
from llama_index.core import (
    VectorStoreIndex, 
    Document, 
    Settings,
    StorageContext,
    load_index_from_storage
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.extractors import TitleExtractor
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.prompts import PromptTemplate

# Vector database
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

# For PDF extraction
from llama_cloud_services import LlamaExtract
from llama_cloud.types import ExtractConfig, ExtractMode
from dotenv import load_dotenv
import requests

load_dotenv()

# Streamlit secrets
import streamlit as st
GROQ_API_KEY_1 = st.secrets["groq_api_key_1"]
GROQ_API_KEY = st.secrets["groq_api_key"]
QDRANT_API_KEY = st.secrets["qdrant_api_key"]
QDRANT_URL = st.secrets["qdrant_url"]
LLAMA_CLOUD_API_KEY = st.secrets["llama_cloud_api_key"]
OPENAI_API_KEY = st.secrets.get("openai_api_key", "")

# Collection names for different languages
ARABIC_COLLECTION = "arabic_docs_llamaindex"
ENGLISH_COLLECTION = "english_docs_llamaindex"

@dataclass
class RAGResponse:
    """Response from RAG system"""
    answer: str
    retrieved_chunks: List[Dict]
    query: str
    query_language: str
    confidence_score: float = 0.0
    sources: List[str] = None

class LanguageDetector:
    """Simple but effective language detector for Arabic and English"""
    
    def __init__(self):
        # Arabic Unicode ranges
        self.arabic_pattern = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]')
        # English pattern (basic Latin characters)
        self.english_pattern = re.compile(r'[a-zA-Z]')
    
    def detect_language(self, text: str) -> str:
        """
        Detect if text is primarily Arabic or English
        Returns: 'ar' for Arabic, 'en' for English
        """
        if not text or not text.strip():
            return 'en'  # Default to English for empty text
        
        clean_text = re.sub(r'[0-9\s\.,!?;:\-\(\)\[\]\"\']+', '', text)
        
        if not clean_text:
            return 'en'
        
        arabic_chars = len(self.arabic_pattern.findall(clean_text))
        english_chars = len(self.english_pattern.findall(clean_text))
        
        total_chars = arabic_chars + english_chars
        
        if total_chars == 0:
            return 'en'
        
        arabic_ratio = arabic_chars / total_chars
        
        # If more than 30% Arabic characters, consider it Arabic
        return 'ar' if arabic_ratio > 0.3 else 'en'

class MultilingualRAGWithLlamaIndex:
    """Enhanced RAG system using LlamaIndex with multilingual support"""
    
    def __init__(self, 
                 llama_api_key: str,
                 qdrant_url: str,
                 qdrant_api_key: str,
                 groq_api_key: str,
                 embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                 groq_model: str = "llama3-70b-8192",
                 chunk_size: int = 512,
                 chunk_overlap: int = 50):
        """Initialize the multilingual RAG system with LlamaIndex"""
        
        self.extractor = LlamaExtract(api_key=llama_api_key)
        self.language_detector = LanguageDetector()
        
        # Initialize Qdrant client
        self.qdrant_client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
        )
        
        # Configure LlamaIndex Settings with HuggingFace embedding
        Settings.embed_model = HuggingFaceEmbedding(
            model_name=embedding_model,
            trust_remote_code=True
        )
        
        # Configure LLM - use different API keys based on language
        self.groq_api_key = groq_api_key
        self.groq_api_key_1 = GROQ_API_KEY_1
        
        Settings.llm = Groq(
            model=groq_model,
            api_key=groq_api_key
        )
        
        # Node parser for chunking
        self.node_parser = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separator=" ",
        )
        
        # Storage for indices and vector stores
        self.indices = {}
        self.vector_stores = {}
        
        print(f"Initialized LlamaIndex RAG system")
        print(f"Embedding model: {embedding_model}")
        print(f"LLM model: {groq_model}")
        print(f"Chunk size: {chunk_size}, overlap: {chunk_overlap}")
        
        # Language-specific schemas
        self.arabic_schema = {
            "additionalProperties": False,
            "properties": {
                "page_content": {
                    "description": "List of content blocks for a page in Arabic.",
                    "items": {
                        "additionalProperties": False,
                        "properties": {
                            "block_type": {
                                "enum": ["title", "paragraph", "list", "table", "columnar_text", "footnote"],
                                "type": "string",
                                "description": "Type of content block."
                            },
                            "content": {
                                "description": "Arabic text content for title, paragraph, or footnote blocks.",
                                "type": "string"
                            },
                            "items": {
                                "description": "List items for list-type blocks in Arabic.",
                                "items": {"type": "string"},
                                "type": "array"
                            },
                            "table_title": {
                                "description": "Optional title above the table in Arabic.",
                                "type": "string"
                            },
                            "headers": {
                                "description": "Column headers for table blocks in Arabic.",
                                "items": {"type": "string"},
                                "type": "array"
                            },
                            "rows": {
                                "description": "Each table row as an array of cell texts in Arabic.",
                                "items": {"items": {"type": "string"}, "type": "array"},
                                "type": "array"
                            },
                            "columns": {
                                "description": "Used for multi-column text (Arabic RTL).",
                                "items": {
                                    "additionalProperties": False,
                                    "properties": {
                                        "column_id": {"type": "integer"},
                                        "content": {"type": "string"}
                                    },
                                    "required": ["column_id", "content"],
                                    "type": "object"
                                },
                                "type": "array"
                            }
                        },
                        "required": ["block_type", "content", "items", "table_title", "headers", "rows", "columns"],
                        "type": "object"
                    },
                    "type": "array"
                }
            },
            "required": ["page_content"],
            "type": "object"
        }
        
        self.english_schema = {
            "additionalProperties": False,
            "properties": {
                "page_content": {
                    "description": "List of content blocks for a page in English.",
                    "items": {
                        "additionalProperties": False,
                        "properties": {
                            "block_type": {
                                "enum": ["title", "paragraph", "list", "table", "columnar_text", "footnote"],
                                "type": "string",
                                "description": "Type of content block."
                            },
                            "content": {
                                "description": "English text content for title, paragraph, or footnote blocks.",
                                "type": "string"
                            },
                            "items": {
                                "description": "List items for list-type blocks in English.",
                                "items": {"type": "string"},
                                "type": "array"
                            },
                            "table_title": {
                                "description": "Optional title above the table in English.",
                                "type": "string"
                            },
                            "headers": {
                                "description": "Column headers for table blocks in English.",
                                "items": {"type": "string"},
                                "type": "array"
                            },
                            "rows": {
                                "description": "Each table row as an array of cell texts in English.",
                                "items": {"items": {"type": "string"}, "type": "array"},
                                "type": "array"
                            },
                            "columns": {
                                "description": "Used for multi-column text.",
                                "items": {
                                    "additionalProperties": False,
                                    "properties": {
                                        "column_id": {"type": "integer"},
                                        "content": {"type": "string"}
                                    },
                                    "required": ["column_id", "content"],
                                    "type": "object"
                                },
                                "type": "array"
                            }
                        },
                        "required": ["block_type", "content", "items", "table_title", "headers", "rows", "columns"],
                        "type": "object"
                    },
                    "type": "array"
                }
            },
            "required": ["page_content"],
            "type": "object"
        }

    def create_rag_prompt(self, query: str, context: str, language: str) -> str:
        """Create RAG prompt for answer generation based on language"""
        
        if language == 'ar':
            prompt = f"""أنت مساعد ذكي متخصص في الإجابة على الأسئلة باللغة العربية بناءً على المعلومات المقدمة.

السياق المتاح:
{context}

السؤال: {query}

التعليمات:
1. اقرأ السياق المقدم بعناية
2. أجب على السؤال بناءً على المعلومات الموجودة في السياق فقط
3. إذا لم تجد معلومات كافية في السياق، قل ذلك بصراحة
4. اذكر رقم المصدر والصفحة عند الإمكان
5. اكتب إجابة واضحة ومفيدة باللغة العربية
6. لا تختلق معلومات غير موجودة في السياق
7. إذا كان المحتوى مدمجاً من عدة كتل أو مقسماً، خذ ذلك في الاعتبار

الإجابة:"""
        else:
            prompt = f"""You are an intelligent assistant specialized in answering questions in English based on the provided information.

Available Context:
{context}

Question: {query}

Instructions:
1. Read the provided context carefully
2. Answer the question based only on the information found in the context
3. If you don't find sufficient information in the context, say so clearly
4. Mention source number and page when possible
5. Write a clear and helpful answer in English
6. Do not make up information not found in the context
7. If content is merged from multiple blocks or split, take that into account

Answer:"""
        
        return prompt

    def get_language_specific_prompt_template(self, language: str) -> PromptTemplate:
        """Get LlamaIndex PromptTemplate for specific language"""
        
        if language == 'ar':
            template_str = """أنت مساعد ذكي متخصص في الإجابة على الأسئلة باللغة العربية بناءً على المعلومات المقدمة.

السياق المتاح:
{context_str}

السؤال: {query_str}

التعليمات:
1. اقرأ السياق المقدم بعناية
2. أجب على السؤال بناءً على المعلومات الموجودة في السياق فقط
3. إذا لم تجد معلومات كافية في السياق، قل ذلك بصراحة
4. اذكر رقم المصدر والصفحة عند الإمكان
5. اكتب إجابة واضحة ومفيدة باللغة العربية
6. لا تختلق معلومات غير موجودة في السياق
7. إذا كان المحتوى مدمجاً من عدة كتل أو مقسماً، خذ ذلك في الاعتبار

الإجابة:"""
        else:
            template_str = """You are an intelligent assistant specialized in answering questions in English based on the provided information.

Available Context:
{context_str}

Question: {query_str}

Instructions:
1. Read the provided context carefully
2. Answer the question based only on the information found in the context
3. If you don't find sufficient information in the context, say so clearly
4. Mention source number and page when possible
5. Write a clear and helpful answer in English
6. Do not make up information not found in the context
7. If content is merged from multiple blocks or split, take that into account

Answer:"""
        
        return PromptTemplate(template=template_str)
    
    def extract_pdf(self, pdf_path: str, language: str = 'auto') -> Tuple[List[Dict], str]:
        """Extract structured content from PDF with language detection"""
        import time
        import random
        import uuid
    
        timestamp = str(int(time.time()))        
        random_suffix = str(random.randint(1000, 9999)) 
        unique_id = str(uuid.uuid4())[:8]       
        agent_name = f"agent_{timestamp}_{random_suffix}_{unique_id}"
        
        try:
            # Detect language if auto
            if language == 'auto':
                # Quick sample extraction to detect language
                sample_config = ExtractConfig(
                    extraction_mode=ExtractMode.MULTIMODAL,
                    extraction_target="PER_PAGE",
                    chunk_mode="SECTION",
                    high_resolution_mode=False,
                    cite_sources=False,
                    use_reasoning=False,
                    confidence_scores=False
                )
                
                sample_agent = self.extractor.create_agent(
                    name=f"{agent_name}_sample",
                    data_schema=self.english_schema,
                    config=sample_config
                )
                
                sample_result = sample_agent.extract(pdf_path)
                
                # Analyze first few pages for language detection
                sample_text = ""
                for page_data in sample_result.data[:3]:
                    for block in page_data.get("page_content", []):
                        content = block.get("content", "")
                        if content:
                            sample_text += content + " "
                
                detected_language = self.language_detector.detect_language(sample_text)
                print(f"Detected language: {'Arabic' if detected_language == 'ar' else 'English'}")
            else:
                detected_language = language
            
            # Choose appropriate schema
            schema = self.arabic_schema if detected_language == 'ar' else self.english_schema
            
            # Create proper ExtractConfig object
            config = ExtractConfig(
                extraction_mode=ExtractMode.MULTIMODAL,
                extraction_target="PER_PAGE",
                chunk_mode="SECTION",
                high_resolution_mode=True,
                cite_sources=False,
                use_reasoning=False,
                confidence_scores=False
            )
            
            # Create agent with appropriate schema
            agent = self.extractor.create_agent(
                name=f"{agent_name}_{detected_language}",
                data_schema=schema,
                config=config
            )        
            
            result = agent.extract(pdf_path)
            return result.data, detected_language
            
        except Exception as e:
            print(f"Error extracting PDF {pdf_path}: {e}")
            return [], 'en'
    
    def convert_to_llamaindex_documents(self, extracted_data: List[Dict], 
                                       document_name: str, language: str) -> List[Document]:
        """Convert extracted data to LlamaIndex Document objects"""
        documents = []
        
        for page_idx, page_data in enumerate(extracted_data):
            page_number = page_idx + 1
            page_content = page_data.get("page_content", [])
            
            for block_idx, block in enumerate(page_content):
                content = self._extract_block_content(block, language)
                
                if not content.strip():
                    continue
                
                # Create LlamaIndex Document with rich metadata
                doc = Document(
                    text=content,
                    metadata={
                        "document_name": document_name,
                        "page_number": page_number,
                        "block_index": block_idx,
                        "block_type": block.get("block_type", "unknown"),
                        "language": language,
                        "source": f"{document_name}_page_{page_number}_block_{block_idx}",
                        "extracted_at": datetime.now().isoformat()
                    }
                )
                
                documents.append(doc)
        
        print(f"Created {len(documents)} LlamaIndex documents for {language.upper()}")
        return documents
    
    def _extract_block_content(self, block: Dict, language: str) -> str:
        """Extract text content from a block regardless of type"""
        block_type = block.get("block_type", "unknown")
        
        if block_type in ["title", "paragraph", "footnote"]:
            return block.get("content", "")
        
        elif block_type == "list":
            items = block.get("items", [])
            bullet = "•"
            return "\n".join(f"{bullet} {item}" for item in items if item.strip())
        
        elif block_type == "columnar_text":
            columns = block.get("columns", [])
            return "\n".join(col.get("content", "") for col in columns)
        
        elif block_type == "table":
            return self._format_table_content(block, language)
        
        return ""
    
    def _format_table_content(self, block: Dict, language: str) -> str:
        """Format table as readable text"""
        table_title = block.get("table_title", "")
        headers = block.get("headers", [])
        rows = block.get("rows", [])
        
        content_parts = []
        
        if table_title.strip():
            table_prefix = "جدول:" if language == 'ar' else "Table:"
            content_parts.append(f"{table_prefix} {table_title}")
        
        if headers:
            header_prefix = "الأعمدة:" if language == 'ar' else "Columns:"
            content_parts.append(f"{header_prefix} " + " | ".join(headers))
        
        for row in rows:
            if any(cell.strip() for cell in row):
                if headers and len(row) == len(headers):
                    row_parts = []
                    for header, cell in zip(headers, row):
                        if cell.strip():
                            row_parts.append(f"{header}: {cell}")
                    if row_parts:
                        content_parts.append(" | ".join(row_parts))
                else:
                    content_parts.append(" | ".join(cell for cell in row if cell.strip()))
        
        return "\n".join(content_parts)
    
    def setup_vector_store(self, collection_name: str) -> QdrantVectorStore:
        """Setup Qdrant vector store for LlamaIndex"""
        try:
            # Delete existing collection
            self.qdrant_client.delete_collection(collection_name)
            print(f"Deleted existing collection: {collection_name}")
        except Exception as e:
            print(f"Collection {collection_name} doesn't exist: {e}")
        
        # Create vector store
        vector_store = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name=collection_name,
        )
        
        # Store the vector store for later use
        self.vector_stores[collection_name] = vector_store
        
        print(f"Created vector store for collection: {collection_name}")
        return vector_store
    
    def create_index(self, documents: List[Document], 
                    collection_name: str, language: str) -> VectorStoreIndex:
        """Create LlamaIndex vector store index"""
        # Setup vector store
        vector_store = self.setup_vector_store(collection_name)
        
        # Create storage context
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        # Use language-specific LLM
        if language == 'ar':
            Settings.llm = Groq(model="llama3-70b-8192", api_key=self.groq_api_key)
        else:
            Settings.llm = Groq(model="llama3-70b-8192", api_key=self.groq_api_key_1)
        
        # Create index
        print(f"Creating vector index for {len(documents)} documents...")
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            node_parser=self.node_parser,
            show_progress=True
        )
        
        # Store index for later use
        self.indices[collection_name] = index
        
        print(f"Successfully created index for collection: {collection_name}")
        return index
    
    def load_existing_index(self, collection_name: str, language: str) -> Optional[VectorStoreIndex]:
        """Load existing index from Qdrant if it exists"""
        try:
            # Check if collection exists in Qdrant
            collections = self.qdrant_client.get_collections()
            collection_names = [c.name for c in collections.collections]
            
            if collection_name not in collection_names:
                print(f"Collection {collection_name} doesn't exist in Qdrant")
                return None
            
            # Create vector store connection
            vector_store = QdrantVectorStore(
                client=self.qdrant_client,
                collection_name=collection_name,
            )
            
            # Store the vector store
            self.vector_stores[collection_name] = vector_store
            
            # Create storage context
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            
            # Use language-specific LLM
            if language == 'ar':
                Settings.llm = Groq(model="llama3-70b-8192", api_key=self.groq_api_key)
            else:
                Settings.llm = Groq(model="llama3-70b-8192", api_key=self.groq_api_key_1)
            
            # Load the index
            index = VectorStoreIndex.from_vector_store(
                vector_store=vector_store,
                storage_context=storage_context
            )
            
            # Store index for later use
            self.indices[collection_name] = index
            
            print(f"Successfully loaded existing index for collection: {collection_name}")
            return index
            
        except Exception as e:
            print(f"Error loading existing index for {collection_name}: {e}")
            return None
    
    def process_pdf_to_llamaindex(self, pdf_path: str, 
                                 language: str = 'auto',
                                 document_name: str = None) -> Tuple[VectorStoreIndex, str]:
        """Complete pipeline: Extract PDF -> Create LlamaIndex Documents -> Build Index"""
        if document_name is None:
            document_name = os.path.basename(pdf_path).replace('.pdf', '')
        
        print(f"Processing PDF with LlamaIndex: {pdf_path}")
        
        # Step 1: Extract content
        print("1. Extracting PDF content...")
        extracted_data, detected_language = self.extract_pdf(pdf_path, language)
        
        if not extracted_data:
            print("Failed to extract PDF content")
            return None, None
        
        # Step 2: Convert to LlamaIndex Documents
        print("2. Converting to LlamaIndex documents...")
        documents = self.convert_to_llamaindex_documents(
            extracted_data, document_name, detected_language
        )
        
        # Step 3: Determine collection name
        collection_name = ARABIC_COLLECTION if detected_language == 'ar' else ENGLISH_COLLECTION
        print(f"Using collection: {collection_name}")
        
        # Step 4: Create index
        print("3. Creating vector index...")
        index = self.create_index(documents, collection_name, detected_language)
        
        print("✅ LlamaIndex processing pipeline completed!")
        return index, detected_language
    
    def create_query_engine(self, collection_name: str, language: str,
                           similarity_top_k: int = 5,
                           response_mode: str = "compact"):
        """Create a query engine for the specified collection with custom prompts"""
        
        # First try to load existing index if not in memory
        if collection_name not in self.indices:
            print(f"Loading existing index for collection: {collection_name}")
            index = self.load_existing_index(collection_name, language)
            if not index:
                print(f"No index found for collection: {collection_name}")
                return None
        else:
            index = self.indices[collection_name]
        
        # Use language-specific LLM
        if language == 'ar':
            Settings.llm = Groq(model="llama3-70b-8192", api_key=self.groq_api_key)
        else:
            Settings.llm = Groq(model="llama3-70b-8192", api_key=self.groq_api_key_1)
        
        # Create retriever with improved settings
        retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=similarity_top_k,
        )
        
        # Get language-specific prompt template
        qa_prompt_template = self.get_language_specific_prompt_template(language)
        
        # Create query engine with custom prompt and better response synthesis
        query_engine = RetrieverQueryEngine.from_args(
            retriever=retriever,
            response_mode=ResponseMode.COMPACT,
            text_qa_template=qa_prompt_template,  # Add custom prompt here
            node_postprocessors=[
                SimilarityPostprocessor(similarity_cutoff=0.3)  # Lower threshold
            ]
        )
        
        print(f"Created query engine for {collection_name} with custom {language} prompt (top_k={similarity_top_k})")
        return query_engine
    
    def ask_question(self, query: str, 
                    similarity_top_k: int = 5,
                    query_language: str = None) -> RAGResponse:
        """Ask a question using LlamaIndex RAG with custom prompts"""
        print(f"Processing question: {query}")
        
        # Detect language if not provided
        if query_language is None:
            query_language = self.language_detector.detect_language(query)
            print(f"Detected language: {'Arabic' if query_language == 'ar' else 'English'}")
        
        # Choose collection
        collection_name = ARABIC_COLLECTION if query_language == 'ar' else ENGLISH_COLLECTION
        print(f"Using collection: {collection_name}")
        
        # Create query engine with custom prompts
        query_engine = self.create_query_engine(
            collection_name, query_language, similarity_top_k
        )
        
        if not query_engine:
            no_info_message = ("لم أجد معلومات في قاعدة البيانات للإجابة على سؤالك." 
                              if query_language == 'ar' 
                              else "I couldn't find information in the database to answer your question.")
            
            return RAGResponse(
                answer=no_info_message,
                retrieved_chunks=[],
                query=query,
                query_language=query_language,
                confidence_score=0.0
            )
        
        # Generate response
        print("Generating answer with custom prompt...")
        try:
            response = query_engine.query(query)
            print(f"Raw response type: {type(response)}")
            print(f"Response content: {str(response)[:200]}...")
            
            # Extract source information
            retrieved_chunks = []
            sources = set()
            
            if hasattr(response, 'source_nodes') and response.source_nodes:
                print(f"Found {len(response.source_nodes)} source nodes")
                for i, source_node in enumerate(response.source_nodes):
                    print(f"Source node {i}: score={getattr(source_node, 'score', 'N/A')}")
                    
                    chunk_info = {
                        "score": getattr(source_node, 'score', 0.0),
                        "content": source_node.node.text[:500] + "..." if len(source_node.node.text) > 500 else source_node.node.text,
                        "metadata": source_node.node.metadata,
                        "document_name": source_node.node.metadata.get("document_name", "Unknown"),
                        "page_number": source_node.node.metadata.get("page_number", "?"),
                        "block_type": source_node.node.metadata.get("block_type", "unknown"),
                        "language": source_node.node.metadata.get("language", query_language)
                    }
                    retrieved_chunks.append(chunk_info)
                    
                    sources.add(f"{chunk_info['document_name']} (page {chunk_info['page_number']})")
            else:
                print("No source nodes found in response")
            
            # Calculate confidence
            confidence = (sum(chunk['score'] for chunk in retrieved_chunks) / len(retrieved_chunks) 
                         if retrieved_chunks else 0.0)
            
            # Ensure we have a proper response
            answer_text = str(response) if response else "No response generated"
            
            return RAGResponse(
                answer=answer_text,
                retrieved_chunks=retrieved_chunks,
                query=query,
                query_language=query_language,
                confidence_score=confidence,
                sources=list(sources)
            )
            
        except Exception as e:
            print(f"Error during query processing: {e}")
            import traceback
            traceback.print_exc()
            
            error_message = (f"حدث خطأ أثناء معالجة السؤال: {str(e)}" 
                           if query_language == 'ar' 
                           else f"An error occurred while processing the question: {str(e)}")
            
            return RAGResponse(
                answer=error_message,
                retrieved_chunks=[],
                query=query,
                query_language=query_language,
                confidence_score=0.0
            )

    def ask_question_with_manual_retrieval(self, query: str, 
                                         similarity_top_k: int = 5,
                                         query_language: str = None) -> RAGResponse:
        """Alternative method using manual retrieval and custom prompt formatting"""
        print(f"Processing question with manual retrieval: {query}")
        
        # Detect language if not provided
        if query_language is None:
            query_language = self.language_detector.detect_language(query)
            print(f"Detected language: {'Arabic' if query_language == 'ar' else 'English'}")
        
        # Choose collection
        collection_name = ARABIC_COLLECTION if query_language == 'ar' else ENGLISH_COLLECTION
        print(f"Using collection: {collection_name}")
        
        # Load index if needed
        if collection_name not in self.indices:
            print(f"Loading existing index for collection: {collection_name}")
            index = self.load_existing_index(collection_name, query_language)
            if not index:
                print(f"No index found for collection: {collection_name}")
                no_info_message = ("لم أجد معلومات في قاعدة البيانات للإجابة على سؤالك." 
                                  if query_language == 'ar' 
                                  else "I couldn't find information in the database to answer your question.")
                
                return RAGResponse(
                    answer=no_info_message,
                    retrieved_chunks=[],
                    query=query,
                    query_language=query_language,
                    confidence_score=0.0
                )
        else:
            index = self.indices[collection_name]
        
        try:
            # Manual retrieval
            retriever = VectorIndexRetriever(
                index=index,
                similarity_top_k=similarity_top_k,
            )
            
            retrieved_nodes = retriever.retrieve(query)
            
            # Extract context and metadata
            retrieved_chunks = []
            sources = set()
            context_parts = []
            
            for i, node in enumerate(retrieved_nodes):
                chunk_info = {
                    "score": getattr(node, 'score', 0.0),
                    "content": node.node.text[:500] + "..." if len(node.node.text) > 500 else node.node.text,
                    "metadata": node.node.metadata,
                    "document_name": node.node.metadata.get("document_name", "Unknown"),
                    "page_number": node.node.metadata.get("page_number", "?"),
                    "block_type": node.node.metadata.get("block_type", "unknown"),
                    "language": node.node.metadata.get("language", query_language)
                }
                retrieved_chunks.append(chunk_info)
                
                sources.add(f"{chunk_info['document_name']} (page {chunk_info['page_number']})")
                
                # Add to context with source information
                context_parts.append(f"المصدر {i+1} - {chunk_info['document_name']} (صفحة {chunk_info['page_number']}):\n{node.node.text}" 
                                   if query_language == 'ar' 
                                   else f"Source {i+1} - {chunk_info['document_name']} (page {chunk_info['page_number']}):\n{node.node.text}")
            
            # Create context string
            context = "\n\n".join(context_parts)
            
            # Use language-specific LLM
            if query_language == 'ar':
                llm = Groq(model="llama3-70b-8192", api_key=self.groq_api_key)
            else:
                llm = Groq(model="llama3-70b-8192", api_key=self.groq_api_key_1)
            
            # Create custom prompt
            custom_prompt = self.create_rag_prompt(query, context, query_language)
            
            # Generate response
            print("Generating answer with custom prompt...")
            response = llm.complete(custom_prompt)
            
            # Calculate confidence
            confidence = (sum(chunk['score'] for chunk in retrieved_chunks) / len(retrieved_chunks) 
                         if retrieved_chunks else 0.0)
            
            return RAGResponse(
                answer=str(response),
                retrieved_chunks=retrieved_chunks,
                query=query,
                query_language=query_language,
                confidence_score=confidence,
                sources=list(sources)
            )
            
        except Exception as e:
            print(f"Error during manual retrieval and query processing: {e}")
            import traceback
            traceback.print_exc()
            
            error_message = (f"حدث خطأ أثناء معالجة السؤال: {str(e)}" 
                           if query_language == 'ar' 
                           else f"An error occurred while processing the question: {str(e)}")
            
            return RAGResponse(
                answer=error_message,
                retrieved_chunks=[],
                query=query,
                query_language=query_language,
                confidence_score=0.0
            )

def main():
    """Example usage with LlamaIndex-based multilingual RAG system"""
    
    # Initialize system
    rag_system = MultilingualRAGWithLlamaIndex(
        llama_api_key=LLAMA_CLOUD_API_KEY,
        qdrant_url=QDRANT_URL,
        qdrant_api_key=QDRANT_API_KEY,
        groq_api_key=GROQ_API_KEY,
        embedding_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        chunk_size=512,
        chunk_overlap=50
    )
    
    print("="*60)
    print("LLAMAINDEX MULTILINGUAL RAG SYSTEM WITH CUSTOM PROMPTS")
    print("="*60)
    
    # Process documents
    # english_pdf_path = "Documents/saudi-economy-watch.pdf"
    # print("\n📄 Processing English PDF...")
    # english_index, english_lang = rag_system.process_pdf_to_llamaindex(english_pdf_path, "en")

     
    # arab_pdf_path = "Documents/ksa-personal-data-protection-law-series-part-1-ar.pdf"
    # print("\n📄 Processing Arab PDF...")
    # arab_index, arab_lang = rag_system.process_pdf_to_llamaindex(arab_pdf_path, "ar")
    
    # Test questions - more specific to document content
    test_questions = [
        "could you give me Contact list of Saudi Economy Watch?",
        "أذكر لي حقوق أصحاب البیانات الشخصیة، أصحاب البيانات ووصف الحق مع مواد النظام"
    ]
    
    print("\n" + "="*60)
    print("TESTING Q&A WITH CUSTOM PROMPTS")
    print("="*60)
    
    for question in test_questions:
        print(f"\n🤔 Question: {question}")
        print("-" * 50)
        
        # Test with built-in LlamaIndex query engine
        print("🔧 Using LlamaIndex Query Engine with Custom Prompts:")
        response1 = rag_system.ask_question(
            query=question,
            similarity_top_k=15
        )
        
        print(f"🌐 Language: {'Arabic' if response1.query_language == 'ar' else 'English'}")
        print(f"📋 Answer:")
        print(response1.answer)
        print(f"📊 Confidence: {response1.confidence_score:.3f}")
        print(f"📚 Sources: {', '.join(response1.sources) if response1.sources else 'None'}")
        
        print("\n" + "-" * 30)
        
        # # Test with manual retrieval method
        # print("🔧 Using Manual Retrieval with Custom Prompts:")
        # response2 = rag_system.ask_question_with_manual_retrieval(
        #     query=question,
        #     similarity_top_k=10
        # )
        
        # print(f"📋 Answer:")
        # print(response2.answer)
        # print(f"📊 Confidence: {response2.confidence_score:.3f}")
        # print(f"📚 Sources: {', '.join(response2.sources) if response2.sources else 'None'}")
        
        # print("\n" + "="*60)

if __name__ == "__main__":
    main()