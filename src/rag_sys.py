import json
import uuid
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import hashlib
import re
import random,time
# Vector database and embeddings
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer

import openai
import tiktoken

# Your existing extraction imports
from llama_cloud_services import LlamaExtract
from dotenv import load_dotenv
import os
import requests
load_dotenv()

# fro deployment
import streamlit as st
GROQ_API_KEY_1=st.secrets["groq_api_key_1"]
GROQ_API_KEY = st.secrets["groq_api_key"]
QDRANT_API_KEY = st.secrets["qdrant_api_key"]
QDRANT_URL = st.secrets["qdrant_url"]
LLAMA_CLOUD_API_KEY = st.secrets["llama_cloud_api_key"]


# Collection names for different languages
ARABIC_COLLECTION = "arabic_docs"
ENGLISH_COLLECTION = "english_docs"

@dataclass
class DocumentChunk:
    """Represents a chunk of document content with metadata"""
    id: str
    content: str
    chunk_type: str  # title, paragraph, list_item, table_row, hybrid, etc.
    page_number: int
    document_name: str
    block_index: int
    language: str  # 'ar' or 'en'
    parent_content: Optional[str] = None  # For context (e.g., table title for rows)
    metadata: Dict[str, Any] = None
    token_count: int = 0
    chunk_strategy: str = "original"  # original, merged, split, hybrid

@dataclass
class RAGResponse:
    """Response from RAG system"""
    answer: str
    retrieved_chunks: List[Dict]
    query: str
    query_language: str  # Detected language
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
        
        # Clean text and remove numbers/punctuation for analysis
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
        if arabic_ratio > 0.3:
            return 'ar'
        else:
            return 'en'
    
    def get_language_confidence(self, text: str) -> Dict[str, float]:
        """Get confidence scores for both languages"""
        if not text or not text.strip():
            return {'ar': 0.0, 'en': 1.0}
        
        clean_text = re.sub(r'[0-9\s\.,!?;:\-\(\)\[\]\"\']+', '', text)
        
        if not clean_text:
            return {'ar': 0.0, 'en': 1.0}
        
        arabic_chars = len(self.arabic_pattern.findall(clean_text))
        english_chars = len(self.english_pattern.findall(clean_text))
        
        total_chars = arabic_chars + english_chars
        
        if total_chars == 0:
            return {'ar': 0.0, 'en': 1.0}
        
        arabic_confidence = arabic_chars / total_chars
        english_confidence = english_chars / total_chars
        
        return {
            'ar': arabic_confidence,
            'en': english_confidence
        }

class SimpleGroqChat:
    """Simple Groq API wrapper for chat completions."""
    
    def __init__(self, api_key: str, model: str = "llama3-70b-8192"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
    
    def chat(self, messages: list,lang, temperature: float = 0.1, max_tokens: int = 1000) -> str:
        """Send chat completion request to Groq."""
        if lang=="ar":
            self.api_key=GROQ_API_KEY
        else:
            self.api_key=GROQ_API_KEY_1
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

class HybridChunkingStrategy:
    """Advanced chunking strategy that combines elements intelligently"""
    
    def __init__(self, 
                 target_chunk_size: int = 400,
                 max_chunk_size: int = 600,
                 min_chunk_size: int = 100,
                 model_name: str = "cl100k_base"):
        self.target_chunk_size = target_chunk_size
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        
        # Initialize tokenizer for accurate token counting
        try:
            self.tokenizer = tiktoken.get_encoding(model_name)
        except:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
            print(f"Warning: Could not load {model_name} tokenizer, using cl100k_base")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.tokenizer.encode(text))
    
    def should_merge_blocks(self, current_block: Dict, next_block: Dict, 
                           current_tokens: int, next_tokens: int) -> bool:
        """Determine if two blocks should be merged"""
        total_tokens = current_tokens + next_tokens
        
        # Don't merge if it would exceed max chunk size
        if total_tokens > self.max_chunk_size:
            return False
        
        current_type = current_block.get("block_type")
        next_type = next_block.get("block_type")
        
        # Merge small titles with following content
        if current_type == "title" and current_tokens < 50:
            return True
        
        # Merge small paragraphs if total is reasonable
        if (current_type == "paragraph" and next_type == "paragraph" and 
            total_tokens <= self.target_chunk_size):
            return True
        
        # Merge list items if they're small
        if (current_type == "list" and next_type == "list" and 
            total_tokens <= self.target_chunk_size):
            return True
        
        # Merge footnotes with preceding content if small
        if next_type == "footnote" and next_tokens < 100:
            return True
        
        return False
    
    def split_large_content(self, content: str, chunk_type: str, 
                           base_metadata: Dict, language: str = 'en') -> List[Tuple[str, Dict]]:
        """Split large content into smaller chunks while preserving meaning"""
        tokens = self.count_tokens(content)
        
        if tokens <= self.max_chunk_size:
            return [(content, base_metadata)]
        
        # Split based on content type and language
        if chunk_type == "paragraph":
            return self._split_paragraph(content, base_metadata, language)
        elif chunk_type == "list":
            return self._split_list(content, base_metadata)
        else:
            # Fallback: split on word boundaries
            return self._split_on_words(content, base_metadata)
    
    def _split_paragraph(self, content: str, base_metadata: Dict, language: str) -> List[Tuple[str, Dict]]:
        """Split paragraph on sentence boundaries based on language"""
        if language == 'ar':
            # Arabic sentence boundaries
            sentence_endings = r'[.!?؟。]'
        else:
            # English sentence boundaries
            sentence_endings = r'[.!?]'
        
        sentences = re.split(f'({sentence_endings})', content)
        
        chunks = []
        current_chunk = ""
        
        for i in range(0, len(sentences), 2):
            sentence = sentences[i]
            ending = sentences[i + 1] if i + 1 < len(sentences) else ""
            full_sentence = sentence + ending
            
            if not sentence.strip():
                continue
            
            test_chunk = current_chunk + " " + full_sentence if current_chunk else full_sentence
            
            if self.count_tokens(test_chunk) > self.max_chunk_size and current_chunk:
                # Save current chunk and start new one
                chunks.append((current_chunk.strip(), base_metadata.copy()))
                current_chunk = full_sentence
            else:
                current_chunk = test_chunk
        
        if current_chunk.strip():
            chunks.append((current_chunk.strip(), base_metadata.copy()))
        
        return chunks
    
    def _split_list(self, content: str, base_metadata: Dict) -> List[Tuple[str, Dict]]:
        """Split list content intelligently"""
        # Split on bullet points or numbers
        items = re.split(r'\n(?=[-•*\d+\.])', content)
        
        chunks = []
        current_chunk = ""
        
        for item in items:
            if not item.strip():
                continue
            
            test_chunk = current_chunk + "\n" + item if current_chunk else item
            
            if self.count_tokens(test_chunk) > self.max_chunk_size and current_chunk:
                chunks.append((current_chunk.strip(), base_metadata.copy()))
                current_chunk = item
            else:
                current_chunk = test_chunk
        
        if current_chunk.strip():
            chunks.append((current_chunk.strip(), base_metadata.copy()))
        
        return chunks
    
    def _split_on_words(self, content: str, base_metadata: Dict) -> List[Tuple[str, Dict]]:
        """Fallback: split on word boundaries"""
        words = content.split()
        chunks = []
        current_chunk = ""
        
        for word in words:
            test_chunk = current_chunk + " " + word if current_chunk else word
            
            if self.count_tokens(test_chunk) > self.max_chunk_size and current_chunk:
                chunks.append((current_chunk.strip(), base_metadata.copy()))
                current_chunk = word
            else:
                current_chunk = test_chunk
        
        if current_chunk.strip():
            chunks.append((current_chunk.strip(), base_metadata.copy()))
        
        return chunks

class MultilingualRAGSystem:
    """Enhanced RAG system with multilingual support for Arabic and English"""
    
    def __init__(self, 
                 llama_api_key: str,
                 qdrant_url: str,
                 qdrant_api_key: str,
                 groq_api_key: str = None,
                 openai_api_key: str = None,
                 embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                 llm_provider: str = "groq",
                 chunk_target_size: int = 400,
                 chunk_max_size: int = 600):
        """Initialize the multilingual RAG system"""
        
        self.extractor = LlamaExtract(api_key=llama_api_key)
        self.language_detector = LanguageDetector()
        
        # Initialize Qdrant Cloud client
        self.qdrant_client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
        )
        
        # Initialize multilingual embedding model
        self.embedding_model = SentenceTransformer(embedding_model)
        print(f"Initialized embedding model: {embedding_model}")
        print(f"Embedding dimension: {self.embedding_model.get_sentence_embedding_dimension()}")
        
        # Initialize chunking strategy
        self.chunking_strategy = HybridChunkingStrategy(
            target_chunk_size=chunk_target_size,
            max_chunk_size=chunk_max_size
        )
        
        # Initialize LLM
        self.llm_provider = llm_provider
        if llm_provider == "groq" and groq_api_key:
            self.groq_client = SimpleGroqChat(groq_api_key)
            print("Initialized Groq client")
        elif llm_provider == "openai" and openai_api_key:
            self.openai_client = openai.OpenAI(api_key=openai_api_key)
            print("Initialized OpenAI client")
        else:
            print(f"Warning: No valid API key provided for {llm_provider}")
        
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
    
    def extract_pdf(self, pdf_path: str, indice: int , language: str = 'auto', agent_name: str = "multilingual_agent") -> Tuple[List[Dict], str]:
        """Extract structured content from PDF with language detection"""
        import time
        import random
        import uuid
    
        timestamp = str(int(time.time()))        
        random_suffix = str(random.randint(1000, 9999)) 
        unique_id = str(uuid.uuid4())[:8]       

        # Combine them into a unique agent name
        agent_name = f"agent_{timestamp}_{random_suffix}_{unique_id}"
        try:
            # Detect language if auto
            if language == 'auto':
                # Quick sample extraction to detect language
                sample_agent = self.extractor.create_agent(
                    name=f"{agent_name}_sample",
                    data_schema=self.english_schema,  # Use generic schema for sampling
                    config={
                        "extraction_mode": "MULTIMODAL",
                        "extraction_target": "PER_PAGE",
                        "chunk_mode": "SECTION",
                        "high_resolution_mode": False,  # Faster for language detection
                        "cite_sources": False,
                        "use_reasoning": False,
                        "confidence_scores": False
                    }
                )
                
                sample_result = sample_agent.extract(pdf_path)
                
                # Analyze first few pages for language detection
                sample_text = ""
                for page_data in sample_result.data[:3]:  # Check first 3 pages
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
            
            # Create agent with appropriate schema
            agent = self.extractor.create_agent(
                name=f"{agent_name}_{detected_language}",
                data_schema=schema,
                config={
                    "extraction_mode": "MULTIMODAL",
                    "extraction_target": "PER_PAGE", 
                    "chunk_mode": "SECTION",
                    "high_resolution_mode": True,
                    "cite_sources": False,
                    "use_reasoning": False,          
                    "confidence_scores": True
                }
            )        
            
            result = agent.extract(pdf_path)
            return result.data, detected_language
            
        except Exception as e:
            print(f"Error extracting PDF {pdf_path}: {e}")
            return [], 'en'
    
    def chunk_extracted_content_hybrid(self, extracted_data: List[Dict], 
                                      document_name: str, language: str) -> List[DocumentChunk]:
        """Convert extracted data into optimally sized chunks using hybrid strategy"""
        all_chunks = []
        
        for page_idx, page_data in enumerate(extracted_data):
            page_number = page_idx + 1
            page_content = page_data.get("page_content", [])
            
            # Process blocks with lookahead for intelligent merging
            chunks = self._process_page_blocks_hybrid(page_content, page_number, document_name, language)
            all_chunks.extend(chunks)
        
        print(f"Created {len(all_chunks)} hybrid chunks for {language.upper()} document")
        self._print_chunk_statistics(all_chunks)
        
        return all_chunks
    
    def _process_page_blocks_hybrid(self, blocks: List[Dict], page_number: int, 
                                   document_name: str, language: str) -> List[DocumentChunk]:
        chunks = []
        i = 0

        while i < len(blocks):
            current_block = blocks[i]
            current_content = self._extract_block_content(current_block, language)
            current_tokens = self.chunking_strategy.count_tokens(current_content)

            if current_block.get("block_type") == "table":
                table_chunks = self._create_table_chunks(current_block, page_number, document_name, i, language)
                chunks.extend(table_chunks)
                print(f"[Page {page_number}] Table split into {len(table_chunks)} chunk(s).")
                i += 1
                continue

            merged_content = current_content
            merged_blocks = [current_block]
            j = i + 1

            while j < len(blocks):
                next_block = blocks[j]
                next_content = self._extract_block_content(next_block, language)
                next_tokens = self.chunking_strategy.count_tokens(next_content)

                if next_block.get("block_type") == "table":
                    break

                if self.chunking_strategy.should_merge_blocks(
                    current_block, next_block,
                    self.chunking_strategy.count_tokens(merged_content),
                    next_tokens):
                    merged_content += "\n\n" + next_content
                    merged_blocks.append(next_block)
                    j += 1
                else:
                    break

            merged_chunks = self._create_chunks_from_merged_content(
                merged_content, merged_blocks, page_number, document_name, i, language
            )
            chunks.extend(merged_chunks)

            print(f"[Page {page_number}] Created {len(merged_chunks)} chunk(s) from merged blocks starting at index {i}.")
            i = j

        return chunks
    
    def _extract_block_content(self, block: Dict, language: str) -> str:
        """Extract text content from a block regardless of type"""
        block_type = block.get("block_type", "unknown")
        
        if block_type in ["title", "paragraph", "footnote"]:
            return block.get("content", "")
        
        elif block_type == "list":
            items = block.get("items", [])
            bullet = "•" if language == 'en' else "•"
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
                    # Format as header: value pairs
                    row_parts = []
                    for header, cell in zip(headers, row):
                        if cell.strip():
                            row_parts.append(f"{header}: {cell}")
                    if row_parts:
                        content_parts.append(" | ".join(row_parts))
                else:
                    # Simple row format
                    content_parts.append(" | ".join(cell for cell in row if cell.strip()))
        
        return "\n".join(content_parts)
    
    def _create_table_chunks(self, block: Dict, page_number: int, 
                            document_name: str, block_index: int, language: str) -> List[DocumentChunk]:
        rows = block.get("rows", [])
        headers = block.get("headers", [])
        table_title = block.get("table_title", "")

        # Convert each row to text
        row_texts = []
        for row in rows:
            if headers and len(row) == len(headers):
                row_texts.append(" | ".join(f"{h}: {c}" for h, c in zip(headers, row) if c.strip()))
            else:
                row_texts.append(" | ".join(c for c in row if c.strip()))

        chunks = []
        current_rows = []
        current_tokens = 0
        max_tokens = self.chunking_strategy.max_chunk_size

        for idx, row_text in enumerate(row_texts):
            row_tokens = self.chunking_strategy.count_tokens(row_text)

            # If adding this row exceeds max chunk size, start a new chunk
            if current_rows and current_tokens + row_tokens > max_tokens:
                chunk = self._build_table_chunk(current_rows, headers, table_title,
                                                page_number, document_name, block_index, len(chunks), language)
                chunks.append(chunk)
                current_rows = []
                current_tokens = 0

            current_rows.append(row_text)
            current_tokens += row_tokens

        # Final remaining rows
        if current_rows:
            chunk = self._build_table_chunk(current_rows, headers, table_title,
                                            page_number, document_name, block_index, len(chunks), language)
            chunks.append(chunk)

        # Update all chunks with total_splits information
        total_splits = len(chunks)
        for idx, chunk in enumerate(chunks):
            if chunk.metadata is None:
                chunk.metadata = {}
            chunk.metadata['total_splits'] = total_splits
            chunk.metadata['split_index'] = idx

        return chunks

    def _build_table_chunk(self, row_texts, headers, table_title, page_number, 
                           document_name, block_index, split_index, language):
        content_parts = []
        
        if table_title.strip():
            table_prefix = "جدول:" if language == 'ar' else "Table:"
            content_parts.append(f"{table_prefix} {table_title}")
        
        if headers:
            header_prefix = "الأعمدة:" if language == 'ar' else "Columns:"
            content_parts.append(f"{header_prefix} " + " | ".join(headers))
        
        content_parts.extend(row_texts)

        content = "\n".join(content_parts)
        tokens = self.chunking_strategy.count_tokens(content)

        chunk_id = self._generate_chunk_id(document_name, page_number, f"{block_index}_{split_index}", content)

        return DocumentChunk(
            id=chunk_id,
            content=content,
            chunk_type="table",
            page_number=page_number,
            document_name=document_name,
            block_index=f"{block_index}_{split_index}",
            language=language,
            metadata={
                "headers": headers,
                "row_count": len(row_texts),
                "original_block_type": "table",
                "split_index": split_index,
                "row_texts": row_texts
            },
            token_count=tokens,
            chunk_strategy="split" if split_index > 0 else "intact"
        )
    
    def _create_chunks_from_merged_content(self, merged_content: str, 
                                         merged_blocks: List[Dict],
                                         page_number: int, document_name: str, 
                                         base_index: int, language: str) -> List[DocumentChunk]:
        """Create optimally-sized chunks from merged content"""
        tokens = self.chunking_strategy.count_tokens(merged_content)
        
        base_metadata = {
            "merged_blocks": len(merged_blocks),
            "original_block_types": [b.get("block_type") for b in merged_blocks]
        }
        
        if tokens <= self.chunking_strategy.max_chunk_size:
            # Content fits in one chunk
            chunk_type = self._determine_chunk_type(merged_blocks)
            chunk_id = self._generate_chunk_id(document_name, page_number, 
                                             base_index, merged_content)
            
            return [DocumentChunk(
                id=chunk_id,
                content=merged_content,
                chunk_type=chunk_type,
                page_number=page_number,
                document_name=document_name,
                block_index=base_index,
                language=language,
                metadata=base_metadata,
                token_count=tokens,
                chunk_strategy="merged" if len(merged_blocks) > 1 else "original"
            )]
        
        else:
            # Need to split the merged content
            chunk_type = self._determine_chunk_type(merged_blocks)
            split_chunks = self.chunking_strategy.split_large_content(
                merged_content, chunk_type, base_metadata, language
            )
            
            result_chunks = []
            for idx, (chunk_content, chunk_metadata) in enumerate(split_chunks):
                chunk_id = self._generate_chunk_id(document_name, page_number, 
                                                 f"{base_index}_{idx}", chunk_content)
                
                chunk_metadata["split_index"] = idx
                chunk_metadata["total_splits"] = len(split_chunks)
                
                result_chunks.append(DocumentChunk(
                    id=chunk_id,
                    content=chunk_content,
                    chunk_type=chunk_type,
                    page_number=page_number,
                    document_name=document_name,
                    block_index=f"{base_index}_{idx}",
                    language=language,
                    metadata=chunk_metadata,
                    token_count=self.chunking_strategy.count_tokens(chunk_content),
                    chunk_strategy="split"
                ))
            
            return result_chunks
    
    def _determine_chunk_type(self, blocks: List[Dict]) -> str:
        """Determine the primary chunk type from merged blocks"""
        if len(blocks) == 1:
            return blocks[0].get("block_type", "unknown")
        
        # For merged blocks, use hybrid type or most common type
        types = [b.get("block_type") for b in blocks]
        type_counts = {}
        
        for t in types:
            type_counts[t] = type_counts.get(t, 0) + 1
        
        most_common = max(type_counts, key=type_counts.get)
        
        if len(set(types)) > 1:
            return f"hybrid_{most_common}"
        else:
            return most_common
    
    def _generate_chunk_id(self, document_name: str, page_number: int, 
                          block_index: Any, content: str) -> str:
        """Generate unique chunk ID"""
        content_hash = hashlib.md5(
            f"{document_name}_{page_number}_{block_index}_{content}".encode()
        ).hexdigest()[:8]
        return f"{document_name}_{page_number}_{block_index}_{content_hash}"
    
    def _print_chunk_statistics(self, chunks: List[DocumentChunk]):
        """Print statistics about created chunks"""
        if not chunks:
            return
        
        token_counts = [chunk.token_count for chunk in chunks]
        chunk_types = {}
        strategies = {}
        
        for chunk in chunks:
            chunk_types[chunk.chunk_type] = chunk_types.get(chunk.chunk_type, 0) + 1
            strategies[chunk.chunk_strategy] = strategies.get(chunk.chunk_strategy, 0) + 1
        
        print(f"""
📊 Chunk Statistics:
   Total chunks: {len(chunks)}
   Token range: {min(token_counts)} - {max(token_counts)}
   Average tokens: {sum(token_counts) / len(token_counts):.1f}
   
📝 Chunk types: {dict(chunk_types)}
🔄 Strategies: {dict(strategies)}
        """)
    
    def setup_qdrant_collection(self, collection_name: str, vector_size: int = None):
        """Create or recreate Qdrant collection"""
        if vector_size is None:
            vector_size = self.embedding_model.get_sentence_embedding_dimension()
            
        try:
            self.qdrant_client.delete_collection(collection_name)
            print(f"Deleted existing collection: {collection_name}")
        except Exception as e:
            print(f"Collection {collection_name} doesn't exist or couldn't be deleted: {e}")
        
        self.qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )
        print(f"Created collection: {collection_name} with vector size: {vector_size}")
    
    def index_chunks_to_qdrant(self, chunks: List[DocumentChunk], collection_name: str):
        """Index document chunks to Qdrant with embeddings"""
        if not chunks:
            print("No chunks to index")
            return
        
        print(f"Generating embeddings for {len(chunks)} chunks...")
        texts = [chunk.content for chunk in chunks]
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        
        points = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            point_id = str(uuid.uuid4())
            
            point = PointStruct(
                id=point_id,
                vector=embedding.tolist(),
                payload={
                    "chunk_id": chunk.id,
                    "content": chunk.content,
                    "chunk_type": chunk.chunk_type,
                    "page_number": chunk.page_number,
                    "document_name": chunk.document_name,
                    "block_index": chunk.block_index,
                    "language": chunk.language,
                    "parent_content": chunk.parent_content,
                    "metadata": chunk.metadata,
                    "token_count": chunk.token_count,
                    "chunk_strategy": chunk.chunk_strategy,
                    "indexed_at": datetime.now().isoformat()
                }
            )
            points.append(point)
        
        # Upload in batches
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            try:
                self.qdrant_client.upsert(collection_name=collection_name, points=batch)
                print(f"Uploaded batch {i//batch_size + 1}/{(len(points) + batch_size - 1)//batch_size}")
            except Exception as e:
                print(f"Error uploading batch {i//batch_size + 1}: {e}")
        
        print(f"Successfully indexed {len(chunks)} chunks to collection '{collection_name}'")
    
    def process_pdf_to_qdrant_multilingual(self, pdf_path: str, 
                                           indice: int,
                                          language: str = 'auto',
                                          document_name: str = None):
        """Complete pipeline: Extract PDF -> Detect Language -> Chunk -> Index to appropriate collection"""
        if document_name is None:
            document_name = os.path.basename(pdf_path).replace('.pdf', '')
        
        print(f"Processing PDF with multilingual support: {pdf_path}")
        
        # Step 1: Extract structured content with language detection
        print("1. Extracting PDF content and detecting language...")
        extracted_data, detected_language = self.extract_pdf(pdf_path, indice,language)
        if not extracted_data:
            print("Failed to extract PDF content")
            return None, None
        
        # Step 2: Determine collection name based on language
        collection_name = ARABIC_COLLECTION if detected_language == 'ar' else ENGLISH_COLLECTION
        print(f"Using collection: {collection_name} for language: {detected_language}")
        
        # Step 3: Hybrid chunking
        print("2. Applying hybrid chunking strategy...")
        chunks = self.chunk_extracted_content_hybrid(extracted_data, document_name, detected_language)
        
        # Step 4: Setup Qdrant collection
        print("3. Setting up Qdrant collection...")
        self.setup_qdrant_collection(collection_name)
        
        # Step 5: Index to Qdrant
        print("4. Indexing to Qdrant...")
        self.index_chunks_to_qdrant(chunks, collection_name)
        
        print("✅ Multilingual processing pipeline completed successfully!")
        return chunks, detected_language
    
    def retrieve_documents(self, query: str, query_language: str = None, 
                          limit: int = 5, filter_by_type: List[str] = None) -> List[Dict]:
        """Retrieve relevant documents for a query from appropriate language collection"""
        
        # Detect query language if not provided
        if query_language is None:
            query_language = self.language_detector.detect_language(query)
        
        # Choose appropriate collection
        collection_name = ARABIC_COLLECTION if query_language == 'ar' else ENGLISH_COLLECTION
        
        print(f"Query language detected as: {'Arabic' if query_language == 'ar' else 'English'}")
        print(f"Searching in collection: {collection_name}")
        
        query_embedding = self.embedding_model.encode([query])[0]
        
        search_filter = None
        if filter_by_type:
            search_filter = {
                "must": [
                    {
                        "key": "chunk_type",
                        "match": {"any": filter_by_type}
                    }
                ]
            }
        
        try:
            results = self.qdrant_client.query_points(
                collection_name=collection_name,
                query=query_embedding.tolist(),
                limit=limit,
                query_filter=search_filter
            )
        except Exception as e:
            print(f"Error querying collection {collection_name}: {e}")
            return []
        
        formatted_results = []
        for result in results.points:
            formatted_results.append({
                "score": result.score,
                "content": result.payload["content"],
                "chunk_type": result.payload["chunk_type"],
                "page_number": result.payload["page_number"],
                "document_name": result.payload["document_name"],
                "language": result.payload.get("language", query_language),
                "parent_content": result.payload.get("parent_content"),
                "metadata": result.payload.get("metadata", {}),
                "chunk_id": result.payload.get("chunk_id"),
                "token_count": result.payload.get("token_count", 0),
                "chunk_strategy": result.payload.get("chunk_strategy", "unknown")
            })
        
        return formatted_results
    
    def generate_answer(self, query: str, retrieved_chunks: List[Dict], 
                       query_language: str, model: str = None, temperature: float = 0.1) -> str:
        """Generate answer using LLM based on retrieved context and query language"""
        context = self._format_context(retrieved_chunks, query_language)
        prompt = self._create_rag_prompt(query, context, query_language)
        
        if self.llm_provider == "groq":
            return self._generate_with_groq(prompt,query_language, temperature)
        else:
            return "Error: No valid LLM provider configured."
    
    def _format_context(self, retrieved_chunks: List[Dict], language: str) -> str:
        """Format retrieved chunks into context string with language-specific formatting"""
        from collections import defaultdict
        
        # Group chunks by base block_index
        grouped_chunks = defaultdict(list)
        for chunk in retrieved_chunks:
            full_block_index = str(chunk.get('block_index', ''))
            base_index = full_block_index.split('_')[0] if '_' in full_block_index else full_block_index
            grouped_chunks[base_index].append(chunk)
        
        context_parts = []
        source_counter = 1
        
        # Language-specific labels
        if language == 'ar':
            source_label = "مصدر"
            content_type_label = "نوع المحتوى:"
            page_label = "الصفحة:"
            document_label = "الوثيقة:"
            token_count_label = "عدد الرموز:"
            strategy_label = "استراتيجية التقسيم:"
            part_label = "جزء"
            of_label = "من"
            combined_label = "(مجمّع من"
            pieces_label = "قطع)"
            content_label = "المحتوى:"
            original_context_label = "السياق الأصلي:"
        else:
            source_label = "Source"
            content_type_label = "Content Type:"
            page_label = "Page:"
            document_label = "Document:"
            token_count_label = "Token Count:"
            strategy_label = "Chunking Strategy:"
            part_label = "Part"
            of_label = "of"
            combined_label = "(combined from"
            pieces_label = "pieces)"
            content_label = "Content:"
            original_context_label = "Original Context:"
        
        for base_index, chunks in grouped_chunks.items():
            # Sort chunks by split_index to preserve order
            def sort_key(c):
                return c.get('metadata', {}).get('split_index', -1)
            
            chunks_sorted = sorted(chunks, key=sort_key)
            
            # Combine contents of all chunks for this block
            combined_content = "\n".join(c['content'] for c in chunks_sorted)
            
            # Representative metadata from the first chunk
            representative = chunks_sorted[0]
            meta = representative.get('metadata', {})
            chunk_type = representative.get('chunk_type', 'unknown')
            page_number = representative.get('page_number', '?')
            document_name = representative.get('document_name', '?')
            token_count = sum(c.get('token_count', 0) for c in chunks_sorted)
            chunk_strategy = representative.get('chunk_strategy', 'unknown')
            parent_content = representative.get('parent_content')
            total_splits = meta.get('total_splits', len(chunks_sorted))
            
            context_part = f"""
{source_label} {source_counter}:
{content_type_label} {chunk_type}
{page_label} {page_number}
{document_label} {document_name}
{token_count_label} {token_count}
{strategy_label} {chunk_strategy}
{part_label} 1 {of_label} {total_splits} {combined_label} {len(chunks_sorted)} {pieces_label}
{content_label}
{combined_content}
"""
            if parent_content:
                context_part += f"{original_context_label} {parent_content}\n"
            
            context_parts.append(context_part)
            source_counter += 1
        
        return "\n" + "="*50 + "\n".join(context_parts) + "="*50
    
    def _create_rag_prompt(self, query: str, context: str, language: str) -> str:
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
    
    def ask_question(self, query: str, 
                    retrieval_limit: int = 5, 
                    filter_by_type: List[str] = None,
                    model: str = None,
                    temperature: float = 0.1,
                    query_language: str = None) -> RAGResponse:
        """
        Complete multilingual RAG pipeline: Detect Language -> Retrieve -> Generate
        
        Args:
            query: User question
            retrieval_limit: Number of chunks to retrieve
            filter_by_type: Filter chunks by type
            model: LLM model to use
            temperature: Generation temperature
            query_language: Force specific language ('ar' or 'en'), None for auto-detect
            
        Returns:
            RAGResponse with answer and metadata
        """
        print(f"Processing question: {query}")
        
        # Detect query language if not provided
        if query_language is None:
            query_language = self.language_detector.detect_language(query)
            confidence = self.language_detector.get_language_confidence(query)
            print(f"Language detection: {query_language} (confidence: {confidence})")
        
        # Step 1: Retrieve relevant documents from appropriate collection
        print("1. Retrieving relevant documents...")
        retrieved_chunks = self.retrieve_documents(
            query=query,
            query_language=query_language,
            limit=retrieval_limit,
            filter_by_type=filter_by_type
        )
        
        if not retrieved_chunks:
            no_info_message = ("لم أجد معلومات ذات صلة في قاعدة البيانات للإجابة على سؤالك." 
                              if query_language == 'ar' 
                              else "I couldn't find relevant information in the database to answer your question.")
            
            return RAGResponse(
                answer=no_info_message,
                retrieved_chunks=[],
                query=query,
                query_language=query_language,
                confidence_score=0.0
            )
        
        print(f"Retrieved {len(retrieved_chunks)} relevant chunks from {query_language.upper()} collection")
        
        # Print retrieval info
        for i, chunk in enumerate(retrieved_chunks, 1):
            print(f"  {i}. {chunk['chunk_type']} (page {chunk['page_number']}) - "
                  f"score: {chunk['score']:.3f} - strategy: {chunk['chunk_strategy']} - "
                  f"tokens: {chunk.get('token_count', '?')}")
        
        # Step 2: Generate answer using language-specific prompting
        print("2. Generating answer...")
        answer = self.generate_answer(query, retrieved_chunks, query_language, model, temperature)
        
        # Calculate average confidence score
        avg_score = sum(chunk['score'] for chunk in retrieved_chunks) / len(retrieved_chunks)
        
        # Extract unique sources
        sources = sorted(set(f"{chunk['document_name']} (page {chunk['page_number']})" 
                            for chunk in retrieved_chunks))
        
        return RAGResponse(
            answer=answer,
            retrieved_chunks=retrieved_chunks,
            query=query,
            query_language=query_language,
            confidence_score=avg_score,
            sources=sources
        )
    
    def _generate_with_groq(self, prompt: str, lang: str, temperature: float) -> str:
        """Generate answer using Groq"""
        for attempt in range(5):
  
    
          try:
              messages = [
                  {"role": "user", "content": prompt}
              ]
              response = self.groq_client.chat(messages,lang, temperature=temperature)
              return response
          except openai.error.RateLimitError:
              time.sleep((2 ** attempt) + random.random())
    

    
    def analyze_multilingual_effectiveness(self, 
                                         arabic_queries: List[str] = None,
                                         english_queries: List[str] = None) -> Dict:
        """Analyze effectiveness of multilingual RAG system"""
        
        if arabic_queries is None:
            arabic_queries = [
                "ما هي الالتزامات الأساسية؟",
                "كيف يتم حماية البيانات الشخصية؟",
                "ما هي العقوبات المفروضة؟"
            ]
        
        if english_queries is None:
            english_queries = [
                "What are the main obligations?",
                "How is personal data protected?",
                "What are the imposed penalties?"
            ]
        
        print("🔍 Analyzing multilingual RAG effectiveness...")
        
        analysis_results = {
            "arabic_analysis": [],
            "english_analysis": [],
            "language_detection_accuracy": []
        }
        
        # Test Arabic queries
        print("\n📋 Testing Arabic queries...")
        for query in arabic_queries:
            print(f"  Testing: {query}")
            
            # Test language detection
            detected_lang = self.language_detector.detect_language(query)
            detection_correct = detected_lang == 'ar'
            analysis_results["language_detection_accuracy"].append({
                "query": query,
                "expected": "ar",
                "detected": detected_lang,
                "correct": detection_correct
            })
            
            # Test retrieval
            results = self.retrieve_documents(query, 'ar', limit=10)
            
            if results:
                analysis_results["arabic_analysis"].append({
                    "query": query,
                    "results_count": len(results),
                    "avg_score": sum(r['score'] for r in results) / len(results),
                    "languages_found": list(set(r.get('language', 'unknown') for r in results))
                })
        
        # Test English queries
        print("\n📋 Testing English queries...")
        for query in english_queries:
            print(f"  Testing: {query}")
            
            # Test language detection
            detected_lang = self.language_detector.detect_language(query)
            detection_correct = detected_lang == 'en'
            analysis_results["language_detection_accuracy"].append({
                "query": query,
                "expected": "en",
                "detected": detected_lang,
                "correct": detection_correct
            })
            
            # Test retrieval
            results = self.retrieve_documents(query, 'en', limit=10)
            
            if results:
                analysis_results["english_analysis"].append({
                    "query": query,
                    "results_count": len(results),
                    "avg_score": sum(r['score'] for r in results) / len(results),
                    "languages_found": list(set(r.get('language', 'unknown') for r in results))
                })
        
        # Print summary
        detection_accuracy = sum(1 for item in analysis_results["language_detection_accuracy"] 
                               if item["correct"]) / len(analysis_results["language_detection_accuracy"])
        
        print(f"\n📊 Multilingual RAG Analysis Summary:")
        print(f"   Language Detection Accuracy: {detection_accuracy:.2%}")
        print(f"   Arabic queries tested: {len(arabic_queries)}")
        print(f"   English queries tested: {len(english_queries)}")
        
        return analysis_results

def main():
    """Example usage with multilingual RAG system"""
    
    # Initialize multilingual RAG system
    rag_system = MultilingualRAGSystem(
        llama_api_key=LLAMA_CLOUD_API_KEY,
        qdrant_url=QDRANT_URL,
        qdrant_api_key=QDRANT_API_KEY,
        groq_api_key=GROQ_API_KEY,
        llm_provider="groq",
        chunk_target_size=400,
        chunk_max_size=600
    )
    
    print("="*60)
    print("MULTILINGUAL RAG SYSTEM - PROCESSING DOCUMENTS")
    print("="*60)
    
    # Process Arabic PDF
    arabic_pdf_path = "Documents/ksa-personal-data-protection-law-series-part-1-ar.pdf"
    print("\n📄 Processing Arabic PDF...")
    # arabic_chunks, arabic_lang = rag_system.process_pdf_to_qdrant_multilingual(arabic_pdf_path,3,"ar")
    
    # Process English PDF
    english_pdf_path = "Documents/saudi-economy-watch.pdf"
    print("\n📄 Processing English PDF...")
    # english_chunks, english_lang = rag_system.process_pdf_to_qdrant_multilingual(english_pdf_path,1,"eng")
    
    # Example multilingual questions
    test_questions = [
        "could you give me Contact list of Saudi Economy Watch?"
    ]
    
    print("\n" + "="*60)
    print("MULTILINGUAL RAG SYSTEM - Q&A TESTING")
    print("="*60)
    
    for question in test_questions[:2]:  # Test first 2 questions for demo
        print(f"\n🤔 Question: {question}")
        print("-" * 50)
        
        # Get answer using multilingual RAG
        response = rag_system.ask_question(
            query=question,
            retrieval_limit=20,
            temperature=0.1
        )
        
        print(f"🌐 Detected Language: {'Arabic' if response.query_language == 'ar' else 'English'}")
        print(f"📋 Answer:")
        print(response.answer)
        print(f"\n📊 Confidence Score: {response.confidence_score:.3f}")
        print(f"📚 Sources: {', '.join(response.sources)}")
        print(f"📄 Retrieved Chunks: {len(response.retrieved_chunks)}")
        
        # Show chunk details
        print("📝 Retrieved Chunk Details:")
        for i, chunk in enumerate(response.retrieved_chunks, 1):
            lang_label = "AR" if chunk.get('language') == 'ar' else "EN"
            print(f"   {i}. [{lang_label}] {chunk['chunk_type']} - {chunk['chunk_strategy']} - "
                  f"{chunk.get('token_count', 0)} tokens - score: {chunk['score']:.3f}")
        
        print("\n" + "="*60)
    
    # Run multilingual effectiveness analysis
    print("\n🔍 Running multilingual effectiveness analysis...")
    # analysis = rag_system.analyze_multilingual_effectiveness()

if __name__ == "__main__":
    main()