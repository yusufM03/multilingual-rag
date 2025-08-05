import json
import uuid
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import hashlib
import re

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

# Environment variables
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")
COLLECTION_NAME = "pdf_rag"

@dataclass
class DocumentChunk:
    """Represents a chunk of document content with metadata"""
    id: str
    content: str
    chunk_type: str  # title, paragraph, list_item, table_row, hybrid, etc.
    page_number: int
    document_name: str
    block_index: int
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
    confidence_score: float = 0.0
    sources: List[str] = None

class SimpleGroqChat:
    """Simple Groq API wrapper for chat completions."""
    
    def __init__(self, api_key: str, model: str = "llama3-70b-8192"):
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
                           base_metadata: Dict) -> List[Tuple[str, Dict]]:
        """Split large content into smaller chunks while preserving meaning"""
        tokens = self.count_tokens(content)
        
        if tokens <= self.max_chunk_size:
            return [(content, base_metadata)]
        
        # For Arabic text, split on sentence boundaries
        if chunk_type == "paragraph":
            return self._split_paragraph(content, base_metadata)
        elif chunk_type == "list":
            return self._split_list(content, base_metadata)
        else:
            # Fallback: split on word boundaries
            return self._split_on_words(content, base_metadata)
    
    def _split_paragraph(self, content: str, base_metadata: Dict) -> List[Tuple[str, Dict]]:
        """Split paragraph on sentence boundaries"""
        # Arabic sentence boundaries
        sentence_endings = r'[.!?؟。]'
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

class EnhancedRAGSystem:
    """Enhanced RAG system with hybrid chunking strategy"""
    
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
        """Initialize the enhanced RAG system with hybrid chunking"""
        
        self.extractor = LlamaExtract(api_key=llama_api_key)
        
        # Initialize Qdrant Cloud client
        self.qdrant_client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
        )
        
        # Initialize embedding model
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
        
        # Your schema (keeping existing extraction logic)
        self.schema = {
            "additionalProperties": False,
            "properties": {
                "page_content": {
                    "description": "List of content blocks for a page.",
                    "items": {
                        "additionalProperties": False,
                        "properties": {
                            "block_type": {
                                "enum": ["title", "paragraph", "list", "table", "columnar_text", "footnote"],
                                "type": "string",
                                "description": "Type of content block."
                            },
                            "content": {
                                "description": "Text content for title, paragraph, or footnote blocks.",
                                "type": "string"
                            },
                            "items": {
                                "description": "List items for list-type blocks.",
                                "items": {"type": "string"},
                                "type": "array"
                            },
                            "table_title": {
                                "description": "Optional title above the table.",
                                "type": "string"
                            },
                            "headers": {
                                "description": "Column headers for table blocks.",
                                "items": {"type": "string"},
                                "type": "array"
                            },
                            "rows": {
                                "description": "Each table row as an array of cell texts.",
                                "items": {"items": {"type": "string"}, "type": "array"},
                                "type": "array"
                            },
                            "columns": {
                                "description": "Used for multi-column text (like Arabic RTL).",
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
    
    def extract_pdf(self, pdf_path: str, agent_name: str = "w") -> List[Dict]:
        """Extract structured content from PDF using LlamaExtract"""
        try:
            agent = self.extractor.create_agent(
                name=agent_name,
                data_schema=self.schema,
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
            return result.data
        except Exception as e:
            print(f"Error extracting PDF {pdf_path}: {e}")
            return []
    
    def chunk_extracted_content_hybrid(self, extracted_data: List[Dict], 
                                      document_name: str) -> List[DocumentChunk]:
          """Convert extracted data into optimally sized chunks using hybrid strategy"""
          all_chunks = []
          
          for page_idx, page_data in enumerate(extracted_data):
              page_number = page_idx + 1
              page_content = page_data.get("page_content", [])
              
              # Process blocks with lookahead for intelligent merging
              chunks = self._process_page_blocks_hybrid(page_content, page_number, document_name)
              all_chunks.extend(chunks)
          
          print(f"Created {len(all_chunks)} hybrid chunks")
          self._print_chunk_statistics(all_chunks)
          
          return all_chunks
      
    def _process_page_blocks_hybrid(self, blocks: List[Dict], page_number: int, document_name: str) -> List[DocumentChunk]:
          chunks = []
          i = 0

          while i < len(blocks):
              current_block = blocks[i]
              current_content = self._extract_block_content(current_block)
              current_tokens = self.chunking_strategy.count_tokens(current_content)

              if current_block.get("block_type") == "table":
                  table_chunks = self._create_table_chunks(current_block, page_number, document_name, i)
                  chunks.extend(table_chunks)
                  print(f"[Page {page_number}] Table split into {len(table_chunks)} chunk(s).")
                  i += 1
                  continue

              merged_content = current_content
              merged_blocks = [current_block]
              j = i + 1

              while j < len(blocks):
                  next_block = blocks[j]
                  next_content = self._extract_block_content(next_block)
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
                  merged_content, merged_blocks, page_number, document_name, i
              )
              chunks.extend(merged_chunks)

              print(f"[Page {page_number}] Created {len(merged_chunks)} chunk(s) from merged blocks starting at index {i}.")
              i = j

          return chunks

      
    def _extract_block_content(self, block: Dict) -> str:
        """Extract text content from a block regardless of type"""
        block_type = block.get("block_type", "unknown")
        
        if block_type in ["title", "paragraph", "footnote"]:
            return block.get("content", "")
        
        elif block_type == "list":
            items = block.get("items", [])
            return "\n".join(f"• {item}" for item in items if item.strip())
        
        elif block_type == "columnar_text":
            columns = block.get("columns", [])
            return "\n".join(col.get("content", "") for col in columns)
        
        elif block_type == "table":
            return self._format_table_content(block)
        
        return ""
    
    def _format_table_content(self, block: Dict) -> str:
        """Format table as readable text"""
        table_title = block.get("table_title", "")
        headers = block.get("headers", [])
        rows = block.get("rows", [])
        
        content_parts = []
        
        if table_title.strip():
            content_parts.append(f"جدول: {table_title}")
        
        if headers:
            content_parts.append("الأعمدة: " + " | ".join(headers))
        
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
                            document_name: str, block_index: int) -> List[DocumentChunk]:
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
                                                page_number, document_name, block_index, len(chunks))
                chunks.append(chunk)
                print(f"[Table chunk created] split_index={len(chunks)-1}, rows={len(current_rows)}, tokens={chunk.token_count}")

                current_rows = []
                current_tokens = 0

            current_rows.append(row_text)
            current_tokens += row_tokens

        # Final remaining rows
        if current_rows:
            chunk = self._build_table_chunk(current_rows, headers, table_title,
                                            page_number, document_name, block_index, len(chunks))
            chunks.append(chunk)
            print(f"[Table chunk created] split_index={len(chunks)-1}, rows={len(current_rows)}, tokens={chunk.token_count}")

        # Validation: Ensure all rows are chunked
        total_chunked_rows = sum(len(c.metadata.get("row_texts", [])) if c.metadata else 0 for c in chunks)
        assert total_chunked_rows == len(row_texts), f"Row count mismatch: chunked={total_chunked_rows}, original={len(row_texts)}"

        # Update all chunks with total_splits information and store row_texts for validation
        total_splits = len(chunks)
        for idx, chunk in enumerate(chunks):
            if chunk.metadata is None:
                chunk.metadata = {}
            chunk.metadata['total_splits'] = total_splits
            chunk.metadata['split_index'] = idx
            chunk.metadata['row_texts'] = chunk.content.split('\n')[2:]  # exclude title and headers lines

        return chunks

    def _build_table_chunk(self, row_texts, headers, table_title, page_number, document_name, block_index, split_index):
        content_parts = []
        if table_title.strip():
            content_parts.append(f"جدول: {table_title}")
        if headers:
            content_parts.append("الأعمدة: " + " | ".join(headers))
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
            metadata={
                "headers": headers,
                "row_count": len(row_texts),
                "original_block_type": "table",
                "split_index": split_index,
                "row_texts": row_texts  # Add for verification and ordering downstream
            },
            token_count=tokens,
            chunk_strategy="split" if split_index > 0 else "intact"
        )

    
    def _create_chunks_from_merged_content(self, merged_content: str, 
                                         merged_blocks: List[Dict],
                                         page_number: int, document_name: str, 
                                         base_index: int) -> List[DocumentChunk]:
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
                metadata=base_metadata,
                token_count=tokens,
                chunk_strategy="merged" if len(merged_blocks) > 1 else "original"
            )]
        
        else:
            # Need to split the merged content
            chunk_type = self._determine_chunk_type(merged_blocks)
            split_chunks = self.chunking_strategy.split_large_content(
                merged_content, chunk_type, base_metadata
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
    
    # Keep all your existing methods for Qdrant operations, retrieval, and generation
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
    
    def process_pdf_to_qdrant_hybrid(self, pdf_path: str, collection_name: str, 
                                   document_name: str = None):
        """Complete pipeline with hybrid chunking: Extract PDF -> Hybrid Chunk -> Index"""
        if document_name is None:
            document_name = os.path.basename(pdf_path).replace('.pdf', '')
        
        print(f"Processing PDF with hybrid chunking: {pdf_path}")
        
        # Step 1: Extract structured content
        print("1. Extracting PDF content...")
        extracted_data = self.extract_pdf(pdf_path)
        if not extracted_data:
            print("Failed to extract PDF content")
            return
        
        # Step 2: Hybrid chunking
        print("2. Applying hybrid chunking strategy...")
        chunks = self.chunk_extracted_content_hybrid(extracted_data, document_name)
        
        # Step 3: Setup Qdrant collection
        print("3. Setting up Qdrant collection...")
        self.setup_qdrant_collection(collection_name)
        
        # Step 4: Index to Qdrant
        print("4. Indexing to Qdrant...")
        self.index_chunks_to_qdrant(chunks, collection_name)
        
        print("✅ Hybrid chunking pipeline completed successfully!")
        return chunks
    
    def retrieve_documents(self, query: str, collection_name: str, limit: int = 5, 
                          filter_by_type: List[str] = None) -> List[Dict]:
        """Retrieve relevant documents for a query"""
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
        
        results = self.qdrant_client.query_points(
            collection_name=collection_name,
            query=query_embedding.tolist(),
            limit=limit,
            query_filter=search_filter
        )
        
        formatted_results = []
        for result in results.points:
            formatted_results.append({
                "score": result.score,
                "content": result.payload["content"],
                "chunk_type": result.payload["chunk_type"],
                "page_number": result.payload["page_number"],
                "document_name": result.payload["document_name"],
                "parent_content": result.payload.get("parent_content"),
                "metadata": result.payload.get("metadata", {}),
                "chunk_id": result.payload.get("chunk_id"),
                "token_count": result.payload.get("token_count", 0),
                "chunk_strategy": result.payload.get("chunk_strategy", "unknown")
            })
        
        return formatted_results
    
    def generate_answer(self, query: str, retrieved_chunks: List[Dict], 
                       model: str = None, temperature: float = 0.1) -> str:
        """Generate answer using LLM based on retrieved context"""
        context = self._format_context(retrieved_chunks)
        prompt = self._create_rag_prompt(query, context)
        
        if self.llm_provider == "groq":
            return self._generate_with_groq(prompt, model or "llama3-8b-8192", temperature)
        elif self.llm_provider == "openai":
            return self._generate_with_openai(prompt, model or "gpt-3.5-turbo", temperature)
        else:
            return "Error: No valid LLM provider configured."
    
    def _format_context(self, retrieved_chunks: List[Dict]) -> str:
        """Format retrieved chunks into context string with merged split handling"""
        from collections import defaultdict
        
        # Group chunks by base block_index (e.g., "3" from "3", "3_0", "3_1")
        grouped_chunks = defaultdict(list)
        for chunk in retrieved_chunks:
            full_block_index = str(chunk.get('block_index', ''))
            base_index = full_block_index.split('_')[0] if '_' in full_block_index else full_block_index
            grouped_chunks[base_index].append(chunk)
        
        context_parts = []
        source_counter = 1
        
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
    مصدر {source_counter}:
    نوع المحتوى: {chunk_type}
    الصفحة: {page_number}
    الوثيقة: {document_name}
    عدد الرموز: {token_count}
    استراتيجية التقسيم: {chunk_strategy}
    جزء 1 من {total_splits} (مجمّع من {len(chunks_sorted)} قطع)
    المحتوى:
    {combined_content}
    """
            if parent_content:
                context_part += f"السياق الأصلي: {parent_content}\n"
            
            context_parts.append(context_part)
            source_counter += 1
        
        return "\n" + "="*50 + "\n".join(context_parts) + "="*50


    def ask_question(self, query: str, collection_name: str, 
                    retrieval_limit: int = 5, 
                    filter_by_type: List[str] = None,
                    model: str = None,
                    temperature: float = 0.1) -> RAGResponse:
        """
        Complete RAG pipeline with hybrid chunking: Retrieve + Generate
        
        Args:
            query: User question
            collection_name: Qdrant collection name
            retrieval_limit: Number of chunks to retrieve
            filter_by_type: Filter chunks by type
            model: LLM model to use
            temperature: Generation temperature
            
        Returns:
            RAGResponse with answer and metadata
        """
        print(f"Processing question: {query}")
        
        # Step 1: Retrieve relevant documents
        print("1. Retrieving relevant documents...")
        retrieved_chunks = self.retrieve_documents(
            query=query,
            collection_name=collection_name,
            limit=retrieval_limit,
            filter_by_type=filter_by_type
        )
        
        if not retrieved_chunks:
            return RAGResponse(
                answer="لم أجد معلومات ذات صلة في قاعدة البيانات للإجابة على سؤالك.",
                retrieved_chunks=[],
                query=query,
                confidence_score=0.0
            )
        
        print(f"Retrieved {len(retrieved_chunks)} relevant chunks")
        
        # Print retrieval info
        for i, chunk in enumerate(retrieved_chunks, 1):
            print(f"  {i}. {chunk['chunk_type']} (صفحة {chunk['page_number']}) - "
                  f"نقاط: {chunk['score']:.3f} - استراتيجية: {chunk['chunk_strategy']} - "
                  f"رموز: {chunk.get('token_count', '?')}")
        
        # Step 2: Format context by grouping chunks for completeness
        context = self._format_context(retrieved_chunks)
        
        # Step 3: Create prompt and generate answer
        prompt = self._create_rag_prompt(query, context)
        print("2. Generating answer...")
        if self.llm_provider == "groq":
            answer = self._generate_with_groq(prompt, model or "llama3-8b-8192", temperature)
        elif self.llm_provider == "openai":
            answer = self._generate_with_openai(prompt, model or "gpt-3.5-turbo", temperature)
        else:
            answer = "Error: No valid LLM provider configured."
        
        # Calculate average confidence score
        avg_score = sum(chunk['score'] for chunk in retrieved_chunks) / len(retrieved_chunks)
        
        # Extract unique sources in a neat format
        sources = sorted(set(f"{chunk['document_name']} (صفحة {chunk['page_number']})" 
                            for chunk in retrieved_chunks))
        
        return RAGResponse(
            answer=answer,
            retrieved_chunks=retrieved_chunks,
            query=query,
            confidence_score=avg_score,
            sources=sources
        )

    
    def _create_rag_prompt(self, query: str, context: str) -> str:
        """Create RAG prompt for answer generation"""
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
        
        return prompt
    
    def _generate_with_groq(self, prompt: str, model: str, temperature: float) -> str:
        """Generate answer using Groq"""
        try:
            messages = [
                {"role": "user", "content": prompt}
            ]
            response = self.groq_client.chat(messages, temperature=temperature)
            return response
        except Exception as e:
            return f"خطأ في توليد الإجابة باستخدام Groq: {e}"
    
    def _generate_with_openai(self, prompt: str, model: str, temperature: float) -> str:
        """Generate answer using OpenAI"""
        try:
            response = self.openai_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"خطأ في توليد الإجابة باستخدام OpenAI: {e}"
    
    
    def analyze_chunking_effectiveness(self, collection_name: str, 
                                     sample_queries: List[str] = None) -> Dict:
        """Analyze the effectiveness of the hybrid chunking strategy"""
        
        if sample_queries is None:
            sample_queries = [
                "ما هي الالتزامات الأساسية؟",
                "كيف يتم حماية البيانات الشخصية؟",
                "ما هي العقوبات المفروضة؟"
            ]
        
        print("🔍 Analyzing chunking effectiveness...")
        
        # Get collection info
        try:
            collection_info = self.qdrant_client.get_collection(collection_name)
            total_points = collection_info.points_count
        except:
            total_points = "Unknown"
        
        analysis_results = {
            "collection_stats": {
                "total_chunks": total_points,
                "collection_name": collection_name
            },
            "query_analysis": []
        }
        
        for query in sample_queries:
            print(f"  Testing query: {query}")
            
            # Retrieve for this query
            results = self.retrieve_documents(query, collection_name, limit=10)
            
            if results:
                chunk_strategies = {}
                chunk_types = {}
                token_ranges = []
                scores = [r['score'] for r in results]
                
                for result in results:
                    strategy = result.get('chunk_strategy', 'unknown')
                    chunk_strategies[strategy] = chunk_strategies.get(strategy, 0) + 1
                    
                    chunk_type = result.get('chunk_type', 'unknown')
                    chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
                    
                    tokens = result.get('token_count', 0)
                    if tokens > 0:
                        token_ranges.append(tokens)
                
                query_analysis = {
                    "query": query,
                    "results_count": len(results),
                    "avg_score": sum(scores) / len(scores),
                    "score_range": [min(scores), max(scores)],
                    "chunk_strategies": chunk_strategies,
                    "chunk_types": chunk_types,
                    "token_stats": {
                        "min": min(token_ranges) if token_ranges else 0,
                        "max": max(token_ranges) if token_ranges else 0,
                        "avg": sum(token_ranges) / len(token_ranges) if token_ranges else 0
                    }
                }
                
                analysis_results["query_analysis"].append(query_analysis)
        
        # Print summary
        print("\n📊 Chunking Effectiveness Analysis:")
        print(f"   Collection: {collection_name} ({total_points} chunks)")
        
        for qa in analysis_results["query_analysis"]:
            print(f"\n   Query: {qa['query']}")
            print(f"   Results: {qa['results_count']}, Avg Score: {qa['avg_score']:.3f}")
            print(f"   Strategies: {qa['chunk_strategies']}")
            print(f"   Types: {qa['chunk_types']}")
            print(f"   Token range: {qa['token_stats']['min']}-{qa['token_stats']['max']} "
                  f"(avg: {qa['token_stats']['avg']:.0f})")
        
        return analysis_results

# Example usage and comparison
def main():
    """Example usage with hybrid chunking"""
    
    # Initialize enhanced RAG system
    rag_system = EnhancedRAGSystem(
        llama_api_key=LLAMA_CLOUD_API_KEY,
        qdrant_url=QDRANT_URL,
        qdrant_api_key=QDRANT_API_KEY,
        groq_api_key=GROQ_API_KEY,
        llm_provider="groq",
        chunk_target_size=400,  # Target 400 tokens per chunk
        chunk_max_size=600      # Maximum 600 tokens per chunk
    )
    
    # Process PDF with hybrid chunking
    pdf_path = "Documents/ksa-personal-data-protection-law-series-part-1-ar.pdf"
    collection_name = "arabic_legal_docs_hybrid"
    
    print("="*60)
    print("PROCESSING PDF WITH HYBRID CHUNKING")
    print("="*60)
    
    # Uncomment to process PDF with hybrid chunking:
    chunks = rag_system.process_pdf_to_qdrant_hybrid(pdf_path, collection_name)
    
    # Analyze chunking effectiveness
    # analysis = rag_system.analyze_chunking_effectiveness(collection_name)
    
    # Example questions for testing
    questions = [
        "ما هي مبادئ مبادئ معالجة البیانات الشخصیة"
    ]
    
    print("\n" + "="*60)
    print("RAG SYSTEM WITH HYBRID CHUNKING - Q&A")
    print("="*60)
    
    for question in questions:
        print(f"\n🤔 السؤال: {question}")
        print("-" * 50)
        
        # Get answer using hybrid RAG
        response = rag_system.ask_question(
            query=question,
            collection_name=collection_name,
            retrieval_limit=15,
            temperature=0.1
        )
        
        print(f"📋 الإجابة:")
        print(response.answer)
        print(f"\n📊 نقاط الثقة: {response.confidence_score:.3f}")
        print(f"📚 المصادر: {', '.join(response.sources)}")
        print(f"📄 عدد المقاطع المسترجعة: {len(response.retrieved_chunks)}")
        
        # Show chunk details
        print("📝 تفاصيل المقاطع المسترجعة:")
        for i, chunk in enumerate(response.retrieved_chunks, 1):
            print(f"   {i}. {chunk['chunk_type']} - {chunk['chunk_strategy']} - "
                  f"{chunk.get('token_count', 0)} رمز - نقاط: {chunk['score']:.3f}")
        
        print("\n" + "="*60)


if __name__ == "__main__":
    main()