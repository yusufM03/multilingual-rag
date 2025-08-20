
from dataclasses import dataclass
from typing import List, Dict
@dataclass
class RAGResponse:
    """Response from RAG system"""
    answer: str
    retrieved_chunks: List[Dict]
    query: str
    query_language: str
    confidence_score: float = 0.0
    sources: List[str] = None
    llm_latency: float = 0.0
    retrieval_latency: float = 0.0