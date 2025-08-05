"""
Simple Groq RAG System - Compatible with different LlamaIndex versions
Save as simple_groq_rag.py
"""

import os
import requests
import json
from llama_index import StorageContext, load_index_from_storage, ServiceContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Configuration
VECTOR_STORE_DIR = "./vector_store"

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

class SimpleGroqRAG:
    """Simple RAG system using Groq for completion."""
    
    def __init__(self, groq_api_key: str, vector_store_dir: str = VECTOR_STORE_DIR):
        self.groq_chat = SimpleGroqChat(groq_api_key)
        self.vector_store_dir = vector_store_dir
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
        
        self._load_index()
    
    def _load_index(self):
        """Load the vector index."""
        try:
            embed_model = HuggingFaceEmbedding(model_name="intfloat/multilingual-e5-large")
            storage_context = StorageContext.from_defaults(persist_dir=self.vector_store_dir)
            service_context = ServiceContext.from_defaults(embed_model=embed_model, llm=None)
            
            self.index = load_index_from_storage(storage_context, service_context=service_context)
            self.retriever = self.index.as_retriever(similarity_top_k=5)
            
            print("✅ Index loaded successfully!")
            
        except Exception as e:
            print(f"❌ Error loading index: {e}")
            raise
    
    def retrieve_context(self, query: str, top_k: int = 5) -> tuple:
        """Retrieve relevant context from the index."""
        try:
            # Update retriever with new top_k if needed
            self.retriever = self.index.as_retriever(similarity_top_k=top_k)
            nodes = self.retriever.retrieve(query)
            
            # Extract text and metadata
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
        
        # Create prompt for RAG
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
    
    def query(self, question: str, top_k: int = 5) -> dict:
        """Complete RAG query with context retrieval and answer generation."""
        try:
            # Step 1: Retrieve relevant context
            print(f"🔍 Searching for relevant context...")
            context, sources = self.retrieve_context(question, top_k)
            
            if not context or "Error" in context:
                return {
                    "answer": "Could not retrieve relevant context from documents.",
                    "sources": [],
                    "model": self.groq_chat.model
                }
            
            # Step 2: Generate answer using Groq
            print(f"🦙 Generating answer with {self.groq_chat.model}...")
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
            print(f"✅ Switched to {self.groq_chat.model}")
        else:
            print(f"❌ Model '{model_key}' not available.")
            print(f"Available models: {list(self.available_models.keys())}")
    
    def test_api(self) -> bool:
        """Test if Groq API is working."""
        test_response = self.groq_chat.chat([
            {"role": "user", "content": "Hello! Please respond with 'API test successful'."}
        ])
        
        if "API test successful" in test_response:
            print("✅ Groq API test successful!")
            return True
        else:
            print(f"❌ API test failed: {test_response}")
            return False

def main():
    """Interactive RAG system."""
    print("🦙 Simple Groq RAG System")
    print("=" * 50)
    
    # Get API key
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        api_key = input("🔑 Enter your Groq API key: ").strip()
    
    if not api_key:
        print("❌ API key required!")
        return
    
    # Initialize system
    try:
        rag = SimpleGroqRAG(api_key)
        
        # Test API
        if not rag.test_api():
            print("❌ API test failed. Please check your API key.")
            return
            
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        return
    
    print(f"\n✅ System ready! Using model: {rag.groq_chat.model}")
    print("\nCommands:")
    print("- Ask any question")
    print("- 'models' - Show available models")
    print("- 'switch <model>' - Switch model")
    print("- 'test' - Test API")
    print("- 'quit' - Exit")
    print("-" * 50)
    
    while True:
        try:
            user_input = input(f"\n🔍 Query [{rag.groq_chat.model}]: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            elif user_input.lower() == 'models':
                print("\n🤖 Available models:")
                for key, model in rag.available_models.items():
                    current = " (current)" if model == rag.groq_chat.model else ""
                    print(f"  {key}: {model}{current}")
            
            elif user_input.lower().startswith('switch '):
                model_key = user_input.split(' ', 1)[1]
                rag.switch_model(model_key)
            
            elif user_input.lower() == 'test':
                rag.test_api()
            
            elif user_input:
                result = rag.query(user_input)
                
                print(f"\n🦙 Answer:")
                print(result['answer'])
                
                if result.get('sources'):
                    print(f"\n📚 Sources ({len(result['sources'])} found):")
                    for source in result['sources']:
                        metadata = source['metadata']
                        print(f"  📄 {metadata.get('file_name', 'Unknown')} "
                              f"(Page {metadata.get('page_number', '?')}) "
                              f"- Score: {source['score']}")
                
                print(f"\n💭 Context words used: {result.get('context_used', 'Unknown')}")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()