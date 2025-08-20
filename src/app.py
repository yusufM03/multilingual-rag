
import streamlit as st
import pandas as pd
import plotly.express as px
import os
import sys
import tempfile
import json
from datetime import datetime
import time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.store_logs import store_logs_ToCosmosDB, update_feedback
from utils.language_detector import LanguageDetector
import uuid

from core import MultilingualRAGWithLlamaIndex
from config.settings import configs
from qdrant_client.http import models as qmodels





st.set_page_config(
    page_title="Multilingual RAG System - LlamaIndex",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with better structure
st.markdown("""
<style>
/* Base styles */
.stApp {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
}

/* Dark mode styles */
@media (prefers-color-scheme: dark) {
    .stApp {
        background-color: #0e1117 !important;
        color: white !important;
    }
    section[data-testid="stSidebar"] {
        background-color: #1c1f26 !important;
    }
    .upload-section, .qa-section, .answer-box, .metric-card {
        background-color: #1c1f26 !important;
        color: white !important;
        border: 1px solid #333 !important;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 6px;
    }
}

/* Light mode styles */
@media (prefers-color-scheme: light) {
    .stApp {
        background-color: white !important;
        color: black !important;
    }
    section[data-testid="stSidebar"] {
        background-color: #f8f9fa !important;
    }
    .upload-section, .qa-section, .answer-box, .metric-card {
        background-color: #ffffff !important;
        color: black !important;
        border: 1px solid #e9ecef !important;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 6px;
    }
}

/* RTL support for Arabic text */
.rtl-text {
    direction: rtl;
    text-align: right;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

/* Source box styling */
.source-box {
    background-color: rgba(128, 128, 128, 0.1);
    padding: 8px;
    margin: 5px 0;
    border-radius: 4px;
    border-left: 3px solid #667eea;
}

/* Main header styling */
.main-header {
    text-align: center;
    padding: 1rem 0;
    margin-bottom: 2rem;
}

.main-header h1 {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-size: 2.5rem;
    margin: 0;
}

/* Error and success message styling */
.stAlert {
    border-radius: 8px;
    margin: 10px 0;
}
</style>
""", unsafe_allow_html=True)


# Initialize session state
def initialize_session_state():
    """Initialize all session state variables"""
    session_vars = {
        'rag_system': None,
        'processed_documents': [],
        'qa_history': [],
        'system_initialized': False,
        'collection_stats': {'doc': 0},
        'current_session_id': None
    }
    
    for var, default_value in session_vars.items():
        if var not in st.session_state:
            st.session_state[var] = default_value


def feedback_ui(session_id):
    """Display feedback UI with proper session handling"""
    if not session_id:
        st.warning("Session ID not available for feedback")
        return
        
    st.write("Was this answer helpful?")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("👍 Like", key=f"like_{session_id}"):
            try:
                success = update_feedback(session_id, "like")
                if success:
                    st.success("Thanks for your positive feedback!")
                else:
                    st.error("Failed to update feedback. Please try again.")
            except Exception as e:
                st.error(f"Error updating feedback: {str(e)}")
                
    with col2:
        if st.button("👎 Dislike", key=f"dislike_{session_id}"):
            try:
                success = update_feedback(session_id, "dislike")
                if success:
                    st.success("Thanks for your feedback! We'll work on improving.")
                else:
                    st.error("Failed to update feedback. Please try again.")
            except Exception as e:
                st.error(f"Error updating feedback: {str(e)}")


def validate_api_keys():
    """Validate required API keys"""
    required_keys = {
        'qdrant_url': 'Qdrant Cloud URL',
        'qdrant_api_key': 'Qdrant API Key',
        'groq_api_key': 'Groq API Key',
        'groq_api_key_1': 'Groq API Key (Secondary)',
        'llama_cloud_api_key': 'LlamaCloud API Key'
    }
    
    missing_keys = []
    for key, description in required_keys.items():
        if key not in st.secrets or not st.secrets[key]:
            missing_keys.append(f"{description} ({key})")
    
    return missing_keys


def initialize_rag_system():
    """Initialize the LlamaIndex RAG system with error handling"""
    if st.session_state.system_initialized and st.session_state.rag_system:
        return st.session_state.rag_system
        
    try:
        # Check if all required secrets are set
        missing_keys = validate_api_keys()
        if missing_keys:
            st.error(f"Missing required secrets: {', '.join(missing_keys)}")
            st.info("Please set these secrets in your Streamlit secrets.toml file.")
            return None
        
        # Initialize the LlamaIndex RAG system
        rag_system = MultilingualRAGWithLlamaIndex(
            llama_api_key=configs.LLAMA_CLOUD_API_KEY,
            qdrant_url= configs.QDRANT_URL,
            qdrant_api_key= configs.QDRANT_API_KEY,
            groq_api_key= configs.GROQ_API_KEY,
            groq_api_key_1= configs.GROQ_API_KEY_1,
            embedding_model= configs.EMBEDDING_MODEL,
            chunk_size= configs.CHUNK_SIZE,
            chunk_overlap= configs.CHUNK_OVERLAP 
        ) 
        
        st.session_state.rag_system = rag_system
        st.session_state.system_initialized = True
        st.success("✅ LlamaIndex RAG System initialized successfully!")
        return rag_system
        
    except Exception as e:
        st.error(f"❌ Failed to initialize RAG system: {str(e)}")
        st.session_state.system_initialized = False
        return None


def display_header():
    """Display the main header"""
    st.markdown("""
    <div class="main-header">
        <h1>🌐 RAG System - LlamaIndex</h1>
        <p>Multilingual Document Q&A System supporting Arabic and English</p>
    </div>
    """, unsafe_allow_html=True)


def display_system_status():
    """Display system status in sidebar"""
    st.sidebar.markdown("## 🚀 System Status")
    
    # System initialization status
    if st.session_state.system_initialized and st.session_state.rag_system:
        st.sidebar.success("✅ System Ready")
        
        # Display collection statistics
        st.sidebar.markdown("### 📊 Document Statistics")
        st.sidebar.metric("Documents Uploaded", st.session_state.collection_stats['doc'])
        
        # Display Q&A statistics
        qa_count = len(st.session_state.qa_history)
        st.sidebar.metric("Questions Asked", qa_count)
        
        # System info
        st.sidebar.markdown("### ℹ️ System Info")
        st.sidebar.info("Framework: LlamaIndex")
        st.sidebar.info("Vector Store: Qdrant")
        st.sidebar.info("LLM: Groq Llama3-70B")
        
    else:
        st.sidebar.error("❌ System Not Ready")
        if st.sidebar.button("🔄 Retry Initialization"):
            st.session_state.system_initialized = False
            st.session_state.rag_system = None
            st.rerun()


def display_document_upload():
    """Display document upload section"""
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    st.markdown("## 📄 Document Upload & Processing")
    
    # Check if system is ready
    if not st.session_state.system_initialized or not st.session_state.rag_system:
        st.warning("⚠️ System not initialized. Please check API keys and try again.")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_files = st.file_uploader(
            "Upload PDF documents",
            type=['pdf'],
            accept_multiple_files=True,  # Allow multiple files
            help="Upload one or more PDF files (Arabic or English)."
        )
    
    with col2:
        chunk_size = st.slider(
            "Chunk Size (tokens)",
            min_value=256,
            max_value=1024,
            value=512,
            step=64,
            help="Adjust the size for text chunks in LlamaIndex"
        )
        
        chunk_overlap = st.slider(
            "Chunk Overlap (tokens)",
            min_value=0,
            max_value=200,
            value=50,
            step=10,
            help="Overlap between consecutive chunks"
        )
        
        # Collection mode (simplified)
        collection_mode = "Append to existing"
    
    # Process documents button
    if uploaded_files and st.button("🚀 Process Documents with LlamaIndex", type="primary"):
        if not isinstance(uploaded_files, list):
            uploaded_files = [uploaded_files]
        process_documents_llamaindex(uploaded_files, chunk_size, chunk_overlap, collection_mode)
    
    # Display processed documents
    if st.session_state.processed_documents:
        st.markdown("### 📚 Processed Documents")
        display_processed_documents_table()
    
    st.markdown('</div>', unsafe_allow_html=True)


def display_processed_documents_table():
    """Display processed documents in a formatted table"""
    try:
        df_data = []
        for doc in st.session_state.processed_documents:
            df_data.append({
                "Document": doc["name"],
                "Language": "Arabic" if doc["language"] == "ar" else "English",
                "LlamaIndex Documents": doc.get("documents", "N/A"),
                "Collection": doc["collection"],
                "Processing Time": f"{doc['processing_time']:.2f}s",
                "Status": "✅ Indexed",
                "Mode": doc.get("mode", "Unknown")
            })
        
        if df_data:
            df = pd.DataFrame(df_data)
            st.dataframe(df, use_container_width=True)
    except Exception as e:
        st.error(f"Error displaying documents table: {str(e)}")


def process_documents_llamaindex(uploaded_files, chunk_size, chunk_overlap, collection_mode):
    """Process uploaded documents using LlamaIndex pipeline"""
    if not st.session_state.rag_system:
        st.error("RAG system not initialized")
        return
        
    try:
        # Update chunking parameters
        st.session_state.rag_system.node_parser.chunk_size = chunk_size
        st.session_state.rag_system.node_parser.chunk_overlap = chunk_overlap
    except AttributeError:
        st.warning("Could not update chunk parameters. Using defaults.")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    processed_count = 0
    
    for i, uploaded_file in enumerate(uploaded_files):
        try:
            # Update progress
            progress = (i + 1) / len(uploaded_files)
            progress_bar.progress(progress)
            status_text.text(f"Processing {uploaded_file.name} with LlamaIndex...")
            
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            start_time = time.time()
            document_name = uploaded_file.name.replace('.pdf', '')
            
            # Process document
            index, detected_language = process_pdf_append_mode(
                st.session_state.rag_system,
                tmp_file_path,
                "auto",  # Auto-detect language
                document_name
            )
            
            processing_time = time.time() - start_time
            
            if index and detected_language:
                # Determine collection name
                collection_name = "arabic_docs_llamaindex" if detected_language == 'ar' else "english_docs_llamaindex"
                
                # Count documents in index
                doc_count = len(index.docstore.docs) if hasattr(index, 'docstore') and hasattr(index.docstore, 'docs') else "N/A"
                
                # Add to processed documents
                doc_info = {
                    "name": uploaded_file.name,
                    "language": detected_language,
                    "documents": doc_count,
                    "collection": collection_name,
                    "processing_time": processing_time,
                    "processed_at": datetime.now().isoformat(),
                    "mode": collection_mode
                }
                st.session_state.processed_documents.append(doc_info)
                
                # Update collection stats
                st.session_state.collection_stats['doc'] += 1
                processed_count += 1
                
                st.success(f"✅ Processed {uploaded_file.name} "
                          f"({'Arabic' if detected_language == 'ar' else 'English'}) "
                          f"- Added to {collection_name}")
            else:
                st.error(f"❌ Failed to process {uploaded_file.name}")
            
            # Clean up temporary file
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
            
        except Exception as e:
            st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
            continue
    
    # Final progress update
    progress_bar.progress(1.0)
    status_text.text(f"✅ Processing complete! {processed_count}/{len(uploaded_files)} documents processed successfully.")
    
    # Clear progress indicators after a short delay
    time.sleep(2)
    progress_bar.empty()
    status_text.empty()

def extract_chunck(rag_system,pdf_path,document_name,detected_language):
              
# Extract PDF content
    extracted_data, detected_language = rag_system.extract_pdf(pdf_path, detected_language)
    
    if not extracted_data:
        return None, None
    print('extracting done')
    # Convert to LlamaIndex Documents
    documents = rag_system.convert_to_llamaindex_documents(
        extracted_data, document_name, detected_language
    )

    
    
    if not documents:
        return None, None
    # # chunck documents 
    # documents=rag_system.chunk_documents(
    #     documents
    # )

    
    

    # save_dir = f"chunked_files/{document_name}"
    # os.makedirs(save_dir, exist_ok=True)

    # for i, doc in enumerate(documents):
    #     chunk_filename = os.path.join(save_dir, f"chunk_{i+1}.txt")
    #     with open(chunk_filename, "w", encoding="utf-8") as f:
    #         f.write(doc.text)

    # print(f"Saved {len(documents)} chunks to {save_dir}")
    return documents
  



def process_pdf_append_mode(rag_system, pdf_path, language, document_name):
    """Process PDF in append mode without clearing existing collection"""
    try:
        # Detect language
        print("detecting language")
        detected_language = rag_system.detect_language_pdf(pdf_path)
        print(detected_language)
        
        collection_name = "arabic_docs_llamaindex" if detected_language == 'ar' else "english_docs_llamaindex"
        existing_index = rag_system.load_existing_index(collection_name, detected_language)
        if existing_index:
            # Scroll through Qdrant manually to check file_name
            offset = 0
            batch_size = 100
            duplicate_found = False

            while True:
                # Unpack points correctly
                points, _ = rag_system.qdrant_client.scroll(
                    collection_name=collection_name,
                    limit=batch_size,
                    offset=offset
                )

                if not points:
                    break

                for point in points:
                    if point.payload.get("file_name") == document_name:
                        duplicate_found = True
                        break
                if duplicate_found:
                    break

                offset += batch_size
            
            if duplicate_found:
                print(f"The file '{document_name}' has already been loaded. Skipping insertion.")
                return existing_index, detected_language
            else:
                # Extract chunks from PDF
                documents = extract_chunck(rag_system, pdf_path, document_name, detected_language)
                if not documents:
                    print("No documents extracted")
                    return None, detected_language
                
                # Add file_name to metadata
                for doc in documents:
                    if not doc.metadata:
                        doc.metadata = {}
                    doc.metadata["file_name"] = document_name
                
                
                # Insert all chunks
                for doc in documents:
                    existing_index.insert(doc)
                print(f"Inserted {len(documents)} chunks into existing collection.")
                return existing_index, detected_language
            
        
        
        
        
        
        # If no existing index, create new
        else:
            index = rag_system.create_index(documents, collection_name, detected_language)
            print(f"Created new index with {len(documents)} chunks.")
            return index, detected_language

    except Exception as e:
        st.error(f"Error in PDF processing: {str(e)}")
        return None, None



def display_qa_interface():
    """Display Q&A interface"""
    st.markdown('<div class="qa-section">', unsafe_allow_html=True)
    st.markdown("## 🤔 Ask Questions - LlamaIndex RAG")
    
    if not st.session_state.system_initialized or not st.session_state.rag_system:
        st.warning("⚠️ System not ready. Please initialize the system first.")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    if not st.session_state.processed_documents:
        st.info("📚 Please upload and process some documents first.")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    # Question input
    col1, col2 = st.columns([3, 1])
    
    with col1:
        question = st.text_area(
            "Enter your question (Arabic or English):",
            height=100,
            placeholder="مثال: ما هي مبادئ معالجة البيانات الشخصية؟\nExample: What are the main data protection principles?",
            key="question_input"
        )
    
    with col2:
        st.markdown("### ⚙️ LlamaIndex Settings")
        
        similarity_top_k = st.slider(
            "Number of sources (top_k)",
            min_value=3,
            max_value=20,
            value=8,
            help="How many relevant chunks to retrieve from vector index"
        )
    
    # Submit button
    if st.button("🔍 Get Answer with LlamaIndex", type="primary", disabled=not question.strip()):
        get_answer_llamaindex(question, similarity_top_k)
    
    # Display Q&A history
    display_qa_history()
    
    st.markdown('</div>', unsafe_allow_html=True)


def get_answer_llamaindex(question, similarity_top_k):
    """Get answer using LlamaIndex RAG system and log to Cosmos DB."""
    with st.spinner("🤖 LlamaIndex is thinking... Please wait"):
        try:
            start_time = time.time()
            
            # Get response from RAG system
            
            response = st.session_state.rag_system.ask_question(
                    query=question,
                    similarity_top_k=similarity_top_k
                )
          

            processing_time_ms = (time.time() - start_time) * 1000
            session_id = str(uuid.uuid4())

            # Session history entry for UI
            qa_entry_ui = {
                "question": question,
                "answer": response.answer,
                "language": getattr(response, 'query_language', 'unknown'),
                "confidence": getattr(response, 'confidence_score', 0.0),
                "sources": getattr(response, 'sources', []),
                "retrieved_chunks": getattr(response, 'retrieved_chunks', []),
                "processing_time": processing_time_ms / 1000,
                "timestamp": datetime.now().isoformat(),
                "top_k": similarity_top_k,
                "session_id": session_id
            }
            
            # Add to history
            st.session_state.qa_history.insert(0, qa_entry_ui)
            st.session_state.current_session_id = session_id

            # Log to Cosmos DB
            try:
                qa_log_entry = {
                    "question": question,
                    "language": getattr(response, 'query_language', 'unknown'),
                    "timestamp": datetime.utcnow().isoformat(),
                    "retrieved_chunks": getattr(response, 'retrieved_chunks', []),
                    "similarity_scores": [chunk.get('score', None) for chunk in getattr(response, 'retrieved_chunks', [])],
                    "sources": getattr(response, 'sources', []),
                    "top_k": similarity_top_k,
                    "retrieval_latency_ms": getattr(response, 'retrieval_latency', 0),
                    "answer": response.answer,
                    "llm_model": "Groq-Llama-3-70B",
                    "generation_latency_ms": getattr(response, 'llm_latency', 0),
                    "confidence": getattr(response, 'confidence_score', 0.0),
                    "feedback": None,
                    "processing_time_ms": round(processing_time_ms, 2),
                    "session_id": session_id,
                    "app_version": "v1.0.0",
                    "embedding_model_version": "text-embedding-multilingual-v2",
                    "prompt_version": "rag_template_v3"
                }
                # Store to db 
                store_logs_ToCosmosDB(qa_log_entry)


            except Exception as log_error:
                st.warning(f"Logging failed: {str(log_error)}")

            # Display answer
            display_answer_llamaindex(qa_entry_ui)
            
            # Show feedback UI
            feedback_ui(session_id)

        except Exception as e:
            st.error(f"❌ Error getting answer from LlamaIndex: {str(e)}")


def display_answer_llamaindex(qa_entry):
    """Display a Q&A entry with LlamaIndex-specific formatting"""
    language = qa_entry.get("language", "unknown")
    is_arabic = language == "ar"
    text_class = "rtl-text" if is_arabic else ""
    
    st.markdown("---")
    
    # Question
    st.markdown(f"### 🤔 Question ({'Arabic' if is_arabic else 'English'})")
    st.markdown(f'<div class="{text_class}">{qa_entry["question"]}</div>', unsafe_allow_html=True)
    
    # Answer
    st.markdown("### 💡 Answer")
    st.markdown(
        f"""
        <div class="answer-box {text_class}">
            {qa_entry['answer']}
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Confidence", f"{qa_entry.get('confidence', 0):.3f}")
    with col2:
        st.metric("Sources", len(qa_entry.get('sources', [])))
    with col3:
        st.metric("Retrieved Chunks", len(qa_entry.get('retrieved_chunks', [])))
    with col4:
        st.metric("Response Time", f"{qa_entry.get('processing_time', 0):.2f}s")
    with col5:
        method_short = qa_entry.get('method', 'Standard')[:8] + ("..." if len(qa_entry.get('method', '')) > 8 else "")
        st.metric("Method", method_short)
    
    # Sources
    sources = qa_entry.get('sources', [])
    if sources:
        st.markdown("### 📚 Sources")
        for i, source in enumerate(sources, 1):
            st.markdown(f'<div class="source-box">📄 {i}. {source}</div>', unsafe_allow_html=True)


def display_qa_history():
    """Display Q&A history"""
    if st.session_state.qa_history:
        st.markdown("### 📈 Recent Questions & Answers")
        
        # Add clear history button
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("🗑️ Clear History"):
                st.session_state.qa_history = []
                st.rerun()
        
        # Show recent entries (limit to 3 for performance)
        for idx, qa_entry in enumerate(st.session_state.qa_history[:3]):
            question_preview = qa_entry['question'][:50] + "..." if len(qa_entry['question']) > 50 else qa_entry['question']
            with st.expander(f"Q: {question_preview} (Method: {qa_entry.get('method', 'Standard')})"):
                display_answer_llamaindex(qa_entry)


def display_analytics():
    """Display analytics and statistics"""
    st.markdown("## 📊 LlamaIndex System Analytics")
    
    if not st.session_state.processed_documents and not st.session_state.qa_history:
        st.info("📈 Analytics will appear here after processing documents and asking questions.")
        return
    
    # Document statistics
    if st.session_state.processed_documents:
        st.markdown("### 📚 Document Processing Statistics")
        
        try:
            doc_df = pd.DataFrame(st.session_state.processed_documents)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Language distribution
                lang_counts = doc_df['language'].value_counts()
                lang_counts.index = ['Arabic' if x == 'ar' else 'English' for x in lang_counts.index]
                
                fig_lang = px.pie(
                    values=lang_counts.values,
                    names=lang_counts.index,
                    title="Documents by Language"
                )
                st.plotly_chart(fig_lang, use_container_width=True)
            
            with col2:
                # Processing time chart
                fig_bar = px.bar(
                    doc_df,
                    x='name',
                    y='processing_time',
                    color='language',
                    title="Processing Time by Document",
                    labels={'processing_time': 'Time (seconds)', 'name': 'Document'}
                )
                fig_bar.update_layout(xaxis_tickangle=45)
                st.plotly_chart(fig_bar, use_container_width=True)
                
        except Exception as e:
            st.error(f"Error displaying document analytics: {str(e)}")
    
    # Q&A statistics
    if st.session_state.qa_history:
        st.markdown("### 🤔 Question & Answer Statistics")
        
        try:
            qa_df = pd.DataFrame(st.session_state.qa_history)
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                avg_confidence = qa_df['confidence'].mean() if 'confidence' in qa_df.columns else 0
                st.metric("Average Confidence", f"{avg_confidence:.3f}")
            
            with col2:
                avg_response_time = qa_df['processing_time'].mean() if 'processing_time' in qa_df.columns else 0
                st.metric("Avg Response Time", f"{avg_response_time:.2f}s")
            
            with col3:
                total_questions = len(qa_df)
                st.metric("Total Questions", total_questions)
            
            with col4:
                if 'retrieved_chunks' in qa_df.columns:
                    avg_chunks = qa_df['retrieved_chunks'].apply(lambda x: len(x) if isinstance(x, list) else 0).mean()
                    st.metric("Avg Retrieved Chunks", f"{avg_chunks:.1f}")
                else:
                    st.metric("Avg Retrieved Chunks", "N/A")
            
            # Additional charts
            col1, col2 = st.columns(2)
            
            with col1:
                # Confidence score distribution
                if 'confidence' in qa_df.columns:
                    fig_conf = px.histogram(
                        qa_df,
                        x='confidence',
                        title="Confidence Score Distribution",
                        nbins=10
                    )
                    st.plotly_chart(fig_conf, use_container_width=True)
            
            with col2:
                # Language distribution in questions
                if 'language' in qa_df.columns:
                    lang_counts = qa_df['language'].value_counts()
                    lang_counts.index = ['Arabic' if x == 'ar' else 'English' if x == 'en' else 'Unknown' for x in lang_counts.index]
                    
                    fig_lang_qa = px.bar(
                        x=lang_counts.index,
                        y=lang_counts.values,
                        title="Questions by Language"
                    )
                    st.plotly_chart(fig_lang_qa, use_container_width=True)
                    
        except Exception as e:
            st.error(f"Error displaying Q&A analytics: {str(e)}")


def display_settings():
    """Display system settings and configuration"""
    st.markdown("## ⚙️ System Settings & Configuration")
    
    # API Key status
    st.markdown("### 🔑 API Configuration")
    
    try:
        api_keys = {
            'Qdrant URL': st.secrets.get("qdrant_url", "Not set"),
            'Qdrant API Key': "✅ Set" if st.secrets.get("qdrant_api_key") else "❌ Not set",
            'Groq API Key': "✅ Set" if st.secrets.get("groq_api_key") else "❌ Not set", 
            'Groq API Key (Secondary)': "✅ Set" if st.secrets.get("groq_api_key_1") else "❌ Not set",
            'LlamaCloud API Key': "✅ Set" if st.secrets.get("llama_cloud_api_key") else "❌ Not set"
        }
        
        for key, status in api_keys.items():
            if "Not set" in status:
                st.error(f"{key}: {status}")
            else:
                st.success(f"{key}: {status}")
                
    except Exception as e:
        st.error(f"Error checking API keys: {str(e)}")
    
    # System configuration
    st.markdown("### 🔧 LlamaIndex Configuration")
    if st.session_state.rag_system:
        col1, col2 = st.columns(2)
        
        with col1:
            st.info("**Embedding Model**: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
            st.info("**LLM Model**: Groq Llama3-70B-8192")
            st.info("**Vector Store**: Qdrant")
        
        with col2:
            try:
                chunk_size = getattr(st.session_state.rag_system.node_parser, 'chunk_size', 'N/A')
                chunk_overlap = getattr(st.session_state.rag_system.node_parser, 'chunk_overlap', 'N/A')
                st.info(f"**Chunk Size**: {chunk_size}")
                st.info(f"**Chunk Overlap**: {chunk_overlap}")
                st.info("**Framework**: LlamaIndex")
            except AttributeError:
                st.info("**Chunk Size**: N/A")
                st.info("**Chunk Overlap**: N/A")
                st.info("**Framework**: LlamaIndex")
    else:
        st.warning("System not initialized - configuration unavailable")
    
    # Data Management
    st.markdown("### 💾 Data Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Export functionality
        if st.button("📥 Export Q&A History"):
            if st.session_state.qa_history:
                try:
                    export_data = {
                        "qa_history": st.session_state.qa_history,
                        "processed_documents": st.session_state.processed_documents,
                        "system_type": "llamaindex",
                        "exported_at": datetime.now().isoformat(),
                        "total_questions": len(st.session_state.qa_history),
                        "total_documents": len(st.session_state.processed_documents)
                    }
                    
                    json_data = json.dumps(export_data, indent=2, ensure_ascii=False)
                    
                    st.download_button(
                        label="📄 Download JSON",
                        data=json_data,
                        file_name=f"llamaindex_rag_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                        help="Download your Q&A history and document information"
                    )
                    st.success("Export ready for download!")
                except Exception as e:
                    st.error(f"Export failed: {str(e)}")
            else:
                st.warning("No history to export")
    
    with col2:
        # Clear data functionality
        if st.button("🗑️ Clear All Data", type="secondary"):
            if st.button("⚠️ Confirm Clear All", type="primary"):
                st.session_state.qa_history = []
                st.session_state.processed_documents = []
                st.session_state.collection_stats = {'doc': 0}
                st.success("All data cleared!")
                st.rerun()
    
    # System Information
    st.markdown("### ℹ️ System Information")
    
    system_info = {
        "Streamlit Version": st.__version__,
        "Current Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "System Status": "✅ Ready" if st.session_state.system_initialized else "❌ Not Ready",
        "Documents Processed": len(st.session_state.processed_documents),
        "Questions Asked": len(st.session_state.qa_history),
        "Collections": len(set([doc['collection'] for doc in st.session_state.processed_documents])) if st.session_state.processed_documents else 0
    }
    
    for key, value in system_info.items():
        st.text(f"{key}: {value}")


def main():
    """Main application function"""
    try:
        # Initialize session state
        initialize_session_state()
        
        # Initialize RAG system
        initialize_rag_system()
        
        # Display header
        display_header()
        
        # Sidebar
        display_system_status()
        
        # Main content tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📄 Documents", "🤔 Q&A", "📊 Analytics", "⚙️ Settings"])
        
        with tab1:
            display_document_upload()
        
        with tab2:
            display_qa_interface()
        
        with tab3:
            display_analytics()
        
        with tab4:
            display_settings()
        
        # Footer
        st.markdown("---")
        st.markdown(
            "<div style='text-align: center; color: #666; padding: 1rem;'>"
            "🌐 Multilingual RAG System with LlamaIndex | Built with Streamlit | Supports Arabic & English"
            "</div>",
            unsafe_allow_html=True
        )
        
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.error("Please refresh the page and try again.")


if __name__ == "__main__":
    main()