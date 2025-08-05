import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import tempfile
import json
from datetime import datetime
import time
from typing import List, Dict, Any
import hashlib


from rag_sys import RAGResponse , MultilingualRAGSystem

# Configure Streamlit page
st.set_page_config(
    page_title="Multilingual RAG System",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .upload-section {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .qa-section {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #e9ecef;
        margin: 1rem 0;
    }
    
    .answer-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    
    .source-box {
        background-color: #e9ecef;
        padding: 0.8rem;
        border-radius: 6px;
        margin: 0.5rem 0;
        font-size: 0.9em;
    }
    
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    
    .rtl-text {
        direction: rtl;
        text-align: right;
        font-family: 'Noto Sans Arabic', sans-serif;
    }
</style>

<link href="https://fonts.googleapis.com/css2?family=Noto+Sans+Arabic:wght@400;600&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

# Initialize session state
def initialize_session_state():
    """Initialize all session state variables"""
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = None
    if 'processed_documents' not in st.session_state:
        st.session_state.processed_documents = []
    if 'qa_history' not in st.session_state:
        st.session_state.qa_history = []
    if 'system_initialized' not in st.session_state:
        st.session_state.system_initialized = False
    if 'collection_stats' not in st.session_state:
        st.session_state.collection_stats = {'arabic': 0, 'english': 0}

def validate_api_keys():
    """Validate required API keys"""
    required_keys = {
        'QDRANT_URL': 'Qdrant Cloud URL',
        'QDRANT_API_KEY': 'Qdrant API Key',
        'GROQ_API_KEY': 'Groq API Key',
        'GROQ_API_KEY_1': 'Groq API Key (Secondary)',
        'LLAMA_CLOUD_API_KEY': 'LlamaCloud API Key'
    }
    
    missing_keys = []
    for key, description in required_keys.items():
        if not os.getenv(key):
            missing_keys.append(f"{description} ({key})")
    
    return missing_keys

def initialize_rag_system():
    """Initialize the RAG system with error handling"""
    try:
        # Check if all required environment variables are set
        missing_keys = validate_api_keys()
        if missing_keys:
            st.error(f"Missing required environment variables: {', '.join(missing_keys)}")
            st.info("Please set these environment variables in your .env file or system environment.")
            return None
        
        # Initialize the RAG system
        rag_system = MultilingualRAGSystem(
            llama_api_key=os.getenv("LLAMA_CLOUD_API_KEY"),
            qdrant_url=os.getenv("QDRANT_URL"),
            qdrant_api_key=os.getenv("QDRANT_API_KEY"),
            groq_api_key=os.getenv("GROQ_API_KEY"),
            llm_provider="groq",
            chunk_target_size=400,
            chunk_max_size=600
        )
        
        st.success("✅ RAG System initialized successfully!")
        return rag_system
        
    except Exception as e:
        st.error(f"❌ Failed to initialize RAG system: {str(e)}")
        return None

def display_header():
    """Display the main header"""
    st.markdown("""
    <div class="main-header">
        <h1>🌐 Multilingual RAG System</h1>
        <p>Intelligent Document Processing & Question Answering for Arabic & English</p>
    </div>
    """, unsafe_allow_html=True)

def display_system_status():
    """Display system status in sidebar"""
    st.sidebar.markdown("## 🚀 System Status")
    
    if st.session_state.system_initialized and st.session_state.rag_system:
        st.sidebar.success("✅ System Ready")
        
        # Display collection statistics
        st.sidebar.markdown("### 📊 Document Collections")
        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.metric("Arabic Docs", st.session_state.collection_stats['arabic'])
        with col2:
            st.metric("English Docs", st.session_state.collection_stats['english'])
            
    else:
        st.sidebar.warning("⚠️ System Not Initialized")
        if st.sidebar.button("🔄 Initialize System"):
            with st.spinner("Initializing RAG system..."):
                rag_system = initialize_rag_system()
                if rag_system:
                    st.session_state.rag_system = rag_system
                    st.session_state.system_initialized = True
                    st.rerun()

def display_document_upload():
    """Display document upload section"""
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    st.markdown("## 📄 Document Upload & Processing")
    
    if not st.session_state.system_initialized:
        st.warning("⚠️ Please initialize the system first using the sidebar.")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_files = st.file_uploader(
            "Upload PDF documents (Arabic or English)",
            type=['pdf'],
            accept_multiple_files=True,
            help="Upload one or more PDF files. The system will automatically detect the language."
        )
    
    with col2:
        language_option = st.selectbox(
            "Language Detection",
            ["Auto-detect", "Force Arabic", "Force English"],
            help="Choose how to handle language detection"
        )
        
        chunk_size = st.slider(
            "Target Chunk Size (tokens)",
            min_value=200,
            max_value=800,
            value=400,
            step=50,
            help="Adjust the target size for text chunks"
        )
    
    if uploaded_files and st.button("🚀 Process Documents", type="primary"):
        process_documents(uploaded_files, language_option, chunk_size)
    
    # Display processed documents
    if st.session_state.processed_documents:
        st.markdown("### 📚 Processed Documents")
        df = pd.DataFrame([
            {
                "Document": doc["name"],
                "Language": "Arabic" if doc["language"] == "ar" else "English",
                "Chunks": doc["chunks"],
                "Pages": doc["pages"],
                "Processing Time": f"{doc['processing_time']:.2f}s",
                "Status": "✅ Ready"
            }
            for doc in st.session_state.processed_documents
        ])
        st.dataframe(df, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def process_documents(uploaded_files, language_option, chunk_size):
    """Process uploaded documents"""
    # Update chunking parameters
    if st.session_state.rag_system:
        st.session_state.rag_system.chunking_strategy.target_chunk_size = chunk_size
        st.session_state.rag_system.chunking_strategy.max_chunk_size = int(chunk_size * 1.5)
    
    # Map language options
    lang_map = {
        "Auto-detect": "auto",
        "Force Arabic": "ar", 
        "Force English": "en"
    }
    language = lang_map[language_option]
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, uploaded_file in enumerate(uploaded_files):
        try:
            # Update progress
            progress = (i + 1) / len(uploaded_files)
            progress_bar.progress(progress)
            status_text.text(f"Processing {uploaded_file.name}...")
            
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            start_time = time.time()
            
            # Process document
            chunks, detected_language = st.session_state.rag_system.process_pdf_to_qdrant_multilingual(
                pdf_path=tmp_file_path,
                indice=len(st.session_state.processed_documents) + 1,
                language=language,
                document_name=uploaded_file.name.replace('.pdf', '')
            )
            
            processing_time = time.time() - start_time
            
            if chunks:
                # Add to processed documents
                doc_info = {
                    "name": uploaded_file.name,
                    "language": detected_language,
                    "chunks": len(chunks),
                    "pages": max(chunk.page_number for chunk in chunks) if chunks else 0,
                    "processing_time": processing_time,
                    "processed_at": datetime.now().isoformat()
                }
                st.session_state.processed_documents.append(doc_info)
                
                # Update collection stats
                if detected_language == 'ar':
                    st.session_state.collection_stats['arabic'] += 1
                else:
                    st.session_state.collection_stats['english'] += 1
                
                st.success(f"✅ Successfully processed {uploaded_file.name} "
                          f"({'Arabic' if detected_language == 'ar' else 'English'}) "
                          f"- {len(chunks)} chunks created")
            else:
                st.error(f"❌ Failed to process {uploaded_file.name}")
            
            # Clean up temporary file
            os.unlink(tmp_file_path)
            
        except Exception as e:
            st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
            continue
    
    progress_bar.progress(1.0)
    status_text.text("✅ All documents processed!")
    time.sleep(1)
    progress_bar.empty()
    status_text.empty()

def display_qa_interface():
    """Display Q&A interface"""
    st.markdown('<div class="qa-section">', unsafe_allow_html=True)
    st.markdown("## 🤔 Ask Questions")
    
    if not st.session_state.system_initialized:
        st.warning("⚠️ Please initialize the system first using the sidebar.")
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
            help="Ask questions in Arabic or English. The system will automatically detect the language."
        )
    
    with col2:
        st.markdown("### ⚙️ Settings")
        
        retrieval_limit = st.slider(
            "Number of sources",
            min_value=3,
            max_value=20,
            value=5,
            help="How many relevant chunks to retrieve"
        )
        
        temperature = st.slider(
            "Response creativity",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            step=0.1,
            help="Lower values = more focused, Higher values = more creative"
        )
        
        filter_types = st.multiselect(
            "Filter by content type",
            ["title", "paragraph", "list", "table", "hybrid"],
            help="Filter retrieved content by type"
        )
    
    # Submit button
    if st.button("🔍 Get Answer", type="primary", disabled=not question.strip()):
        get_answer(question, retrieval_limit, temperature, filter_types)
    
    # Display Q&A history
    display_qa_history()
    
    st.markdown('</div>', unsafe_allow_html=True)

def get_answer(question, retrieval_limit, temperature, filter_types):
    """Get answer for the question"""
    with st.spinner("🤖 Thinking... Please wait"):
        try:
            start_time = time.time()
            
            # Get response from RAG system
            response = st.session_state.rag_system.ask_question(
                query=question,
                retrieval_limit=retrieval_limit,
                filter_by_type=filter_types if filter_types else None,
                temperature=temperature
            )
            
            processing_time = time.time() - start_time
            
            # Add to history
            qa_entry = {
                "question": question,
                "answer": response.answer,
                "language": response.query_language,
                "confidence": response.confidence_score,
                "sources": response.sources,
                "retrieved_chunks": response.retrieved_chunks,
                "processing_time": processing_time,
                "timestamp": datetime.now().isoformat()
            }
            st.session_state.qa_history.insert(0, qa_entry)  # Add to beginning
            
            # Display the answer
            display_answer(qa_entry)
            
        except Exception as e:
            st.error(f"❌ Error getting answer: {str(e)}")

def display_answer(qa_entry):
    """Display a Q&A entry with proper formatting"""
    language = qa_entry["language"]
    is_arabic = language == "ar"
    
    # Apply RTL styling for Arabic
    text_class = "rtl-text" if is_arabic else ""
    
    st.markdown("---")
    
    # Question
    st.markdown(f"### 🤔 Question ({'Arabic' if is_arabic else 'English'})")
    st.markdown(f'<div class="{text_class}">{qa_entry["question"]}</div>', unsafe_allow_html=True)
    
    # Answer
    st.markdown("### 💡 Answer")
    st.markdown(f'<div class="answer-box {text_class}">{qa_entry["answer"]}</div>', unsafe_allow_html=True)
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Confidence", f"{qa_entry['confidence']:.3f}")
    with col2:
        st.metric("Sources", len(qa_entry['sources']))
    with col3:
        st.metric("Retrieved Chunks", len(qa_entry['retrieved_chunks']))
    with col4:
        st.metric("Response Time", f"{qa_entry['processing_time']:.2f}s")
    
    # Sources
    if qa_entry['sources']:
        st.markdown("### 📚 Sources")
        for i, source in enumerate(qa_entry['sources'], 1):
            st.markdown(f'<div class="source-box">📄 {i}. {source}</div>', unsafe_allow_html=True)
    
    # Detailed chunk information (expandable)
    with st.expander("🔍 View Retrieved Chunks Details"):
        for i, chunk in enumerate(qa_entry['retrieved_chunks'], 1):
            st.markdown(f"**Chunk {i}** - Score: {chunk['score']:.3f}")
            st.markdown(f"- **Type**: {chunk['chunk_type']}")
            st.markdown(f"- **Strategy**: {chunk['chunk_strategy']}")
            st.markdown(f"- **Page**: {chunk['page_number']}")
            st.markdown(f"- **Document**: {chunk['document_name']}")
            st.markdown(f"- **Tokens**: {chunk.get('token_count', 'N/A')}")
            
            # Show content preview
            content_preview = chunk['content'][:200] + "..." if len(chunk['content']) > 200 else chunk['content']
            st.markdown(f'<div class="{text_class}"><small>{content_preview}</small></div>', unsafe_allow_html=True)
            st.markdown("---")

def display_qa_history():
    """Display Q&A history"""
    if st.session_state.qa_history:
        st.markdown("### 📈 Recent Questions & Answers")
        
        # Add clear history button
        if st.button("🗑️ Clear History"):
            st.session_state.qa_history = []
            st.rerun()
        
        # Show recent entries (limit to 3 for performance)
        for qa_entry in st.session_state.qa_history[:3]:
            with st.expander(f"Q: {qa_entry['question'][:50]}..." if len(qa_entry['question']) > 50 else f"Q: {qa_entry['question']}"):
                display_answer(qa_entry)

def display_analytics():
    """Display analytics and statistics"""
    st.markdown("## 📊 System Analytics")
    
    if not st.session_state.processed_documents and not st.session_state.qa_history:
        st.info("📈 Analytics will appear here after processing documents and asking questions.")
        return
    
    # Document statistics
    if st.session_state.processed_documents:
        st.markdown("### 📚 Document Processing Statistics")
        
        # Create charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Language distribution pie chart
            lang_counts = {'Arabic': st.session_state.collection_stats['arabic'], 
                          'English': st.session_state.collection_stats['english']}
            
            if any(lang_counts.values()):
                fig_pie = px.pie(
                    values=list(lang_counts.values()),
                    names=list(lang_counts.keys()),
                    title="Documents by Language"
                )
                st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Processing time chart
            if st.session_state.processed_documents:
                doc_df = pd.DataFrame(st.session_state.processed_documents)
                fig_bar = px.bar(
                    doc_df,
                    x='name',
                    y='processing_time',
                    title="Processing Time by Document",
                    labels={'processing_time': 'Time (seconds)', 'name': 'Document'}
                )
                fig_bar.update_layout(xaxis_tickangle=45)
                st.plotly_chart(fig_bar, use_container_width=True)
    
    # Q&A statistics
    if st.session_state.qa_history:
        st.markdown("### 🤔 Question & Answer Statistics")
        
        qa_df = pd.DataFrame(st.session_state.qa_history)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_confidence = qa_df['confidence'].mean()
            st.metric("Average Confidence", f"{avg_confidence:.3f}")
        
        with col2:
            avg_response_time = qa_df['processing_time'].mean()
            st.metric("Avg Response Time", f"{avg_response_time:.2f}s")
        
        with col3:
            total_questions = len(qa_df)
            st.metric("Total Questions", total_questions)
        
        # Language distribution for questions
        col1, col2 = st.columns(2)
        
        with col1:
            qa_lang_counts = qa_df['language'].value_counts()
            qa_lang_labels = ['Arabic' if lang == 'ar' else 'English' for lang in qa_lang_counts.index]
            
            fig_qa_pie = px.pie(
                values=qa_lang_counts.values,
                names=qa_lang_labels,
                title="Questions by Language"
            )
            st.plotly_chart(fig_qa_pie, use_container_width=True)
        
        with col2:
            # Confidence score distribution
            fig_conf = px.histogram(
                qa_df,
                x='confidence',
                title="Confidence Score Distribution",
                nbins=10
            )
            st.plotly_chart(fig_conf, use_container_width=True)

def display_settings():
    """Display system settings and configuration"""
    st.markdown("## ⚙️ System Settings")
    
    # API Key status
    st.markdown("### 🔑 API Configuration")
    api_keys = {
        'Qdrant URL': os.getenv("QDRANT_URL", "Not set"),
        'Qdrant API Key': "Set" if os.getenv("QDRANT_API_KEY") else "Not set",
        'Groq API Key': "Set" if os.getenv("GROQ_API_KEY") else "Not set", 
        'Groq API Key (Secondary)': "Set" if os.getenv("GROQ_API_KEY_1") else "Not set",
        'LlamaCloud API Key': "Set" if os.getenv("LLAMA_CLOUD_API_KEY") else "Not set"
    }
    
    for key, status in api_keys.items():
        if "Not set" in status:
            st.error(f"❌ {key}: {status}")
        else:
            st.success(f"✅ {key}: {status}")
    
    # System configuration
    st.markdown("### 🔧 System Configuration")
    if st.session_state.rag_system:
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"**Embedding Model**: {st.session_state.rag_system.embedding_model}")
            st.info(f"**LLM Provider**: {st.session_state.rag_system.llm_provider}")
        
        with col2:
            st.info(f"**Target Chunk Size**: {st.session_state.rag_system.chunking_strategy.target_chunk_size}")
            st.info(f"**Max Chunk Size**: {st.session_state.rag_system.chunking_strategy.max_chunk_size}")
    
    # Export/Import settings
    st.markdown("### 💾 Data Management")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📥 Export Q&A History"):
            if st.session_state.qa_history:
                export_data = {
                    "qa_history": st.session_state.qa_history,
                    "processed_documents": st.session_state.processed_documents,
                    "exported_at": datetime.now().isoformat()
                }
                
                st.download_button(
                    label="Download JSON",
                    data=json.dumps(export_data, indent=2, ensure_ascii=False),
                    file_name=f"rag_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            else:
                st.warning("No history to export")
    
    with col2:
        if st.button("🗑️ Clear All Data"):
            if st.button("⚠️ Confirm Clear All", type="secondary"):
                st.session_state.processed_documents = []
                st.session_state.qa_history = []
                st.session_state.collection_stats = {'arabic': 0, 'english': 0}
                st.success("✅ All data cleared")
                st.rerun()
    
    with col3:
        if st.button("🔄 Reset System"):
            if st.button("⚠️ Confirm Reset", type="secondary"):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.success("✅ System reset")
                st.rerun()

def main():
    """Main application function"""
    # Initialize session state
    initialize_session_state()
    
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
        "🌐 Multilingual RAG System | Built with Streamlit | Supports Arabic & English"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()