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

# Import the LlamaIndex-based RAG system
from rag_sys import RAGResponse, MultilingualRAGWithLlamaIndex

st.set_page_config(
    page_title="Multilingual RAG System - LlamaIndex",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("""
<style>
/* Use prefers-color-scheme to detect light/dark mode */
@media (prefers-color-scheme: dark) {
    .stApp {
        background-color: #0e1117 !important;  /* Dark background */
        color: white !important;
    }
    section[data-testid="stSidebar"] {
        background-color: #1c1f26 !important;  /* Dark sidebar */
    }
    .upload-section, .qa-section, .answer-box, .metric-card {
        background-color: #1c1f26 !important;
        color: white !important;
        border: 1px solid #333 !important;
    }
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
    }
}

/* Light Mode (default) */
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
    }
}
</style>
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
    try:
        # Check if all required secrets are set
        missing_keys = validate_api_keys()
        if missing_keys:
            st.error(f"Missing required secrets: {', '.join(missing_keys)}")
            st.info("Please set these secrets in your Streamlit secrets.toml file.")
            return None
        
        # Initialize the LlamaIndex RAG system
        rag_system = MultilingualRAGWithLlamaIndex(
            llama_api_key=st.secrets["llama_cloud_api_key"],
            qdrant_url=st.secrets["qdrant_url"],
            qdrant_api_key=st.secrets["qdrant_api_key"],
            groq_api_key=st.secrets["groq_api_key"],
            embedding_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            groq_model="llama3-70b-8192",
            chunk_size=512,
            chunk_overlap=50
        )
        
        st.success("✅ LlamaIndex RAG System initialized successfully!")
        return rag_system
        
    except Exception as e:
        st.error(f"❌ Failed to initialize RAG system: {str(e)}")
        return None

def display_header():
    """Display the main header"""
    st.markdown("""
    <div class="main-header">
        <h1>🌐 Multilingual RAG System - LlamaIndex</h1>
        <p>Intelligent Document Processing & Question Answering for Arabic & English using LlamaIndex</p>
    </div>
    """, unsafe_allow_html=True)

# Display collection information with document counts
def display_system_status():
    """Display system status in sidebar"""
    st.sidebar.markdown("## 🚀 System Status")
    
    if st.session_state.system_initialized and st.session_state.rag_system:
        st.sidebar.success("✅ LlamaIndex System Ready")
        
        # Display collection statistics
        st.sidebar.markdown("### 📊 Document Collections")
        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.metric("Arabic Docs", st.session_state.collection_stats['arabic'])
        with col2:
            st.metric("English Docs", st.session_state.collection_stats['english'])
            
        # Display system info
        st.sidebar.markdown("### ⚙️ System Info")
        st.sidebar.info("**Engine**: LlamaIndex")
        st.sidebar.info("**LLM**: Groq Llama3-70B")
        st.sidebar.info("**Embeddings**: Multilingual MiniLM")
        st.sidebar.info("**Vector DB**: Qdrant")
        
        # Add collection management
        st.sidebar.markdown("### 🗂️ Collection Management") 
        
        if st.sidebar.button("🔍 Check Collections"):
            check_collections_status()
            
        if st.sidebar.button("🗑️ Clear Arabic Collection"):
            clear_collection_confirm("arabic")
            
        if st.sidebar.button("🗑️ Clear English Collection"):
            clear_collection_confirm("english")
            
    else:
        st.sidebar.warning("⚠️ System Not Initialized")
        if st.sidebar.button("🔄 Initialize System"):
            with st.spinner("Initializing LlamaIndex RAG system..."):
                rag_system = initialize_rag_system()
                if rag_system:
                    st.session_state.rag_system = rag_system
                    st.session_state.system_initialized = True
                    st.rerun()

def check_collections_status():
    """Check the actual status of collections in Qdrant"""
    try:
        collections = st.session_state.rag_system.qdrant_client.get_collections()
        collection_names = [c.name for c in collections.collections]
        
        st.sidebar.markdown("**Qdrant Collections:**")
        for name in collection_names:
            if "arabic" in name.lower():
                try:
                    info = st.session_state.rag_system.qdrant_client.get_collection(name)
                    st.sidebar.success(f"🟢 {name}: {info.points_count} points")
                except:
                    st.sidebar.warning(f"🟡 {name}: Status unknown")
            elif "english" in name.lower():
                try:
                    info = st.session_state.rag_system.qdrant_client.get_collection(name)
                    st.sidebar.success(f"🟢 {name}: {info.points_count} points")
                except:
                    st.sidebar.warning(f"🟡 {name}: Status unknown")
    except Exception as e:
        st.sidebar.error(f"Error checking collections: {e}")

def clear_collection_confirm(language):
    """Confirm and clear a specific collection"""
    collection_name = f"{language}_docs_llamaindex"
    
    if st.sidebar.button(f"⚠️ Confirm Clear {language.title()}", key=f"confirm_clear_{language}"):
        try:
            st.session_state.rag_system.qdrant_client.delete_collection(collection_name)
            st.session_state.collection_stats[language] = 0
            
            # Remove from processed documents
            st.session_state.processed_documents = [
                doc for doc in st.session_state.processed_documents 
                if doc['language'] != ('ar' if language == 'arabic' else 'en')
            ]
            
            st.sidebar.success(f"✅ Cleared {language} collection")
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"Error clearing collection: {e}")

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
            accept_multiple_files=False,
            help="Upload one or more PDF files. The system will automatically detect the language and extract structured content using LlamaIndex."
        )
    
    with col2:
        language_option = st.selectbox(
            "Language Detection",
            ["Force Arabic", "Force English"],
            help="Choose how to handle language detection"
        )
        
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
        
        # Add option to append or replace documents
        collection_mode = st.radio(
            "Collection Mode",
            ["Append to existing", "Replace collection"],
            help="Choose whether to add documents to existing collection or replace it entirely"
        )
    
    if uploaded_files and st.button("🚀 Process Documents with LlamaIndex", type="primary"):
        uploaded_files  = uploaded_files if isinstance(uploaded_files, list) else [uploaded_files]
        process_documents_llamaindex(uploaded_files, language_option, chunk_size, chunk_overlap, collection_mode)
    
    # Display processed documents
    if st.session_state.processed_documents:
        st.markdown("### 📚 Processed Documents")
        df = pd.DataFrame([
            {
                "Document": doc["name"],
                "Language": "Arabic" if doc["language"] == "ar" else "English",
                "LlamaIndex Documents": doc["documents"],
                "Collection": doc["collection"],
                "Processing Time": f"{doc['processing_time']:.2f}s",
                "Status": "✅ Indexed",
                "Mode": doc.get("mode", "Unknown")
            }
            for doc in st.session_state.processed_documents
        ])
        st.dataframe(df, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def process_documents_llamaindex(uploaded_files, language_option, chunk_size, chunk_overlap, collection_mode):
    """Process uploaded documents using LlamaIndex pipeline"""
    # Update chunking parameters
    if st.session_state.rag_system:
        st.session_state.rag_system.node_parser.chunk_size = chunk_size
        st.session_state.rag_system.node_parser.chunk_overlap = chunk_overlap
    
    # Map language options
    lang_map = {
        "Force Arabic": "ar", 
        "Force English": "en"
    }
    language = lang_map[language_option]
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Keep track of processed documents in this batch by language
    batch_stats = {'arabic': 0, 'english': 0}
    
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
            
            # Check if this is the first document of this language in this batch
            # and if we should replace the collection
            document_name = uploaded_file.name.replace('.pdf', '')
            
            # Detect language first if needed
            if language == "auto":
                # Quick language detection
                extracted_sample, detected_lang = st.session_state.rag_system.extract_pdf(tmp_file_path, "auto")
                actual_language = detected_lang
            else:
                actual_language = language
            
            # Determine if we need to clear the collection
            should_clear_collection = False
            if collection_mode == "Replace collection":
                if actual_language == 'ar' and batch_stats['arabic'] == 0:
                    should_clear_collection = True
                elif actual_language == 'en' and batch_stats['english'] == 0:
                    should_clear_collection = True
            
            # Temporarily modify the RAG system to control collection clearing
            if should_clear_collection:
                # This is the first document of this language in replace mode
                index, detected_language = st.session_state.rag_system.process_pdf_to_llamaindex(
                    pdf_path=tmp_file_path,
                    language=actual_language,
                    document_name=document_name
                )
                status_message = f"Created new collection for {uploaded_file.name}"
            else:
                # We need to append to existing collection
                # We'll need to modify the process to append rather than replace
                index, detected_language = process_pdf_append_mode(
                    st.session_state.rag_system,
                    tmp_file_path,
                    actual_language,
                    document_name
                )
                status_message = f"Added to existing collection: {uploaded_file.name}"
            
            processing_time = time.time() - start_time
            
            if index:
                # Determine collection name
                collection_name = "arabic_docs_llamaindex" if detected_language == 'ar' else "english_docs_llamaindex"
                
                # Update batch stats
                if detected_language == 'ar':
                    batch_stats['arabic'] += 1
                else:
                    batch_stats['english'] += 1
                
                # Add to processed documents
                doc_info = {
                    "name": uploaded_file.name,
                    "language": detected_language,
                    "documents": len(index.docstore.docs) if hasattr(index, 'docstore') else "N/A",
                    "collection": collection_name,
                    "processing_time": processing_time,
                    "processed_at": datetime.now().isoformat(),
                    "mode": collection_mode
                }
                st.session_state.processed_documents.append(doc_info)
                
                # Update collection stats only for new documents
                if collection_mode == "Append to existing" or should_clear_collection:
                    if detected_language == 'ar':
                        if should_clear_collection:
                            st.session_state.collection_stats['arabic'] = 1
                        else:
                            st.session_state.collection_stats['arabic'] += 1
                    else:
                        if should_clear_collection:
                            st.session_state.collection_stats['english'] = 1
                        else:
                            st.session_state.collection_stats['english'] += 1
                
                st.success(f"✅ {status_message} "
                          f"({'Arabic' if detected_language == 'ar' else 'English'}) "
                          f"- Indexed to {collection_name}")
            else:
                st.error(f"❌ Failed to process {uploaded_file.name}")
            
            # Clean up temporary file
            os.unlink(tmp_file_path)
            
        except Exception as e:
            st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
            continue
    
    progress_bar.progress(1.0)
    status_text.text("✅ All documents processed with LlamaIndex!")
    time.sleep(1)
    progress_bar.empty()
    status_text.empty()

def process_pdf_append_mode(rag_system, pdf_path, language, document_name):
    """Process PDF in append mode without clearing existing collection"""
    try:
        # Extract PDF content
        extracted_data, detected_language = rag_system.extract_pdf(pdf_path, language)
        
        if not extracted_data:
            return None, None
        
        # Convert to LlamaIndex Documents
        documents = rag_system.convert_to_llamaindex_documents(
            extracted_data, document_name, detected_language
        )
        
        # Determine collection name
        collection_name = "arabic_docs_llamaindex" if detected_language == 'ar' else "english_docs_llamaindex"
        
        # Load existing index or create new one
        existing_index = rag_system.load_existing_index(collection_name, detected_language)
        
        if existing_index:
            # Insert new documents into existing index
            for doc in documents:
                existing_index.insert(doc)
            print(f"✅ Added {len(documents)} documents to existing {collection_name}")
            return existing_index, detected_language
        else:
            # Create new index if none exists
            index = rag_system.create_index(documents, collection_name, detected_language)
            print(f"✅ Created new index {collection_name} with {len(documents)} documents")
            return index, detected_language
            
    except Exception as e:
        print(f"Error in append mode processing: {e}")
        return None, None

def display_qa_interface():
    """Display Q&A interface"""
    st.markdown('<div class="qa-section">', unsafe_allow_html=True)
    st.markdown("## 🤔 Ask Questions - LlamaIndex RAG")
    
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
            help="Ask questions in Arabic or English. LlamaIndex will automatically detect the language and use appropriate retrieval."
        )
    
    with col2:
        st.markdown("### ⚙️ LlamaIndex Settings")
        
        similarity_top_k = st.slider(
            "Number of sources (top_k)",
            min_value=3,
            max_value=20,
            value=10,
            help="How many relevant chunks to retrieve from vector index"
        )
        
        retrieval_method = st.selectbox(
            "Retrieval Method",
            ["Manual Retrieval + Custom Prompt"],
            help="Choose between LlamaIndex query engine or manual retrieval with custom prompts"
        )
        
        similarity_cutoff = st.slider(
            "Similarity Cutoff",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            step=0.05,
            help="Minimum similarity score for retrieved chunks (lower = more inclusive)"
        )
        
        # Add debug mode
        debug_mode = st.checkbox(
            "Debug Mode",
            help="Show retrieved chunk contents for debugging"
        )
    
    # Submit button
    if st.button("🔍 Get Answer with LlamaIndex", type="primary", disabled=not question.strip()):
        get_answer_llamaindex(question, similarity_top_k, retrieval_method, similarity_cutoff, debug_mode)
    
    # Display Q&A history
    display_qa_history()
    
    st.markdown('</div>', unsafe_allow_html=True)

def get_answer_llamaindex(question, similarity_top_k, retrieval_method, similarity_cutoff, debug_mode=False):
    """Get answer using LlamaIndex RAG system"""
    with st.spinner("🤖 LlamaIndex is thinking... Please wait"):
        try:
            start_time = time.time()
            
            # Choose retrieval method
            if retrieval_method == "Manual Retrieval + Custom Prompt":
                response = st.session_state.rag_system.ask_question_with_manual_retrieval(
                    query=question,
                    similarity_top_k=similarity_top_k
                )
            else:
                response = st.session_state.rag_system.ask_question(
                    query=question,
                    similarity_top_k=similarity_top_k
                )
            
            processing_time = time.time() - start_time
            
            # Debug mode: show retrieved chunks
            if debug_mode:
                st.markdown("### 🔍 Debug: Retrieved Chunks")
                for i, chunk in enumerate(response.retrieved_chunks, 1):
                    with st.expander(f"Debug Chunk {i} - Score: {chunk['score']:.3f}"):
                        st.write(f"**Document:** {chunk.get('document_name', 'N/A')}")
                        st.write(f"**Page:** {chunk.get('page_number', 'N/A')}")
                        st.write(f"**Block Type:** {chunk.get('block_type', 'N/A')}")
                        st.write("**Content:**")
                        st.text(chunk['content'])
            
            # Add to history
            qa_entry = {
                "question": question,
                "answer": response.answer,
                "language": response.query_language,
                "confidence": response.confidence_score,
                "sources": response.sources if response.sources else [],
                "retrieved_chunks": response.retrieved_chunks,
                "processing_time": processing_time,
                "timestamp": datetime.now().isoformat(),
                "method": retrieval_method,
                "top_k": similarity_top_k
            }
            st.session_state.qa_history.insert(0, qa_entry)  # Add to beginning
            
            # Display the answer
            display_answer_llamaindex(qa_entry)
            
        except Exception as e:
            st.error(f"❌ Error getting answer from LlamaIndex: {str(e)}")
            import traceback
            st.error(f"Traceback: {traceback.format_exc()}")

def display_answer_llamaindex(qa_entry):
    """Display a Q&A entry with LlamaIndex-specific formatting"""
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
    st.markdown(
    f"""
    <div class="answer-box" style="padding: 15px; border-radius: 8px;">
        {qa_entry['answer']}
    </div>
    """,
    unsafe_allow_html=True
)

    
    # Metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Confidence", f"{qa_entry['confidence']:.3f}")
    with col2:
        st.metric("Sources", len(qa_entry['sources']))
    with col3:
        st.metric("Retrieved Chunks", len(qa_entry['retrieved_chunks']))
    with col4:
        st.metric("Response Time", f"{qa_entry['processing_time']:.2f}s")
    with col5:
        st.metric("Method", qa_entry.get('method', 'Standard')[:8] + "...")
    
    # Sources
    if qa_entry['sources']:
        st.markdown("### 📚 Sources")
        for i, source in enumerate(qa_entry['sources'], 1):
            st.markdown(f'<div class="source-box">📄 {i}. {source}</div>', unsafe_allow_html=True)
    
    # Detailed chunk information (expandable)
    with st.expander("🔍 View LlamaIndex Retrieved Chunks Details"):
        for i, chunk in enumerate(qa_entry['retrieved_chunks'], 1):
            st.markdown(f"**Chunk {i}** - Similarity Score: {chunk['score']:.3f}")
            st.markdown(f"- **Document**: {chunk.get('document_name', 'N/A')}")
            st.markdown(f"- **Page**: {chunk.get('page_number', 'N/A')}")
            st.markdown(f"- **Block Type**: {chunk.get('block_type', 'N/A')}")
            st.markdown(f"- **Language**: {chunk.get('language', 'N/A')}")
            
            # Show content preview
            content_preview = chunk['content'][:300] + "..." if len(chunk['content']) > 300 else chunk['content']
            st.markdown(f'<div class="{text_class}"><small><strong>Content:</strong><br>{content_preview}</small></div>', unsafe_allow_html=True)
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
                    title="Documents by Language (LlamaIndex Collections)"
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
                    color='language',
                    title="LlamaIndex Processing Time by Document",
                    labels={'processing_time': 'Time (seconds)', 'name': 'Document'}
                )
                fig_bar.update_layout(xaxis_tickangle=45)
                st.plotly_chart(fig_bar, use_container_width=True)
    
    # Q&A statistics
    if st.session_state.qa_history:
        st.markdown("### 🤔 Question & Answer Statistics")
        
        qa_df = pd.DataFrame(st.session_state.qa_history)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_confidence = qa_df['confidence'].mean()
            st.metric("Average Confidence", f"{avg_confidence:.3f}")
        
        with col2:
            avg_response_time = qa_df['processing_time'].mean()
            st.metric("Avg Response Time", f"{avg_response_time:.2f}s")
        
        with col3:
            total_questions = len(qa_df)
            st.metric("Total Questions", total_questions)
        
        with col4:
            avg_chunks = qa_df['retrieved_chunks'].apply(len).mean()
            st.metric("Avg Retrieved Chunks", f"{avg_chunks:.1f}")
        
        # Additional analytics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Language distribution for questions
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
        
        with col3:
            # Method usage if available
            if 'method' in qa_df.columns:
                method_counts = qa_df['method'].value_counts()
                fig_method = px.bar(
                    x=method_counts.index,
                    y=method_counts.values,
                    title="Retrieval Method Usage"
                )
                st.plotly_chart(fig_method, use_container_width=True)

def display_settings():
    """Display system settings and configuration"""
    st.markdown("## ⚙️ LlamaIndex System Settings")
    
    # API Key status
    st.markdown("### 🔑 API Configuration")
    api_keys = {
        'Qdrant URL': st.secrets.get("qdrant_url", "Not set"),
        'Qdrant API Key': "Set" if st.secrets.get("qdrant_api_key") else "Not set",
        'Groq API Key': "Set" if st.secrets.get("groq_api_key") else "Not set", 
        'Groq API Key (Secondary)': "Set" if st.secrets.get("groq_api_key_1") else "Not set",
        'LlamaCloud API Key': "Set" if st.secrets.get("llama_cloud_api_key") else "Not set"
    }
    
    for key, status in api_keys.items():
        if "Not set" in status:
            st.error(f"❌ {key}: {status}")
        else:
            st.success(f"✅ {key}: {status}")
    
    # System configuration
    st.markdown("### 🔧 LlamaIndex Configuration")
    if st.session_state.rag_system:
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"**Embedding Model**: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
            st.info(f"**LLM Model**: Groq Llama3-70B-8192")
            st.info(f"**Vector Store**: Qdrant")
        
        with col2:
            st.info(f"**Chunk Size**: {st.session_state.rag_system.node_parser.chunk_size}")
            st.info(f"**Chunk Overlap**: {st.session_state.rag_system.node_parser.chunk_overlap}")
            st.info(f"**Framework**: LlamaIndex")
    
    # Collection Information
    st.markdown("### 📊 Collection Information")
    if st.session_state.rag_system:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Arabic Collection**")
            st.code("arabic_docs_llamaindex")
            
        with col2:
            st.markdown("**English Collection**")
            st.code("english_docs_llamaindex")
    
    # Export/Import settings
    st.markdown("### 💾 Data Management")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📥 Export Q&A History"):
            if st.session_state.qa_history:
                export_data = {
                    "qa_history": st.session_state.qa_history,
                    "processed_documents": st.session_state.processed_documents,
                    "system_type": "llamaindex",
                    "exported_at": datetime.now().isoformat()
                }
                
                st.download_button(
                    label="Download JSON",
                    data=json.dumps(export_data, indent=2, ensure_ascii=False),
                    file_name=f"llamaindex_rag_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
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
        "🌐 Multilingual RAG System with LlamaIndex | Built with Streamlit | Supports Arabic & English"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()