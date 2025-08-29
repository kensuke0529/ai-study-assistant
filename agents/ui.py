import streamlit as st
import sys
import os
from pathlib import Path
import time
import json
import uuid
from datetime import datetime

# Import the RAG components directly from the same directory
from rag import (
    SourceType, 
    SourcePreference, 
    EnhancedRAGSystem, 
    DocRetrieverRunnable, 
    GSearchRunnable, 
    EnhancedFlashcardMaker,
    llm,
    vector_store,
    text_splitter,
    folder_path,
    get_conversation_summary,
    clear_session_memory
)

# Page configuration
st.set_page_config(
    page_title="Intelligent RAG System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS for ultra-modern design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background-attachment: fixed;
    }
    
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(20px);
        border-radius: 30px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
        margin: 1rem;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    .main-header {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.95) 0%, rgba(255, 255, 255, 0.8) 100%);
        backdrop-filter: blur(30px);
        color: #2d3748;
        padding: 3rem 2rem;
        border-radius: 25px;
        text-align: center;
        margin-bottom: 3rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.1);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(45deg, #667eea, #764ba2, #f093fb, #f5576c);
        background-size: 400% 400%;
        animation: gradient 15s ease infinite;
        z-index: -1;
        opacity: 0.1;
    }
    
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .main-header h1 {
        font-size: 3.5rem !important;
        font-weight: 800 !important;
        margin: 0 !important;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 1rem !important;
    }
    
    .main-header p {
        font-size: 1.4rem !important;
        color: #4a5568 !important;
        margin: 0 !important;
        font-weight: 400 !important;
    }
    
    .glassmorphism {
        background: rgba(255, 255, 255, 0.25);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.18);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }
    
    .answer-container {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.9) 0%, rgba(255, 255, 255, 0.7) 100%);
        backdrop-filter: blur(30px);
        border: 1px solid rgba(102, 126, 234, 0.2);
        border-radius: 25px;
        padding: 2.5rem;
        margin: 2rem 0;
        position: relative;
        overflow: hidden;
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    
    .answer-container:hover {
        transform: translateY(-2px);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.15);
    }
    
    .answer-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #667eea, #764ba2, #f093fb);
        background-size: 200% 100%;
        animation: shimmer 3s ease-in-out infinite;
    }
    
    @keyframes shimmer {
        0%, 100% { background-position: -200% 0; }
        50% { background-position: 200% 0; }
    }
    
    .flashcard-container {
        background: linear-gradient(135deg, rgba(255, 193, 7, 0.1) 0%, rgba(255, 193, 7, 0.05) 100%);
        backdrop-filter: blur(30px);
        border: 2px solid rgba(255, 193, 7, 0.3);
        border-radius: 25px;
        padding: 2.5rem;
        margin: 2rem 0;
        position: relative;
        overflow: hidden;
        box-shadow: 0 15px 35px rgba(255, 193, 7, 0.1);
    }
    
    .flashcard-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #ffc107, #fd7e14);
    }
    
    .sources-container {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(30px);
        border-radius: 25px;
        padding: 2.5rem;
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        margin: 2rem 0;
    }
    
    .source-item {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.8) 0%, rgba(248, 249, 250, 0.8) 100%);
        backdrop-filter: blur(20px);
        border-left: 5px solid #667eea;
        padding: 2rem;
        margin: 1.5rem 0;
        border-radius: 20px;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.08);
        transition: all 0.3s ease;
    }
    
    .source-item:hover {
        transform: translateX(5px);
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.12);
    }
    
    .source-type {
        font-weight: 700 !important;
        color: #667eea !important;
        font-size: 1.2rem !important;
        margin-bottom: 0.8rem !important;
    }
    
    .source-content {
        color: #4a5568 !important;
        font-size: 1rem !important;
        line-height: 1.6 !important;
    }
    
    .sidebar-section {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(20px);
        padding: 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.3);
        transition: all 0.3s ease;
    }
    
    .sidebar-section:hover {
        transform: translateY(-2px);
        box-shadow: 0 15px 30px rgba(0, 0, 0, 0.15);
    }
    
    .sidebar-section h3 {
        color: #2d3748 !important;
        font-weight: 700 !important;
        font-size: 1.3rem !important;
        margin-bottom: 1rem !important;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 15px !important;
        padding: 0.8rem 2rem !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3) !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 12px 30px rgba(102, 126, 234, 0.4) !important;
    }
    
    .stTextArea > div > div > textarea {
        background: rgba(255, 255, 255, 0.9) !important;
        backdrop-filter: blur(20px) !important;
        border: 2px solid rgba(102, 126, 234, 0.2) !important;
        border-radius: 15px !important;
        padding: 1.5rem !important;
        font-size: 1.1rem !important;
        transition: all 0.3s ease !important;
    }
    
    .stTextArea > div > div > textarea:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
    }
    
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.9) !important;
        backdrop-filter: blur(20px) !important;
        border-radius: 12px !important;
    }
    
    .loading-animation {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 15px;
        margin: 3rem 0;
        padding: 2rem;
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(30px);
        border-radius: 20px;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
    }
    
    .loading-dot {
        width: 16px;
        height: 16px;
        background: linear-gradient(135deg, #667eea, #764ba2);
        border-radius: 50%;
        animation: bounce 1.4s infinite ease-in-out both;
    }
    
    .loading-dot:nth-child(1) { animation-delay: -0.32s; }
    .loading-dot:nth-child(2) { animation-delay: -0.16s; }
    .loading-dot:nth-child(3) { animation-delay: 0s; }
    
    @keyframes bounce {
        0%, 80%, 100% {
            transform: scale(0);
        }
        40% {
            transform: scale(1);
        }
    }
    
    .stats-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1.5rem;
        margin: 2rem 0;
    }
    
    .stat-card {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(30px);
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    
    .stat-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.15);
    }
    
    .stat-number {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
    }
    
    .stat-label {
        color: #4a5568;
        font-weight: 500;
        font-size: 1rem;
    }
    
    .chat-bubble {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(20px);
        padding: 1.5rem;
        border-radius: 18px;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.08);
        transition: all 0.3s ease;
    }
    
    .chat-bubble:hover {
        transform: translateX(5px);
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.12);
    }
    
    .user-badge {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.3rem 1rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 0.5rem;
    }
    
    .session-info {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        border: 1px solid rgba(102, 126, 234, 0.2);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class IntelligentRAGApplication:
    """Enhanced wrapper class to encapsulate the RAG functionality for Streamlit"""
    
    def __init__(self):
        try:
            self.doc_runnable = DocRetrieverRunnable(folder_path, vector_store, text_splitter)
            self.google_search = GSearchRunnable()
            self.enhanced_rag = EnhancedRAGSystem(self.doc_runnable, llm, self.google_search)
            self.flashcard_maker = EnhancedFlashcardMaker(llm)
            print("RAG system initialized successfully")
        except Exception as e:
            print(f"Error initializing RAG system: {e}")
            raise e
    
    def process_query(self, user_prompt: str, source_preference: SourcePreference, 
                     session_id: str = "default", generate_flashcards: bool = False):
        """Process a user query and return results"""
        try:
            # Get the main result from RAG system
            result = self.enhanced_rag.invoke(
                user_prompt=user_prompt, 
                source_preference=source_preference, 
                session_id=session_id
            )
            
            # Generate flashcards if requested
            if generate_flashcards and result["answer"]:
                try:
                    answer_content = result["answer"].content if hasattr(result["answer"], 'content') else str(result["answer"])
                    flashcard_result = self.flashcard_maker.invoke(answer_content, result["sources"], session_id=session_id)
                    result["flashcards"] = flashcard_result.content if hasattr(flashcard_result, 'content') else str(flashcard_result)
                except Exception as e:
                    print(f"Error generating flashcards: {e}")
                    result["flashcards"] = f"Error generating flashcards: {str(e)}"
            
            return result
            
        except Exception as e:
            print(f"Error processing query: {e}")
            return {
                "answer": f"Error processing query: {str(e)}",
                "sources": [],
                "flashcards": None if not generate_flashcards else "Could not generate flashcards due to error"
            }

@st.cache_resource
def initialize_rag_system():
    """Initialize the RAG system with caching"""
    try:
        return IntelligentRAGApplication()
    except Exception as e:
        st.error(f"Failed to initialize RAG system: {str(e)}")
        return None

def display_statistics():
    """Display session statistics"""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    total_queries = len(st.session_state.chat_history)
    doc_queries = sum(1 for chat in st.session_state.chat_history if any(s.get('type') == SourceType.DOCUMENTS for s in chat.get('sources', [])))
    internet_queries = sum(1 for chat in st.session_state.chat_history if any(s.get('type') == SourceType.INTERNET for s in chat.get('sources', [])))
    
    st.markdown("""
    <div class="stats-container">
        <div class="stat-card">
            <div class="stat-number">{}</div>
            <div class="stat-label">Total Queries</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">{}</div>
            <div class="stat-label">Document Searches</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">{}</div>
            <div class="stat-label">Internet Searches</div>
        </div>
    </div>
    """.format(total_queries, doc_queries, internet_queries), unsafe_allow_html=True)

def display_source_information(sources):
    """Display source information in a modern formatted way"""
    if not sources:
        return
    
    st.markdown("""
    <div class="sources-container">
        <h2>📚 Sources Used</h2>
    </div>
    """, unsafe_allow_html=True)
    
    for i, source in enumerate(sources, 1):
        source_type = source['type'].value.upper()
        
        if source["type"].value == "documents":
            icon = "📄"
            source_name = "Documents"
            content = f"Files: {', '.join(source['files'])}"
        elif source["type"].value == "internet":
            icon = "🌐"
            source_name = "Internet"
            content = "Links:\n" + "\n".join([f"• {link}" for link in source['links'][:3]])
        elif source["type"].value == "general_knowledge":
            icon = "🧠"
            source_name = "AI Knowledge"
            content = "Used LLM general knowledge"
        else:
            icon = "❓"
            source_name = "Unknown"
            content = "Unknown source type"
        
        st.markdown(f"""
        <div class="source-item">
            <div class="source-type">{icon} {source_name} - Source {i}</div>
            <div class="source-content">{content.replace(chr(10), '<br>')}</div>
        </div>
        """, unsafe_allow_html=True)

def display_session_info(session_id):
    """Display session information and memory status"""
    try:
        summary = get_conversation_summary(session_id)
        if summary != "No conversation history":
            st.markdown(f"""
            <div class="session-info">
                <h4>🧠 Session Memory</h4>
                <pre style="white-space: pre-wrap; font-size: 0.9rem; color: #4a5568;">{summary}</pre>
            </div>
            """, unsafe_allow_html=True)
    except:
        pass

def display_chat_history():
    """Display chat history in a modern format"""
    if "chat_history" not in st.session_state or not st.session_state.chat_history:
        st.markdown("""
        <div class="sidebar-section">
            <h3>💬 Recent Conversations</h3>
            <p style="color: #6c757d; font-style: italic;">No conversations yet. Start by asking a question!</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    st.markdown("""
    <div class="sidebar-section">
        <h3>💬 Recent Conversations</h3>
    </div>
    """, unsafe_allow_html=True)
    
    for i, chat in enumerate(reversed(st.session_state.chat_history[-5:]), 1):
        query_preview = chat['query'][:50] + "..." if len(chat['query']) > 50 else chat['query']
        time_str = datetime.fromtimestamp(chat['timestamp']).strftime('%m/%d %H:%M')
        
        with st.expander(f"💭 {query_preview}", expanded=False):
            st.markdown(f"""
            <div class="chat-bubble">
                <div class="user-badge">You</div>
                <strong>Query:</strong> {chat['query']}<br><br>
                <strong>Answer:</strong> {chat['answer'][:150]}{'...' if len(chat['answer']) > 150 else ''}<br><br>
                <small style="color: #6c757d;">🕒 {time_str}</small>
            </div>
            """, unsafe_allow_html=True)

def main():
    # Initialize session state
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())[:8]
    
    # Ultra-modern Header with animations
    st.markdown("""
    <div class="main-header">
        <h1>🧠 Intelligent RAG System</h1>
        <p>Your AI-powered research and learning companion with memory</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Display statistics
    display_statistics()
    
    # Initialize the RAG system
    with st.spinner("🚀 Initializing Intelligent RAG System..."):
        rag_app = initialize_rag_system()
    
    if rag_app is None:
        st.error("❌ Failed to initialize the RAG system. Please check your configuration and API keys.")
        return
    
    # Sidebar configuration
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-section">
            <h3>⚙️ Configuration</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Source preference selection
        source_preference = st.selectbox(
            "📖 Source Preference",
            options=[
                SourcePreference.DOCUMENTS_ONLY,
                SourcePreference.INTERNET_ONLY,
                SourcePreference.BOTH_COMBINED
            ],
            index=0,
            format_func=lambda x: {
                SourcePreference.DOCUMENTS_ONLY: "📄 Documents Only",
                SourcePreference.INTERNET_ONLY: "🌐 Internet Only", 
                SourcePreference.BOTH_COMBINED: "🔄 Both Combined"
            }[x],
            help="Choose how the system should prioritize information sources"
        )
        
        # Session management with better UX
        col1, col2 = st.columns([2, 1])
        with col1:
            session_id = st.text_input(
                "🔑 Session ID",
                value=st.session_state.session_id,
                help="Unique identifier for this conversation session"
            )
        with col2:
            if st.button("🔄", help="Generate new session ID"):
                st.session_state.session_id = str(uuid.uuid4())[:8]
                st.rerun()
        
        # Advanced options
        with st.expander("🔧 Advanced Options"):
            generate_flashcards = st.checkbox(
                "🎴 Generate Flashcards",
                value=False,
                help="Automatically generate flashcards from the answer"
            )
            
            show_session_info = st.checkbox(
                "🧠 Show Memory Info",
                value=False,
                help="Display session memory and conversation context"
            )
            
            if st.button("🗑️ Clear Session Memory", help="Clear memory for current session"):
                try:
                    result = clear_session_memory(session_id)
                    st.success(result)
                    time.sleep(1)
                    st.rerun()
                except Exception as e:
                    st.error(f"Error clearing memory: {str(e)}")
        
        st.markdown("---")
        
        # Display session info if requested
        if show_session_info:
            display_session_info(session_id)
        
        # Chat history
        display_chat_history()
        
        # Action buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear History", type="secondary", use_container_width=True):
                if "chat_history" in st.session_state:
                    del st.session_state.chat_history
                st.success("History cleared!")
                time.sleep(1)
                st.rerun()
        
        with col2:
            if st.button("📁 Export Chat", type="secondary", use_container_width=True):
                if "chat_history" in st.session_state and st.session_state.chat_history:
                    chat_data = json.dumps(st.session_state.chat_history, indent=2, default=str)
                    st.download_button(
                        "💾 Download JSON",
                        data=chat_data,
                        file_name=f"chat_history_{session_id}_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                        mime="application/json"
                    )
                else:
                    st.info("No chat history to export")
    
    # Main content area with better layout
    st.markdown("### 💬 Ask Your Question")
    
    # Query input with examples
    example_queries = [
        "What is Kyoto known for?",
        "Explain machine learning basics",
        "How does quantum computing work?",
        "Tell me about renewable energy trends",
        "My name is Sarah, what should I know about AI?"
    ]
    
    # Random example on each load
    import random
    example = random.choice(example_queries)
    
    user_query = st.text_area(
        "",
        height=120,
        placeholder=f"Ask me anything! Try: {example}",
        label_visibility="collapsed",
        help="💡 Tip: Mention your name in your first message for personalized responses!"
    )
    
    # Quick example buttons
    st.markdown("**Quick Examples:**")
    cols = st.columns(len(example_queries[:3]))
    for i, example in enumerate(example_queries[:3]):
        with cols[i]:
            if st.button(f"💡 {example[:20]}...", key=f"example_{i}", use_container_width=True):
                st.session_state.example_query = example
                st.rerun()
    
    # Use example query if selected
    if "example_query" in st.session_state:
        user_query = st.session_state.example_query
        del st.session_state.example_query
    
    # Process button with better styling
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        process_button = st.button(
            "🚀 Process Query", 
            type="primary", 
            use_container_width=True,
            disabled=not user_query.strip()
        )
    
    # Process the query
    if process_button and user_query.strip():
        # Enhanced loading animation
        loading_placeholder = st.empty()
        loading_placeholder.markdown("""
        <div class="loading-animation">
            <div class="loading-dot"></div>
            <div class="loading-dot"></div>
            <div class="loading-dot"></div>
        </div>
        <p style="text-align: center; color: #667eea; font-weight: 600; font-size: 1.1rem;">
            🔍 Processing your query with AI intelligence...
        </p>
        """, unsafe_allow_html=True)
        
        try:
            # Process the query
            start_time = time.time()
            result = rag_app.process_query(
                user_prompt=user_query,
                source_preference=source_preference,
                session_id=session_id,
                generate_flashcards=generate_flashcards
            )
            processing_time = round(time.time() - start_time, 2)
            
            loading_placeholder.empty()
            
            if result:
                # Display the answer with enhanced formatting
                answer_content = result["answer"].content if hasattr(result["answer"], 'content') else str(result["answer"])
                
                st.markdown(f"""
                <div class="answer-container">
                    <h2>🎯 Answer</h2>
                    <div style="font-size: 1.1rem; line-height: 1.7; color: #2d3748;">{answer_content}</div>
                    <div style="margin-top: 1.5rem; padding: 1rem; background: rgba(102, 126, 234, 0.1); border-radius: 10px; font-size: 0.9rem; color: #4a5568;">
                        ⏱️ Processing time: {processing_time}s | 🔗 Session: {session_id}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Display user context if available
                if result.get("user_context"):
                    st.markdown(f"""
                    <div class="session-info">
                        <h4>👤 Personalization Context</h4>
                        <p style="color: #4a5568; margin: 0;">{result['user_context'].strip()}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Display flashcards if generated
                if generate_flashcards and result.get("flashcards"):
                    flashcard_content = result.get("flashcards", "No flashcards generated")
                    
                    st.markdown(f"""
                    <div class="flashcard-container">
                        <h2>🎴 Generated Flashcards</h2>
                        <div style="font-size: 1.05rem; line-height: 1.6; color: #2d3748;">{flashcard_content}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Display source information
                display_source_information(result["sources"])
                
                # Store in session state for history
                if "chat_history" not in st.session_state:
                    st.session_state.chat_history = []
                
                st.session_state.chat_history.append({
                    "query": user_query,
                    "answer": answer_content,
                    "sources": result["sources"],
                    "timestamp": time.time(),
                    "processing_time": processing_time,
                    "session_id": session_id,
                    "source_preference": source_preference.value,
                    "had_flashcards": generate_flashcards and bool(result.get("flashcards")),
                    "user_context": result.get("user_context", "")
                })
                
                # Enhanced success message with actions
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.success(f"✅ Query processed in {processing_time}s!")
                with col2:
                    if st.button("🔄 Ask Follow-up", key="followup"):
                        st.session_state.followup_mode = True
                with col3:
                    if st.button("📋 Copy Answer", key="copy"):
                        st.code(answer_content, language="text")
                        st.info("📋 Answer displayed above for copying!")
                
                # Follow-up suggestions
                if len(st.session_state.chat_history) >= 2:
                    st.markdown("### 💡 Suggested Follow-ups")
                    follow_suggestions = [
                        f"Can you elaborate on {user_query.split()[-2:]}?",
                        "What are the practical applications of this?",
                        "Can you provide more examples?",
                        "How does this relate to current trends?"
                    ]
                    
                    cols = st.columns(2)
                    for i, suggestion in enumerate(follow_suggestions[:4]):
                        with cols[i % 2]:
                            if st.button(f"💭 {suggestion}", key=f"follow_{i}", use_container_width=True):
                                st.session_state.followup_query = suggestion
                                st.rerun()
                
        except Exception as e:
            loading_placeholder.empty()
            st.error(f"❌ An error occurred while processing your query: {str(e)}")
            with st.expander("🔍 Error Details"):
                st.exception(e)
                
                # Helpful suggestions for common errors
                st.markdown("""
                ### 🛠️ Troubleshooting Tips:
                - **API Key Issues**: Check that your OpenAI and Google API keys are properly set
                - **Document Loading**: Ensure PDF files are in the correct directory
                - **Network Issues**: Verify your internet connection for web searches
                - **Memory Issues**: Try clearing session memory or using a new session ID
                """)
    
    # Handle follow-up queries
    if "followup_query" in st.session_state:
        st.info(f"🔄 Follow-up query ready: {st.session_state.followup_query}")
        if st.button("▶️ Process Follow-up", type="primary"):
            # Process the follow-up query
            st.session_state.processing_followup = True
            del st.session_state.followup_query
            st.rerun()
        if st.button("❌ Cancel"):
            del st.session_state.followup_query
            st.rerun()
    
    # Advanced features section
    with st.expander("🔬 Advanced Features", expanded=False):
        st.markdown("""
        ### 🚀 Advanced Capabilities
        
        **Memory & Personalization:**
        - The system remembers your name and preferences across the session
        - Context from previous conversations is used to personalize responses
        - Session memory can be viewed and cleared as needed
        
        **Multi-Source Intelligence:**
        - **Documents**: Search through your uploaded PDF documents
        - **Internet**: Real-time web search for current information  
        - **Combined**: Intelligent fusion of document and web sources
        
        **Interactive Learning:**
        - Auto-generate flashcards from any answer
        - Export conversation history for review
        - Follow-up question suggestions
        """)
        
        # Document upload section
        st.markdown("### 📁 Document Management")
        uploaded_files = st.file_uploader(
            "Upload PDF documents",
            type=['pdf'],
            accept_multiple_files=True,
            help="Upload PDF files to enhance the document search capabilities"
        )
        
        if uploaded_files:
            for file in uploaded_files:
                st.success(f"📄 {file.name} ready for processing")
        
        # Session analytics
        if "chat_history" in st.session_state and st.session_state.chat_history:
            st.markdown("### 📊 Session Analytics")
            
            # Calculate stats
            total_queries = len(st.session_state.chat_history)
            avg_processing_time = sum(chat.get('processing_time', 0) for chat in st.session_state.chat_history) / total_queries
            source_usage = {}
            
            for chat in st.session_state.chat_history:
                pref = chat.get('source_preference', 'unknown')
                source_usage[pref] = source_usage.get(pref, 0) + 1
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Queries", total_queries)
            with col2:
                st.metric("Avg Processing Time", f"{avg_processing_time:.2f}s")
            with col3:
                st.metric("Session Duration", f"{(time.time() - st.session_state.chat_history[0]['timestamp'])/60:.1f}m")
            
            # Source preference chart
            if source_usage:
                st.markdown("**Source Preference Distribution:**")
                for source, count in source_usage.items():
                    percentage = (count / total_queries) * 100
                    st.progress(percentage / 100, text=f"{source.replace('_', ' ').title()}: {count} queries ({percentage:.1f}%)")
    
    # Footer with enhanced information
    st.markdown("---")
    
    # Quick stats footer
    if "chat_history" in st.session_state:
        total_q = len(st.session_state.chat_history)
        if total_q > 0:
            latest_time = datetime.fromtimestamp(st.session_state.chat_history[-1]['timestamp']).strftime('%H:%M:%S')
            st.markdown(f"""
            <div style="text-align: center; padding: 1rem; color: #6c757d; font-size: 0.9rem;">
                📊 Session Stats: {total_q} queries processed | Last activity: {latest_time} | Session: {session_id}
            </div>
            """, unsafe_allow_html=True)
    
    # Enhanced footer
    st.markdown("""
    <div style="text-align: center; padding: 2rem; margin-top: 3rem; 
                background: rgba(255, 255, 255, 0.9); backdrop-filter: blur(30px); 
                border-radius: 20px; border: 1px solid rgba(255, 255, 255, 0.2);
                box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);">
        <h4 style="color: #2d3748; margin-bottom: 1rem;">🧠 Intelligent RAG System</h4>
        <p style="color: #4a5568; margin: 0.5rem 0;">
            Powered by <strong>LangChain</strong> • <strong>OpenAI GPT-4</strong> • <strong>Chroma Vector DB</strong>
        </p>
        <p style="color: #6c757d; font-size: 0.9rem; margin: 0;">
            Built with ❤️ using <strong>Streamlit</strong> | Enhanced Memory & Personalization
        </p>
        <div style="margin-top: 1rem; padding-top: 1rem; border-top: 1px solid rgba(0,0,0,0.1);">
            <span style="color: #667eea; font-weight: 600;">Version 2.0</span> • 
            <span style="color: #4a5568;">Multi-source intelligence with persistent memory</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()