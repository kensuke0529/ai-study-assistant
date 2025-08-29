import os
from dotenv import load_dotenv
import warnings
import json
from urllib3.exceptions import NotOpenSSLWarning
from typing import Dict, List, Tuple, Optional
from enum import Enum
import re

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.tools import Tool
from langchain_google_community import GoogleSearchAPIWrapper
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

warnings.simplefilter("ignore", NotOpenSSLWarning)

class SourceType(Enum):
    DOCUMENTS = "documents"
    INTERNET = "internet"
    BOTH = "both"
    GENERAL_KNOWLEDGE = "general_knowledge"

class SourcePreference(Enum):
    DOCUMENTS_ONLY = "documents_only"
    INTERNET_ONLY = "internet_only"
    BOTH_COMBINED = "both_combined"

# ------------------- Setup
load_dotenv()
folder_path = "../documents/"
persist_dir = os.path.join(folder_path, "chroma_db")

# Disable callbacks to prevent the KeyError
llm = ChatOpenAI(model="gpt-4o-mini", callbacks=[])
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

# ------------------- Embeddings and Vector Store
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=os.environ.get("OPENAI_API_KEY")
)

vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory=persist_dir
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    length_function=len
)

# ------------------- Document Retriever
class DocRetrieverRunnable(Runnable):
    def __init__(self, folder_path, vector_store, text_splitter):
        self.folder_path = folder_path
        self.vector_store = vector_store
        self.text_splitter = text_splitter

    def invoke(self, input, config=None, **kwargs):
        user_prompt = input.get("user_prompt") if isinstance(input, dict) else input

        # Check already embedded files
        try:
            existing = self.vector_store.get(include=["metadatas"])
            embedded_files = set(doc.get("source", "") for doc in existing.get("metadatas", []))
        except Exception as e:
            print(f"Warning: Could not check existing embeddings: {e}")
            embedded_files = set()

        # Find new PDFs
        try:
            pdf_files = [f for f in os.listdir(self.folder_path) if f.endswith(".pdf")]
            new_files = [f for f in pdf_files if f not in embedded_files]
        except FileNotFoundError:
            print(f"Warning: Folder {self.folder_path} not found. Creating it...")
            os.makedirs(self.folder_path, exist_ok=True)
            new_files = []

        if new_files:
            all_docs = []
            for pdf in new_files:
                try:
                    loader = PyPDFLoader(os.path.join(self.folder_path, pdf))
                    docs = loader.load()
                    for doc in docs:
                        doc.metadata["source"] = pdf
                    all_docs.extend(docs)
                except Exception as e:
                    print(f"Error loading {pdf}: {e}")

            if all_docs:
                texts = self.text_splitter.split_documents(all_docs)
                self.vector_store.add_documents(texts)
                self.vector_store.persist()
                print(f"Embedded {len(new_files)} new files into Chroma.")
        else:
            print("No new files to embed.")

        # Retrieve relevant docs
        try:
            retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})  
            docs = retriever.invoke(user_prompt)
            source_files = [d.metadata.get("source", "unknown") for d in docs]
            combined_text = "\n\n".join(d.page_content for d in docs)
            
            return {
                "text": combined_text,
                "source_files": source_files,
                "success": bool(combined_text.strip())
            }
        except Exception as e:
            print(f"Error retrieving documents: {e}")
            return {
                "text": "",
                "source_files": [],
                "success": False
            }

# ------------------- SIMPLIFIED Memory Implementation
class SimplifiedMemoryHistory(BaseChatMessageHistory):
    def __init__(self):
        self.messages: List[BaseMessage] = []
        self.user_context: Dict[str, str] = {}

    def add_messages(self, msgs: List[BaseMessage]) -> None:
        """Manually add messages and extract context"""
        for msg in msgs:
            self.messages.append(msg)
            if isinstance(msg, HumanMessage):
                self._extract_user_context(msg.content)
                print(f"DEBUG: Added human message, total messages: {len(self.messages)}")
            elif isinstance(msg, AIMessage):
                print(f"DEBUG: Added AI message, total messages: {len(self.messages)}")

    def add_user_message(self, message: str) -> None:
        """Add a user message"""
        msg = HumanMessage(content=message)
        self.messages.append(msg)
        self._extract_user_context(message)
        print(f"DEBUG: Manually added user message: '{message[:50]}...', total: {len(self.messages)}")

    def add_ai_message(self, message: str) -> None:
        """Add an AI message"""
        msg = AIMessage(content=message)
        self.messages.append(msg)
        print(f"DEBUG: Manually added AI message, total: {len(self.messages)}")

    def clear(self) -> None:
        self.messages.clear()
        self.user_context.clear()
        print("DEBUG: Cleared all messages and context")

    def _extract_user_context(self, content: str):
        """Extract and store user context like names, preferences, etc."""
        content_lower = content.lower()
        
        # Extract name patterns
        name_patterns = [
            r"my name is (\w+)",
            r"i'm (\w+)",
            r"i am (\w+)",
            r"call me (\w+)"
        ]
        
        for pattern in name_patterns:
            match = re.search(pattern, content_lower)
            if match:
                self.user_context["name"] = match.group(1).capitalize()
                print(f"DEBUG: Extracted name: {self.user_context['name']}")
                break
        
        # Extract preferences
        if "i prefer" in content_lower or "i like" in content_lower:
            self.user_context["latest_preference"] = content
            print("DEBUG: Extracted preference")

    def get_user_context_string(self) -> str:
        """Format user context for inclusion in prompts"""
        if not self.user_context:
            return ""
        
        context_parts = []
        if "name" in self.user_context:
            context_parts.append(f"User's name: {self.user_context['name']}")
        
        if context_parts:
            return "IMPORTANT USER CONTEXT: " + "; ".join(context_parts) + "\n\n"
        return ""

    def get_recent_conversation(self, max_turns: int = 3) -> str:
        """Get recent conversation for context"""
        if not self.messages:
            return ""
        
        # Take last few message pairs
        recent = self.messages[-(max_turns * 2):] if len(self.messages) > max_turns * 2 else self.messages
        
        context = "RECENT CONVERSATION:\n"
        for msg in recent:
            if isinstance(msg, HumanMessage):
                context += f"Human: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                context += f"Assistant: {msg.content[:200]}...\n" if len(msg.content) > 200 else f"Assistant: {msg.content}\n"
        
        return context + "\n"

# Global store for conversation histories
store = {}

def get_session_history(session_id: str) -> SimplifiedMemoryHistory:
    if session_id not in store:
        store[session_id] = SimplifiedMemoryHistory()
        print(f"DEBUG: Created new session history for {session_id}")
    return store[session_id]

# ------------------- Google Search Runnable
class GSearchRunnable(Runnable):  
    def __init__(self) -> None:
        super().__init__()
        self.google_api_key = os.environ.get("GOOGLE_API_KEY")
        if not self.google_api_key:
            print("Warning: GOOGLE_API_KEY not found in environment variables")
            self.search = None
        else:
            self.search = GoogleSearchAPIWrapper(google_api_key=self.google_api_key)

    def invoke(self, user_prompt):
        if not self.search:
            return {
                "text": "Google Search not available - missing API key",
                "links": [],
                "success": False
            }
        
        try:
            results = self.search.results(user_prompt, num_results=2)
            combined_text = ""
            links = []
            
            for i, r in enumerate(results):
                combined_text += f"{i+1}. {r['title']}\n{r['snippet']}\nSource: {r['link']}\n\n"
                links.append(r['link'])
            
            return {
                "text": combined_text,
                "links": links,
                "success": bool(combined_text.strip())
            }
        except Exception as e:
            print(f"Error with Google Search: {e}")
            return {
                "text": f"Google Search error: {str(e)}",
                "links": [],
                "success": False
            }

# ------------------- COMPLETELY REWRITTEN RAG System with Manual Memory
class EnhancedRAGSystem(Runnable):
    def __init__(self, doc_runnable, llm_runnable, google_search_runnable):
        self.doc_runnable = doc_runnable
        self.llm_runnable = llm_runnable
        self.google_search_runnable = google_search_runnable

    def invoke(self, user_prompt: str, source_preference: SourcePreference = SourcePreference.DOCUMENTS_ONLY, session_id: str = "default", **kwargs):
        print(f"Using source preference: {source_preference.value}")
        
        # MANUAL memory management - get session history
        session_history = get_session_history(session_id)
        
        # Get context from session
        user_context = session_history.get_user_context_string()
        conversation_context = session_history.get_recent_conversation()
        
        print(f"DEBUG: User context: '{user_context}'")
        print(f"DEBUG: Conversation available: {bool(conversation_context.strip())}")
        print(f"DEBUG: Total messages in history: {len(session_history.messages)}")
        
        # MANUALLY add current user message to history
        session_history.add_user_message(user_prompt)
        
        doc_result = None
        internet_result = None
        final_answer = None
        sources_used = []
        
        if source_preference == SourcePreference.DOCUMENTS_ONLY:
            doc_result = self.doc_runnable.invoke(user_prompt)
            if doc_result["success"]:
                final_answer = self._generate_answer_from_documents(
                    user_prompt, doc_result, user_context, conversation_context
                )
                sources_used = [{"type": SourceType.DOCUMENTS, "files": doc_result["source_files"]}]
            else:
                final_answer = self._generate_no_results_response(
                    "documents", user_prompt, user_context, conversation_context
                )
                sources_used = []
        
        elif source_preference == SourcePreference.INTERNET_ONLY:
            internet_result = self.google_search_runnable.invoke(user_prompt)
            if internet_result["success"]:
                final_answer = self._generate_answer_from_internet(
                    user_prompt, internet_result, user_context, conversation_context
                )
                sources_used = [{"type": SourceType.INTERNET, "links": internet_result["links"]}]
            else:
                final_answer = self._generate_no_results_response(
                    "internet", user_prompt, user_context, conversation_context
                )
                sources_used = []
        
        elif source_preference == SourcePreference.BOTH_COMBINED:
            doc_result = self.doc_runnable.invoke(user_prompt)
            internet_result = self.google_search_runnable.invoke(user_prompt)
            
            if doc_result["success"] and internet_result["success"]:
                final_answer = self._generate_combined_answer(
                    user_prompt, doc_result, internet_result, user_context, conversation_context
                )
                sources_used = [
                    {"type": SourceType.DOCUMENTS, "files": doc_result["source_files"]},
                    {"type": SourceType.INTERNET, "links": internet_result["links"]}
                ]
            elif doc_result["success"]:
                final_answer = self._generate_answer_from_documents(
                    user_prompt, doc_result, user_context, conversation_context
                )
                sources_used = [{"type": SourceType.DOCUMENTS, "files": doc_result["source_files"]}]
            elif internet_result["success"]:
                final_answer = self._generate_answer_from_internet(
                    user_prompt, internet_result, user_context, conversation_context
                )
                sources_used = [{"type": SourceType.INTERNET, "links": internet_result["links"]}]
            else:
                final_answer = self._generate_general_answer(
                    user_prompt, user_context, conversation_context
                )
                sources_used = [{"type": SourceType.GENERAL_KNOWLEDGE}]

        # MANUALLY add AI response to history
        response_content = final_answer.content if hasattr(final_answer, 'content') else str(final_answer)
        session_history.add_ai_message(response_content)

        return {
            "answer": final_answer,
            "sources": sources_used,
            "raw_doc_result": doc_result,
            "raw_internet_result": internet_result,
            "user_context": user_context,
            "conversation_context": conversation_context
        }

    def _generate_answer_from_documents(self, user_prompt: str, doc_result: dict, user_context: str = "", conversation_context: str = ""):
        """Generate answer using documents with manual context injection"""
        
        # Create a comprehensive prompt with all context
        system_prompt = """You are a helpful AI assistant that answers questions based on provided documents and remembers previous conversation context.

CRITICAL INSTRUCTIONS:
1. Use information from BOTH the provided documents AND the conversation history
2. When asked about information not in documents but mentioned in conversation history, refer to that history
3. Remember user details mentioned in previous messages (like names, preferences) 
4. Always be conversational and personal when appropriate
5. If asked "Do you remember my name?" check the conversation history, NOT the documents"""

        # Build the full prompt with context
        full_prompt = f"""{user_context}{conversation_context}

DOCUMENT CONTENT:
{doc_result['text']}

CURRENT USER QUESTION: {user_prompt}

Remember: Use BOTH the document content AND our conversation history to answer. If asked about personal information like names, check our conversation, not the documents."""

        # Create simple prompt template without RunnableWithMessageHistory
        template = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", full_prompt)
        ])
        
        # Invoke directly
        chain = template | self.llm_runnable
        return chain.invoke({})

    def _generate_answer_from_internet(self, user_prompt: str, internet_result: dict, user_context: str = "", conversation_context: str = ""):
        system_prompt = """You are a helpful AI assistant that uses internet search results and remembers conversation context."""
        
        full_prompt = f"""{user_context}{conversation_context}

INTERNET SEARCH RESULTS:
{internet_result['text']}

CURRENT USER QUESTION: {user_prompt}

Use both search results and conversation history. Remember personal details from our conversation."""
        
        template = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", full_prompt)
        ])
        
        chain = template | self.llm_runnable
        return chain.invoke({})

    def _generate_combined_answer(self, user_prompt: str, doc_result: dict, internet_result: dict, user_context: str = "", conversation_context: str = ""):
        system_prompt = """You are a helpful AI assistant with access to documents, internet sources, and conversation memory."""
        
        full_prompt = f"""{user_context}{conversation_context}

DOCUMENT SOURCES:
{doc_result['text']}

INTERNET SOURCES:
{internet_result['text']}

CURRENT USER QUESTION: {user_prompt}

Provide a comprehensive answer using all available information."""
        
        template = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", full_prompt)
        ])
        
        chain = template | self.llm_runnable
        return chain.invoke({})

    def _generate_general_answer(self, user_prompt: str, user_context: str = "", conversation_context: str = ""):
        system_prompt = """You are a helpful AI assistant using general knowledge while maintaining conversation continuity."""
        
        full_prompt = f"""{user_context}{conversation_context}

CURRENT USER QUESTION: {user_prompt}

Answer using general knowledge while remembering our conversation."""
        
        template = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", full_prompt)
        ])
        
        chain = template | self.llm_runnable
        return chain.invoke({})

    def _generate_no_results_response(self, source_type: str, user_prompt: str, user_context: str = "", conversation_context: str = ""):
        system_prompt = f"""You are a helpful AI assistant. No relevant {source_type} were found, but you should still use conversation history to provide a helpful response."""
        
        full_prompt = f"""{user_context}{conversation_context}

User Question: {user_prompt}

No relevant {source_type} found. Use our conversation history to still provide a helpful response."""
        
        template = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", full_prompt)
        ])
        
        chain = template | self.llm_runnable
        return chain.invoke({})

# ------------------- Enhanced Flashcard Maker with Manual Memory
class EnhancedFlashcardMaker(Runnable):
    def __init__(self, llm_runnable):
        self.llm_runnable = llm_runnable

    def invoke(self, content: str, sources: List[Dict], session_id: str = "default", **kwargs):
        # Get session context manually
        session_history = get_session_history(session_id)
        user_context = session_history.get_user_context_string()
        conversation_context = session_history.get_recent_conversation()
        
        # Create source info string
        source_info_parts = []
        for source in sources:
            if source["type"] == SourceType.DOCUMENTS:
                source_info_parts.append(f"Documents: {', '.join(source['files'])}")
            elif source["type"] == SourceType.INTERNET:
                source_info_parts.append(f"Internet: {', '.join(source['links'][:2])}...")
            elif source["type"] == SourceType.GENERAL_KNOWLEDGE:
                source_info_parts.append("General Knowledge")
        
        source_info = " | ".join(source_info_parts)
        
        system_prompt = """You are an educational flashcard creator that remembers user context and personalizes flashcards accordingly."""
        
        full_prompt = f"""{user_context}{conversation_context}

CONTENT TO MAKE FLASHCARDS FROM:
{content}

SOURCE INFORMATION: {source_info}

Create 3-5 personalized flashcards based on our conversation and the content. Remember who you're making these for.

Format each as:
**Flashcard [Number]:**
**Front:** [Question]
**Back:** [Answer]
**Source:** {source_info}"""
        
        template = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", full_prompt)
        ])
        
        # Manually add flashcard request to conversation
        session_history.add_user_message("Can you create some flashcards about what we just discussed?")
        
        chain = template | self.llm_runnable
        result = chain.invoke({})
        
        # Manually add flashcard response
        result_content = result.content if hasattr(result, 'content') else str(result)
        session_history.add_ai_message(result_content)
        
        return result

# ------------------- Memory Utilities
def get_conversation_summary(session_id: str) -> str:
    """Get a detailed summary of the conversation history"""
    history = get_session_history(session_id)
    
    summary = f"=== SESSION '{session_id}' SUMMARY ===\n"
    summary += f"Total messages: {len(history.messages)}\n"
    
    if history.user_context:
        summary += f"Extracted user context: {history.user_context}\n"
    else:
        summary += "No user context extracted\n"
    
    if history.messages:
        summary += "\nFull conversation:\n"
        for i, msg in enumerate(history.messages):
            msg_type = "Human" if isinstance(msg, HumanMessage) else "AI"
            content_preview = msg.content[:150] + "..." if len(msg.content) > 150 else msg.content
            summary += f"  {i+1}. {msg_type}: {content_preview}\n"
    else:
        summary += "No messages in conversation\n"
    
    return summary

def clear_session_memory(session_id: str) -> str:
    """Clear memory for a specific session"""
    if session_id in store:
        store[session_id].clear()
        return f"Cleared memory for session '{session_id}'"
    return f"No session '{session_id}' found"

# ------------------- Initialize components
doc_runnable = DocRetrieverRunnable(folder_path, vector_store, text_splitter)
google_search = GSearchRunnable()
enhanced_rag = EnhancedRAGSystem(doc_runnable, llm, google_search)
enhanced_flashcard_maker = EnhancedFlashcardMaker(llm)

# ------------------- Main execution with manual memory management
if __name__ == "__main__":
    session_id = "user123"
    
    # Clear any existing session for clean test
    clear_session_memory(session_id)
    print("Starting fresh conversation session...")
    
    conversations = [
        {"prompt": "Hi, my name is Alice and I'm interested in learning about Kyoto", "flashcard": False},
        {"prompt": "What is Kyoto known for?", "flashcard": False},
        {"prompt": "Can you create some flashcards about what we just discussed?", "flashcard": True},
        {"prompt": "Thanks! Do you remember my name?", "flashcard": False}
    ]
    
    source_preference = SourcePreference.DOCUMENTS_ONLY
    
    for i, conversation in enumerate(conversations, 1):
        print(f"\n{'='*60}")
        print(f"CONVERSATION TURN {i}")
        print(f"{'='*60}")
        
        user_prompt = conversation["prompt"]
        use_flashcard = conversation["flashcard"]
        
        print(f"User: {user_prompt}")
        print("-" * 40)
        
        try:
            result = enhanced_rag.invoke(user_prompt, source_preference=source_preference, session_id=session_id)
            
            if use_flashcard:
                print('=== FLASHCARD MAKER MODE ===\n')
                
                content = result["answer"].content if hasattr(result["answer"], 'content') else str(result["answer"])
                flashcard_result = enhanced_flashcard_maker.invoke(content, result["sources"], session_id=session_id)
                flashcard_content = flashcard_result.content if hasattr(flashcard_result, 'content') else str(flashcard_result)
                
                print(flashcard_content)
            else:
                print('=== ANSWER ===\n')
                content = result["answer"].content if hasattr(result["answer"], 'content') else str(result["answer"])
                print(content)
            
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
    
    # Show detailed final memory state
    print(f"\n{'='*60}")
    print("FINAL MEMORY STATE")
    print(f"{'='*60}")
    print(get_conversation_summary(session_id))