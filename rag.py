import os
from dotenv import load_dotenv
import warnings
import json
from urllib3.exceptions import NotOpenSSLWarning
from typing import Dict, List, Tuple, Optional
from enum import Enum

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
    # DOCUMENTS_FIRST = "documents_first"  # Try documents first, fallback to internet
    # INTERNET_FIRST = "internet_first"    # Try internet first, fallback to documents
    BOTH_COMBINED = "both_combined"      # Use both sources and combine them

# ------------------- Setup
load_dotenv()
folder_path = "../documents/"
persist_dir = os.path.join(folder_path, "chroma_db")

llm = ChatOpenAI(model="gpt-4o-mini")  
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

# ------------------- Enhanced Prompt Templates
class EnhancedPromptTemplates:
    def __init__(self, prompt_file="./prompts/prompts.json"):
        try:
            with open(prompt_file, "r") as f:
                prompts_data = json.load(f)
                # Convert the prompt data to the expected format
                self.prompts = {}
                for key, value in prompts_data.items():
                    self.prompts[key] = {
                        "system_message": value["system_message"],
                        "user_message": value["user_message"]
                    }
        except FileNotFoundError:
            # Default prompts if file not found
            self.prompts = {
                "document_only": {
                    "system_message": "You are a helpful AI assistant. Answer the user's question based ONLY on the provided document content. Always cite specific information and indicate which document it comes from when possible.",
                    "user_message": "Document Content:\n{document}\n\nUser Question: {user_prompt}\n\nInstructions:\n1. Answer based ONLY on the document content above\n2. Cite specific passages when making claims\n3. If the document doesn't contain enough information, say so clearly\n4. Indicate which document specific information comes from"
                },
                "internet_only": {
                    "system_message": "You are a helpful AI assistant. Answer the user's question based on the provided internet search results. Always cite the sources and indicate where each piece of information comes from.",
                    "user_message": "Internet Search Results:\n{document}\n\nUser Question: {user_prompt}\n\nInstructions:\n1. Answer based on the search results above\n2. Cite specific sources for your claims\n3. Indicate the reliability and recency of information when possible\n4. Mention if information is limited or if more sources would be helpful"
                },
                "combined_sources": {
                    "system_message": "You are a helpful AI assistant with access to both document sources and internet sources. Provide a comprehensive answer while clearly distinguishing between different types of sources. Always indicate what information comes from documents vs internet sources.",
                    "user_message": "DOCUMENT SOURCES:\n{document_content}\n\nINTERNET SOURCES:\n{internet_content}\n\nUser Question: {user_prompt}\n\nInstructions:\n1. Provide a comprehensive answer using information from both sources\n2. Clearly label information as coming from 'Documents' or 'Internet sources'\n3. Compare and contrast information when there are differences\n4. Prioritize document sources for factual claims when they conflict with internet sources\n5. Use internet sources to provide additional context or recent updates"
                },
                "flashcard_maker": {
                    "system_message": "You are an educational flashcard creator. Create clear, concise flashcards from the provided content. Always include source attribution so students know where the information comes from.",
                    "user_message": "Content: {document}\n\nSource Information: {source_info}\n\nInstructions:\nCreate 3-5 high-quality flashcards from this content. Format each flashcard as follows:\n\n**Flashcard [Number]:**\n**Front:** [Clear, specific question]\n**Back:** [Accurate, concise answer]\n**Source:** {source_info}\n\nGuidelines:\n1. Make questions specific and testable\n2. Keep answers concise but complete\n3. Focus on key concepts and important facts\n4. Ensure questions test understanding, not just memorization\n5. Include the source information with each flashcard"
                },
                "general_knowledge": {
                    "system_message": "You are a helpful AI assistant using your general knowledge to answer questions. Be clear about the limitations of your knowledge and suggest where users might find more current or specific information.",
                    "user_message": "User Question: {user_prompt}\n\nNote: This answer is based on general AI knowledge. For the most current information or specific details, consider consulting recent sources or official documentation."
                }
            }
    
    def get_template(self, template_type: str, include_history: bool = False) -> ChatPromptTemplate:
        if template_type not in self.prompts:
            raise KeyError(f"Template '{template_type}' not found")
        
        prompt_config = self.prompts[template_type]
        system_msg = SystemMessagePromptTemplate.from_template(prompt_config["system_message"])
        
        messages = [system_msg]
        
        # Add message history placeholder if requested
        if include_history:
            messages.append(MessagesPlaceholder(variable_name="history"))
        
        user_msg = HumanMessagePromptTemplate.from_template(prompt_config["user_message"])
        messages.append(user_msg)
        
        return ChatPromptTemplate.from_messages(messages)

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

# ------------------- Fixed Memory Implementation
class InMemoryHistory(BaseChatMessageHistory):
    def __init__(self):
        self.messages: List[BaseMessage] = []

    def add_messages(self, msgs: List[BaseMessage]) -> None:
        self.messages.extend(msgs)

    def clear(self) -> None:
        self.messages.clear()

# Global store for conversation histories
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryHistory()
    return store[session_id]

# ------------------- Enhanced RAG System
class EnhancedRAGSystem(Runnable):
    def __init__(self, doc_runnable, llm_runnable, google_search_runnable):
        self.doc_runnable = doc_runnable
        self.llm_runnable = llm_runnable
        self.google_search_runnable = google_search_runnable
        self.prompt_templates = EnhancedPromptTemplates()

    def invoke(self, user_prompt: str, source_preference: SourcePreference = SourcePreference.DOCUMENTS_ONLY, session_id: str = "default", **kwargs):
        doc_result = None
        internet_result = None
        final_answer = None
        sources_used = []
        
        print(f"Using source preference: {source_preference.value}")
        
        if source_preference == SourcePreference.DOCUMENTS_ONLY:
            doc_result = self.doc_runnable.invoke(user_prompt)
            if doc_result["success"]:
                final_answer = self._generate_answer_from_documents(user_prompt, doc_result, session_id)
                sources_used = [{"type": SourceType.DOCUMENTS, "files": doc_result["source_files"]}]
            else:
                final_answer = "No relevant documents found for your query."
                sources_used = []
        
        elif source_preference == SourcePreference.INTERNET_ONLY:
            internet_result = self.google_search_runnable.invoke(user_prompt)
            if internet_result["success"]:
                final_answer = self._generate_answer_from_internet(user_prompt, internet_result, session_id)
                sources_used = [{"type": SourceType.INTERNET, "links": internet_result["links"]}]
            else:
                final_answer = "No relevant internet results found for your query."
                sources_used = []
        
        # elif source_preference == SourcePreference.DOCUMENTS_FIRST:
        #     doc_result = self.doc_runnable.invoke(user_prompt)
        #     if doc_result["success"]:
        #         final_answer = self._generate_answer_from_documents(user_prompt, doc_result, session_id)
        #         sources_used = [{"type": SourceType.DOCUMENTS, "files": doc_result["source_files"]}]
        #     else:
        #         # Fallback to internet
        #         print("No documents found, falling back to internet search...")
        #         internet_result = self.google_search_runnable.invoke(user_prompt)
        #         if internet_result["success"]:
        #             final_answer = self._generate_answer_from_internet(user_prompt, internet_result, session_id)
        #             sources_used = [{"type": SourceType.INTERNET, "links": internet_result["links"]}]
        #         else:
        #             final_answer = self._generate_general_answer(user_prompt, session_id)
        #             sources_used = [{"type": SourceType.GENERAL_KNOWLEDGE}]
        
        # elif source_preference == SourcePreference.INTERNET_FIRST:
        #     internet_result = self.google_search_runnable.invoke(user_prompt)
        #     if internet_result["success"]:
        #         final_answer = self._generate_answer_from_internet(user_prompt, internet_result, session_id)
        #         sources_used = [{"type": SourceType.INTERNET, "links": internet_result["links"]}]
        #     else:
        #         # Fallback to documents
        #         print("Internet search failed, falling back to documents...")
        #         doc_result = self.doc_runnable.invoke(user_prompt)
        #         if doc_result["success"]:
        #             final_answer = self._generate_answer_from_documents(user_prompt, doc_result, session_id)
        #             sources_used = [{"type": SourceType.DOCUMENTS, "files": doc_result["source_files"]}]
        #         else:
        #             final_answer = self._generate_general_answer(user_prompt, session_id)
        #             sources_used = [{"type": SourceType.GENERAL_KNOWLEDGE}]
        
        elif source_preference == SourcePreference.BOTH_COMBINED:
            doc_result = self.doc_runnable.invoke(user_prompt)
            internet_result = self.google_search_runnable.invoke(user_prompt)
            
            if doc_result["success"] and internet_result["success"]:
                final_answer = self._generate_combined_answer(user_prompt, doc_result, internet_result, session_id)
                sources_used = [
                    {"type": SourceType.DOCUMENTS, "files": doc_result["source_files"]},
                    {"type": SourceType.INTERNET, "links": internet_result["links"]}
                ]
            elif doc_result["success"]:
                final_answer = self._generate_answer_from_documents(user_prompt, doc_result, session_id)
                sources_used = [{"type": SourceType.DOCUMENTS, "files": doc_result["source_files"]}]
            elif internet_result["success"]:
                final_answer = self._generate_answer_from_internet(user_prompt, internet_result, session_id)
                sources_used = [{"type": SourceType.INTERNET, "links": internet_result["links"]}]
            else:
                final_answer = self._generate_general_answer(user_prompt, session_id)
                sources_used = [{"type": SourceType.GENERAL_KNOWLEDGE}]

        return {
            "answer": final_answer,
            "sources": sources_used,
            "raw_doc_result": doc_result,
            "raw_internet_result": internet_result
        }

    def _generate_answer_from_documents(self, user_prompt: str, doc_result: dict, session_id: str = "default"):
        template = self.prompt_templates.get_template("document_only", include_history=True)
        
        # Create a runnable with message history
        runnable_with_history = RunnableWithMessageHistory(
            template | self.llm_runnable,
            get_session_history,
            history_messages_key="history"
        )
        
        return runnable_with_history.invoke(
            {
                "document": doc_result["text"],
                "user_prompt": user_prompt
            },
            config={"configurable": {"session_id": session_id}}
        )

    def _generate_answer_from_internet(self, user_prompt: str, internet_result: dict, session_id: str = "default"):
        template = self.prompt_templates.get_template("internet_only", include_history=True)
        
        runnable_with_history = RunnableWithMessageHistory(
            template | self.llm_runnable,
            get_session_history,
            history_messages_key="history"
        )
        
        return runnable_with_history.invoke(
            {
                "document": internet_result["text"],
                "user_prompt": user_prompt
            },
            config={"configurable": {"session_id": session_id}}
        )

    def _generate_combined_answer(self, user_prompt: str, doc_result: dict, internet_result: dict, session_id: str = "default"):
        template = self.prompt_templates.get_template("combined_sources", include_history=True)
        
        runnable_with_history = RunnableWithMessageHistory(
            template | self.llm_runnable,
            get_session_history,
            history_messages_key="history"
        )
        
        return runnable_with_history.invoke(
            {
                "document_content": doc_result["text"],
                "internet_content": internet_result["text"],
                "user_prompt": user_prompt
            },
            config={"configurable": {"session_id": session_id}}
        )

    def _generate_general_answer(self, user_prompt: str, session_id: str = "default"):
        template = self.prompt_templates.get_template("general_knowledge", include_history=True)
        
        runnable_with_history = RunnableWithMessageHistory(
            template | self.llm_runnable,
            get_session_history,
            history_messages_key="history"
        )
        
        return runnable_with_history.invoke(
            {
                "user_prompt": user_prompt
            },
            config={"configurable": {"session_id": session_id}}
        )

# ------------------- Enhanced Flashcard Maker
class EnhancedFlashcardMaker(Runnable):
    def __init__(self, llm_runnable):
        self.llm_runnable = llm_runnable
        self.prompt_templates = EnhancedPromptTemplates()

    def invoke(self, content: str, sources: List[Dict], **kwargs):
        # Create source info string
        source_info_parts = []
        for source in sources:
            if source["type"] == SourceType.DOCUMENTS:
                source_info_parts.append(f"Documents: {', '.join(source['files'])}")
            elif source["type"] == SourceType.INTERNET:
                source_info_parts.append(f"Internet: {', '.join(source['links'][:2])}...")  # Show first 2 links
            elif source["type"] == SourceType.GENERAL_KNOWLEDGE:
                source_info_parts.append("General Knowledge")
        
        source_info = " | ".join(source_info_parts)
        
        template = self.prompt_templates.get_template("flashcard_maker")
        formatted_prompt = template.format_messages(
            document=content,
            source_info=source_info
        )
        return self.llm_runnable.invoke(formatted_prompt)

# ------------------- Initialize components
doc_runnable = DocRetrieverRunnable(folder_path, vector_store, text_splitter)
google_search = GSearchRunnable()
enhanced_rag = EnhancedRAGSystem(doc_runnable, llm, google_search)
enhanced_flashcard_maker = EnhancedFlashcardMaker(llm)

# ------------------- Main execution
if __name__ == "__main__":
    user_prompt = "What is Kyoto known for?"
    use_flashcard = False
    session_id = "user123"  # You can change this for different conversations
    
    # Choose your source preference:
    source_preference = SourcePreference.DOCUMENTS_ONLY  # Try this with different options!
    
    try:
        print(f"Processing query: '{user_prompt}'")
        print(f"Source preference: {source_preference.value}")
        print("="*60)
        
        result = enhanced_rag.invoke(user_prompt, source_preference=source_preference, session_id=session_id)
        
        if use_flashcard:
            print('=== FLASHCARD MAKER MODE ===\n')
            
            # Get content from the answer
            content = result["answer"].content if hasattr(result["answer"], 'content') else str(result["answer"])
            
            # Create flashcards with source attribution
            flashcard_result = enhanced_flashcard_maker.invoke(content, result["sources"])
            flashcard_content = flashcard_result.content if hasattr(flashcard_result, 'content') else str(flashcard_result)
            
            print(flashcard_content)
        else:
            print('=== ANSWER ===\n')
            content = result["answer"].content if hasattr(result["answer"], 'content') else str(result["answer"])
            print(content)
        
        print("\n" + "="*60)
        print("=== SOURCE INFORMATION ===")
        
        for i, source in enumerate(result["sources"], 1):
            print(f"\nSource {i}: {source['type'].value.upper()}")
            if source["type"] == SourceType.DOCUMENTS:
                print(f"  Files: {', '.join(source['files'])}")
            elif source["type"] == SourceType.INTERNET:
                print(f"  Links:")
                for link in source['links'][:2]:  # Show first 2 links
                    print(f"    - {link}")
            elif source["type"] == SourceType.GENERAL_KNOWLEDGE:
                print(f"  Used LLM general knowledge")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()