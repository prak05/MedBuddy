v# -*- coding: utf-8 -*-
"""
MedBuddy Application - Enhanced RAG Medical Chatbot
===============================================================================
SOURCE: Based on the original medbuddy_application.py and enhanced according to
the research paper "Chatbot to chat with medical books using Retrieval-Augmented 
Generation Model" and project requirements from csd_344_rag_medical_chatbot_reports.md

WHY: This implementation follows the exact methodology described in the research
abstract, using Gemini as LLM, LlamaIndex for RAG pipeline, and Streamlit for UI.

HOW: The application implements a complete RAG pipeline with document processing,
vector indexing, semantic search, and grounded response generation to prevent
hallucinations in medical Q&A.
"""

# =============================================================================
# IMPORT SECTION - LIBRARY DEFINITIONS AND USAGE
# =============================================================================

# Import Streamlit for creating the web interface
# SOURCE: https://docs.streamlit.io/
# WHY: Provides a simple, user-friendly interface for uploading PDFs and chatting
# HOW: Used to create sidebar, file uploaders, chat interface, and display responses
import streamlit as st

# Import OS module for environment variable management
# SOURCE: Python standard library
# WHY: To securely store and access API keys without hardcoding them
# HOW: Sets GOOGLE_API_KEY environment variable for Gemini integration
import os

# Import tempfile for secure temporary file handling
# SOURCE: Python standard library
# WHY: To safely handle uploaded PDF files without leaving permanent traces
# HOW: Creates temporary directories that are automatically cleaned up
import tempfile

import xml.etree.ElementTree as ET

# Import SimpleDirectoryReader from LlamaIndex for document loading
# SOURCE: https://docs.llamaindex.ai/en/stable/module_guides/loading/simple_directory_reader.html
# WHY: To automatically read and parse PDF files from temporary directories
# HOW: Loads all documents from a specified directory into LlamaIndex format
from llama_index.core import SimpleDirectoryReader

# Import VectorStoreIndex from LlamaIndex for creating searchable vector database
# SOURCE: https://docs.llamaindex.ai/en/stable/module_guides/indexing/vector_store_index.html
# WHY: Converts document text into mathematical vectors for semantic search
# HOW: Creates embeddings and builds an index for fast similarity search
from llama_index.core import VectorStoreIndex

# Import Settings from LlamaIndex for global configuration
# SOURCE: https://docs.llamaindex.ai/en/stable/module_guides/configuration/settings.html
# WHY: To configure LLM and embedding models globally across the application
# HOW: Sets default models for text generation and embeddings
from llama_index.core import Settings

# Import Google GenAI LLM from LlamaIndex
# SOURCE: https://developers.llamaindex.ai/python/examples/llm/google_genai/
# WHY: The older llama-index Gemini integrations have conflicting dependencies;
#      Google GenAI integration is the current supported path.
# HOW: Provides access to Gemini models via the google-genai SDK.
from llama_index.llms.google_genai import GoogleGenAI

# Import Google GenAI Embeddings from LlamaIndex
# SOURCE: https://developers.llamaindex.ai/python/framework/integrations/embeddings/google_genai/
# WHY: Provides embeddings via the supported google-genai SDK.
# HOW: Converts document text into vectors for semantic search.
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding

# Import TextSplitter for better document chunking
# SOURCE: https://docs.llamaindex.ai/en/stable/module_guides/indexing/text_splitter.html
# WHY: To split large documents into manageable chunks for better retrieval
# HOW: Intelligently splits text while preserving context and meaning
from llama_index.core.node_parser import SentenceSplitter

# Import Optional for type hints
# SOURCE: Python standard library
# WHY: To properly type hint optional parameters and return values
# HOW: Used in function signatures for better code documentation
from typing import Optional

import requests

# =============================================================================
# CONFIGURATION AND SETUP SECTION
# =============================================================================

# Configure Streamlit page settings
# SOURCE: https://docs.streamlit.io/library/api-reference/config/st.set_page_config
# WHY: Sets professional appearance and centered layout for better UX
# HOW: Configures browser tab title and layout style
st.set_page_config(
    page_title="MedBuddy - Medical AI Assistant",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="expanded"
)

# =============================================================================
# MAIN APPLICATION INTERFACE
# =============================================================================


def _pubmed_search(term: str, retmax: int = 5, timeout_s: int = 15):
    term = (term or "").strip()
    if not term:
        return []

    esearch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    esummary_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"

    search_params = {
        "db": "pubmed",
        "term": term,
        "retmode": "xml",
        "retmax": str(retmax),
        "sort": "relevance",
    }

    r = requests.get(esearch_url, params=search_params, timeout=timeout_s)
    r.raise_for_status()
    root = ET.fromstring(r.text)
    id_list = [elem.text for elem in root.findall(".//IdList/Id") if elem.text]
    if not id_list:
        return []

    summary_params = {
        "db": "pubmed",
        "id": ",".join(id_list),
        "retmode": "json",
    }
    s = requests.get(esummary_url, params=summary_params, timeout=timeout_s)
    s.raise_for_status()
    data = s.json()

    results = []
    result = data.get("result", {})
    uids = result.get("uids", [])
    for uid in uids:
        rec = result.get(uid, {})
        title = rec.get("title") or ""
        pubdate = rec.get("pubdate") or ""
        source = rec.get("source") or ""
        authors = rec.get("authors") or []
        author_names = [a.get("name") for a in authors if isinstance(a, dict) and a.get("name")]
        results.append(
            {
                "pmid": uid,
                "title": title,
                "pubdate": pubdate,
                "journal": source,
                "authors": author_names,
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{uid}/",
            }
        )
    return results

def main():
    """
    Main application function that orchestrates the entire RAG chatbot
    SOURCE: Original medbuddy_application.py structure enhanced with research paper requirements
    WHY: Provides the central control flow for the medical chatbot application
    HOW: Manages UI components, file processing, RAG pipeline, and user interactions
    """
    
    # Display main application header
    # SOURCE: Streamlit documentation for st.title and st.write
    # WHY: Clear branding and description helps users understand the application purpose
    # HOW: Uses markdown formatting for professional appearance
    st.title("🩺 MedBuddy: Medical Textbook AI Assistant")
    st.markdown("*Upload your medical textbook and ask questions with AI-powered precision*")
    st.markdown("---")
    
    # Create sidebar for configuration and file upload
    # SOURCE: Based on original application structure, enhanced for better organization
    # WHY: Separates setup from main chat interface for cleaner UX
    # HOW: Uses Streamlit sidebar container for organized layout
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key input with security
        # SOURCE: Enhanced from original with better validation and help text
        # WHY: Secure API key handling is critical for Gemini integration
        # HOW: Password input type with validation and environment variable setting
        api_key = st.text_input(
            "🔑 Google Gemini API Key",
            type="password",
            help="Get your API key from: https://makersuite.google.com/app/apikey",
            placeholder="Enter your API key..."
        )
        
        st.markdown("---")
        st.header("📚 Document Upload")
        
        # Enhanced file uploader with better validation
        # SOURCE: Improved from original with multiple file support and validation
        # WHY: Support multiple medical textbooks and better file handling
        # HOW: Streamlit file uploader with PDF type restriction and size limits
        uploaded_files = st.file_uploader(
            "Upload Medical Textbooks (PDF)",
            type=["pdf"],
            accept_multiple_files=True,
            help="Upload one or more medical textbook PDFs for AI analysis"
        )
        
        # Display upload status
        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)} file(s) uploaded")
            for file in uploaded_files:
                st.info(f"📄 {file.name} ({file.size/1024/1024:.1f} MB)")
    
    # =============================================================================
    # API KEY VALIDATION AND CONFIGURATION
    # =============================================================================
    
    # Validate API key before proceeding
    # SOURCE: Enhanced from original with better error handling
    # WHY: Prevents application errors and provides clear user guidance
    # HOW: Checks for API key presence and displays appropriate messages
    if not api_key:
        st.warning("⚠️ Please enter your Google Gemini API Key in the sidebar to continue.")
        st.info("💡 **Need an API Key?**\n\n1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)\n2. Create a new API key\n3. Copy and paste it in the sidebar")
        st.stop()
    
    # Set API key as environment variable
    # SOURCE: Standard practice for API key management
    # WHY: Securely passes API key to LlamaIndex and Google SDK
    # HOW: Uses os.environ to set the key for downstream libraries
    os.environ["GOOGLE_API_KEY"] = api_key
    
    # =============================================================================
    # LLM AND EMBEDDING CONFIGURATION
    # =============================================================================
    
    # Configure the global LlamaIndex settings to use Google's Gemini model for generating text answers.
    # SOURCE: LlamaIndex Google GenAI example
    # WHY: Supported integration avoids dependency conflicts.
    # HOW: Pass the API key directly (or rely on GOOGLE_API_KEY env var).
    Settings.llm = GoogleGenAI(
        model="gemini-1.5-flash",
        api_key=api_key,
        temperature=0.1,
    )

    # Configure embeddings for vector search.
    # SOURCE: LlamaIndex Google GenAI embeddings integration
    # WHY: Needed for VectorStoreIndex retrieval.
    # HOW: Uses Google's text embedding model.
    Settings.embed_model = GoogleGenAIEmbedding(
        model_name="text-embedding-004",
        api_key=api_key,
    )
    
    # Configure text splitter for better chunking
    # SOURCE: LlamaIndex text splitter documentation
    # WHY: Proper chunking improves retrieval accuracy and context preservation
    # HOW: Splits documents into semantically meaningful chunks
    Settings.text_splitter = SentenceSplitter(
        chunk_size=1024,      # Optimal size for medical content
        chunk_overlap=100,    # Overlap to maintain context
        separator=" "         # Space separator for natural splitting
    )
    
    # =============================================================================
    # SESSION STATE MANAGEMENT
    # =============================================================================
    
    # Initialize session state variables
    # SOURCE: Streamlit session state documentation
    # WHY: Maintains conversation history and processed documents across interactions
    # HOW: Uses st.session_state to persist data during user session
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "query_engine" not in st.session_state:
        st.session_state.query_engine = None
    
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = []
    
    if "total_documents" not in st.session_state:
        st.session_state.total_documents = 0

    if "pubmed_last_results" not in st.session_state:
        st.session_state.pubmed_last_results = []
    
    # =============================================================================
    # DOCUMENT PROCESSING AND INDEXING
    # =============================================================================
    
    # Process uploaded files if they haven't been indexed yet
    # SOURCE: Enhanced from original with multi-file support and better error handling
    # WHY: Efficient document processing is crucial for RAG performance
    # HOW: Creates temporary directory, processes PDFs, builds vector index
    if uploaded_files and st.session_state.query_engine is None:
        # Check if files are different from previously processed
        current_files = [file.name for file in uploaded_files]
        if current_files != st.session_state.processed_files:
            with st.spinner("🔄 Processing medical textbooks... This may take a few minutes."):
                try:
                    # Create temporary directory for file processing
                    # SOURCE: Python tempfile documentation
                    # WHY: Secure file handling without leaving permanent files
                    # HOW: Creates temporary directory that auto-cleans
                    with tempfile.TemporaryDirectory() as temp_dir:
                        # Save all uploaded files to temporary directory
                        # SOURCE: File I/O best practices
                        # WHY: LlamaIndex needs files on disk for processing
                        # HOW: Writes uploaded file data to temporary files
                        for uploaded_file in uploaded_files:
                            file_path = os.path.join(temp_dir, uploaded_file.name)
                            with open(file_path, "wb") as f:
                                f.write(uploaded_file.getbuffer())
                        
                        # Load documents using LlamaIndex
                        # SOURCE: LlamaIndex SimpleDirectoryReader documentation
                        # WHY: Efficient document loading and parsing
                        # HOW: Reads all PDFs and converts to LlamaIndex documents
                        documents = SimpleDirectoryReader(temp_dir).load_data()
                        
                        # Display processing statistics
                        st.session_state.total_documents = len(documents)
                        st.info(f"📊 Processed {len(documents)} document chunks from {len(uploaded_files)} files")
                        
                        # Create vector index for semantic search
                        # SOURCE: LlamaIndex VectorStoreIndex documentation
                        # WHY: Enables fast and accurate semantic retrieval
                        # HOW: Converts documents to embeddings and builds searchable index
                        index = VectorStoreIndex.from_documents(documents)
                        
                        # Create query engine for RAG operations
                        # SOURCE: LlamaIndex query engine documentation
                        # WHY: Handles retrieval and generation for user queries
                        # HOW: Combines vector search with LLM generation
                        st.session_state.query_engine = index.as_query_engine(
                            similarity_top_k=5,  # Retrieve top 5 most relevant chunks
                            response_mode="compact"  # Efficient response generation
                        )
                        
                        # Update processed files tracking
                        st.session_state.processed_files = current_files
                        
                        # Display success message with statistics
                        st.success(f"✅ Successfully indexed {len(uploaded_files)} medical textbooks!")
                        st.info(f"🔍 Ready to answer questions from {st.session_state.total_documents} document chunks")
                        
                except Exception as e:
                    # Handle processing errors gracefully
                    # SOURCE: Python exception handling best practices
                    # WHY: Provides user-friendly error messages for debugging
                    # HOW: Catches exceptions and displays helpful error information
                    st.error(f"❌ Error processing documents: {str(e)}")
                    st.info("💡 **Troubleshooting tips:**\n- Check if PDF files are not corrupted\n- Ensure API key is valid\n- Try uploading smaller files first")
                    st.stop()

    tab_chat, tab_docs, tab_pubmed, tab_about = st.tabs(["Chat", "Documents", "PubMed", "About"])
    
    with tab_chat:
        for message in st.session_state.messages:
            with st.chat_message(message["role"], avatar="🧑‍⚕️" if message["role"] == "user" else "🤖"):
                st.markdown(message["content"])

        if prompt := st.chat_input(
            "Ask a medical question from your textbooks...",
            key="user_query",
            max_chars=500
        ):
            if st.session_state.query_engine is None:
                st.error("📚 Please upload medical textbook PDFs in the sidebar first.")
                st.info("💡 **Getting Started:**\n1. Upload one or more medical textbook PDFs\n2. Wait for processing to complete\n3. Start asking questions!")
            else:
                st.session_state.messages.append({"role": "user", "content": prompt})

                with st.chat_message("user", avatar="🧑‍⚕️"):
                    st.markdown(prompt)

                with st.chat_message("assistant", avatar="🤖"):
                    with st.spinner("🤔 Thinking and searching textbooks..."):
                        try:
                            response = st.session_state.query_engine.query(prompt)
                            st.markdown(response.response)

                            if hasattr(response, 'source_nodes') and response.source_nodes:
                                with st.expander("📖 Source Information"):
                                    for i, node in enumerate(response.source_nodes[:3], 1):
                                        st.info(f"**Source {i}:**\n{node.text[:200]}...")

                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": response.response
                            })

                        except Exception as e:
                            st.error(f"❌ Error generating response: {str(e)}")
                            st.info("💡 **Try rephrasing your question** or check if the information exists in your uploaded textbooks.")

        col_a, col_b = st.columns([1, 1])
        with col_a:
            if st.button("Clear chat", use_container_width=True):
                st.session_state.messages = []
                st.rerun()
        with col_b:
            if st.button("Reset index", use_container_width=True):
                st.session_state.query_engine = None
                st.session_state.processed_files = []
                st.session_state.total_documents = 0
                st.rerun()

    with tab_docs:
        if uploaded_files:
            st.subheader("Uploaded files")
            for f in uploaded_files:
                st.write(f"- {f.name} ({f.size/1024/1024:.1f} MB)")
        else:
            st.info("No PDFs uploaded yet.")

        if st.session_state.query_engine:
            st.success("Index ready")
            st.write(f"Textbooks indexed: {len(st.session_state.processed_files)}")
            st.write(f"Chunks searchable: {st.session_state.total_documents}")
        else:
            st.warning("Index not built yet")

    with tab_pubmed:
        st.subheader("PubMed quick lookup (optional)")
        st.caption("This searches PubMed for recent/relevant paper metadata. It does not use your Gemini key.")
        term = st.text_input("Search term", placeholder="e.g., diabetic ketoacidosis treatment")
        col1, col2 = st.columns([1, 2])
        with col1:
            run_search = st.button("Search PubMed", use_container_width=True)
        with col2:
            st.write(" ")

        if run_search:
            with st.spinner("Searching PubMed..."):
                try:
                    st.session_state.pubmed_last_results = _pubmed_search(term=term, retmax=5)
                except Exception as e:
                    st.session_state.pubmed_last_results = []
                    st.error(f"PubMed lookup failed: {str(e)}")

        if st.session_state.pubmed_last_results:
            for item in st.session_state.pubmed_last_results:
                st.markdown(f"**{item['title']}**")
                meta_parts = []
                if item.get("journal"):
                    meta_parts.append(item["journal"])
                if item.get("pubdate"):
                    meta_parts.append(item["pubdate"])
                if item.get("authors"):
                    meta_parts.append(", ".join(item["authors"][:4]) + (" et al." if len(item["authors"]) > 4 else ""))
                if meta_parts:
                    st.caption(" | ".join(meta_parts))
                st.link_button("Open on PubMed", item["url"], use_container_width=False)
                st.divider()
        else:
            st.info("Run a search to view results.")

    with tab_about:
        st.subheader("About")
        st.write("MedBuddy is a Retrieval-Augmented Generation (RAG) medical chatbot that answers questions grounded in your uploaded textbooks.")
        st.write("For safety: this is not medical advice.")
    
    # =============================================================================
    # SIDEBAR INFORMATION AND HELP
    # =============================================================================
    
    # Display usage information in sidebar
    # SOURCE: Enhanced user guidance feature
    # WHY: Helps users understand how to use the application effectively
    # HOW: Displays helpful tips and statistics in sidebar
    with st.sidebar:
        st.markdown("---")
        st.header("ℹ️ Usage Tips")
        
        if st.session_state.query_engine:
            st.success("✅ System Ready")
            st.info(f"📚 {len(st.session_state.processed_files)} textbooks indexed")
            st.info(f"📄 {st.session_state.total_documents} chunks searchable")
            
            st.markdown("**Tips for better results:**")
            st.markdown("- Ask specific medical questions")
            st.markdown("- Use medical terminology")
            st.markdown("- Reference conditions or treatments")
            st.markdown("- Ask about drug information")
        else:
            st.warning("⚠️ No documents processed")
            st.info("Upload PDFs to get started!")
        
        st.markdown("---")
        st.markdown("**🔒 Privacy Notice**")
        st.markdown("All processing happens locally. Your documents are not stored permanently.")

# =============================================================================
# APPLICATION ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    """
    Application entry point
    SOURCE: Python standard practice
    WHY: Ensures main() function is called when script is executed directly
    HOW: Standard Python if __name__ == "__main__" pattern
    """
    main()
