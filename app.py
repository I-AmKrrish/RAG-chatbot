import os
import shutil
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="wide")

# --- Custom Premium CSS Styling ---
st.markdown("""
    <style>
    /* Main background and font styling */
    .stApp {
        background-color: #0F172A;
        color: #F8FAFC;
    }
    
    /* Sidebar styling override */
    section[data-testid="stSidebar"] {
        background-color: #1E293B !important;
        border-right: 1px solid #334155;
    }
    
    /* Input elements style */
    div[data-baseweb="input"] {
        background-color: #0F172A !important;
        border: 1px solid #475569 !important;
        border-radius: 8px !important;
    }
    
    /* Modern chat bubbles */
    .user-bubble {
        background: linear-gradient(135deg, #2563EB, #1D4ED8);
        color: white;
        padding: 14px 18px;
        border-radius: 16px 16px 2px 16px;
        margin: 10px 0;
        max-width: 80%;
        margin-left: auto;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    .ai-bubble {
        background-color: #1E293B;
        color: #E2E8F0;
        padding: 14px 18px;
        border-radius: 16px 16px 16px 2px;
        margin: 10px 0;
        max-width: 80%;
        border: 1px solid #334155;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    /* Buttons */
    .stButton>button {
        width: 100%;
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease;
    }
    
    /* Custom status badge */
    .status-badge {
        background-color: #059669;
        color: white;
        padding: 4px 12px;
        border-radius: 50px;
        font-size: 0.75rem;
        font-weight: bold;
        display: inline-block;
        margin-top: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# Helper function to clean text data
def clean_text(text):
    return text.encode("utf-8", errors="surrogatepass").decode("utf-8", errors="ignore")

# --- Resource Caching ---
@st.cache_resource
def get_models():
    embedding_model = MistralAIEmbeddings()
    llm = ChatMistralAI(model="mistral-medium")
    return embedding_model, llm

embedding_model, llm = get_models()

# Initialize session states
if "messages" not in st.session_state:
    st.session_state.messages = []
if "db_ready" not in st.session_state:
    st.session_state.db_ready = False
if "document_name" not in st.session_state:
    st.session_state.document_name = "None"

# --- SIDEBAR INTERFACE ---
with st.sidebar:
    st.title("🤖 Rag chatbot")
    if st.button("+ New Chat", type="secondary"):
        st.session_state.messages = []
        st.rerun()
        
    st.markdown("---")
    
    # DB Directory text input
    db_directory = st.text_input("Database Directory", value="chroma_db")
    
    # File Uploader
    uploaded_file = st.file_uploader("Upload Knowledge Source", type=["pdf"], label_visibility="collapsed")
    
    # Side-by-side action buttons
    col1, col2 = st.columns(2)
    with col1:
        process_btn = st.button("⚡ Process", type="primary", use_container_width=True)
    with col2:
        load_btn = st.button("📂 Load DB", type="secondary", use_container_width=True)
        
    st.markdown("---")
    
    # Status Indicators
    st.markdown("### System Status")
    if st.session_state.db_ready:
        st.markdown(f'<div class="status-badge">● READY</div>', unsafe_allow_html=True)
        st.caption(f"Active file: `{st.session_state.document_name}`")
    else:
        st.markdown(f'<div class="status-badge" style="background-color: #DC2626;">● NO DATABASE</div>', unsafe_allow_html=True)
        
    st.markdown("<br><br>" * 4, unsafe_allow_html=True)
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.caption("Powered by Mistral AI + ChromaDB")

# --- MAIN CHAT INTERFACE ---
# Title Banner
st.markdown("<h2 style='color: #F8FAFC; margin-bottom: 0;'>RAG Knowledge Agent</h2>", unsafe_allow_html=True)
st.caption("Ask questions directly against your uploaded vector database contexts.")

# Handle DB Processing Trigger
if uploaded_file and process_btn:
    with st.spinner("Indexing vectors to database..."):
        try:
            temp_dir = "temp_uploads"
            os.makedirs(temp_dir, exist_ok=True)
            temp_pdf_path = os.path.join(temp_dir, uploaded_file.name)
            
            with open(temp_pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            loader = PyPDFLoader(temp_pdf_path)
            docs = loader.load()
            
            for doc in docs:
                doc.page_content = clean_text(doc.page_content) if doc.page_content else ""
            docs = [doc for doc in docs if doc.page_content.strip()]

            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks = splitter.split_documents(docs)

            if os.path.exists(db_directory):
                shutil.rmtree(db_directory)

            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embedding_model,
                persist_directory=db_directory
            )
            
            st.session_state.db_ready = True
            st.session_state.document_name = uploaded_file.name
            os.remove(temp_pdf_path)
            st.rerun()
            
        except Exception as e:
            st.error(f"Error indexing: {e}")

# Handle Existing DB Loading Trigger
if load_btn:
    if os.path.exists(db_directory):
        st.session_state.db_ready = True
        st.session_state.document_name = "Loaded from Local Path"
        st.rerun()
    else:
        st.sidebar.error("Directory not found!")

# Render Chat History or Empty State Dashboard
if st.session_state.db_ready:
    # Set up retriever
    vectorstore = Chroma(persist_directory=db_directory, embedding_function=embedding_model)
    retriever = vectorstore.as_retriever(
        search_type="mmr", 
        search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5}
    )
    
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant. Use ONLY the provided context to answer the question. If the answer is not in the context, say: 'I could not find the answer in the given documents'"),
        ("human", "Context:\n{context}\n\nQuestion: {question}")
    ])

    # Dynamic Chat Container
    chat_container = st.container()
    with chat_container:
        for idx, msg in enumerate(st.session_state.messages):
            if msg["role"] == "user":
                st.markdown(f'<div class="user-bubble">{msg["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="ai-bubble"><b>RAG Response:</b><br>{msg["content"]}</div>', unsafe_allow_html=True)
                # Expandable sources directly attached underneath the specific response block
                if "sources" in msg:
                    with st.expander("📄 View verified context references", expanded=False):
                        for s_idx, source in enumerate(msg["sources"]):
                            st.markdown(f"**Chunk {s_idx + 1}:**\n```\n{source}\n```")

    # Fixed bottom chat input form interface
    if query := st.chat_input("Type a message to Rag..."):
        # Append User Message
        st.session_state.messages.append({"role": "user", "content": query})
        
        # Get response chunks from database
        docs = retriever.invoke(query)
        context_str = "\n\n".join([doc.page_content for doc in docs])
        sources_list = [doc.page_content for doc in docs]
        
        # Run standard LLM Chain execution 
        final_prompt = prompt_template.invoke({"context": context_str, "question": query})
        response = llm.invoke(final_prompt)
        
        # Save Assistant reply metadata structured with exact context blocks
        st.session_state.messages.append({
            "role": "assistant", 
            "content": response.content,
            "sources": sources_list
        })
        st.rerun()
else:
    # Beautiful empty placeholder landing layout
    st.markdown("<br>" * 3, unsafe_allow_html=True)
    st.info("👋 Workspace Idle. To start, please upload a knowledge book PDF or load an existing path vector index via the sidebar panels.")