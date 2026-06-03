DocMind — AI Document Chat

DocMind is a Retrieval-Augmented Generation (RAG) chatbot that lets you 
upload any PDF document and ask questions about it in natural language.

How it works
1. Upload a PDF — the document is loaded, cleaned, and split into chunks
2. Chunks are embedded using Mistral AI and stored in a ChromaDB vector store
3. When you ask a question, relevant chunks are retrieved using MMR search
4. Mistral AI generates an answer strictly based on the retrieved context

Tech Stack
- **LangChain** — document loading, text splitting, RAG pipeline
- **Mistral AI** — embeddings + language model (mistral-medium)
- **ChromaDB** — local vector store for storing and retrieving embeddings
- **Streamlit** — web UI with PDF upload, chat interface, and source viewer
- **PyPDF** — PDF parsing

Features
- 📄 Upload any PDF and process it instantly
- 💬 Chat with your document in natural language
- 📎 View source passages that the answer was drawn from
- 💾 Save and reload vector databases to avoid re-processing
- 🔍 MMR search for diverse, relevant context retrieval
