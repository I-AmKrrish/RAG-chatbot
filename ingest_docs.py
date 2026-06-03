import os
import shutil
from pathlib import Path
from dotenv import load_dotenv

# LangChain components
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

PERSIST_DIR = "chroma_db"
DOCS_FOLDERS = ["claude_docs", "document_loaders"]

def clean_text(text):
    if not text:
        return ""
    return text.encode("utf-8", errors="surrogatepass").decode("utf-8", errors="ignore")

def load_documents():
    documents = []
    
    for folder in DOCS_FOLDERS:
        folder_path = Path(folder)
        if not folder_path.exists():
            print(f"Folder '{folder}' does not exist, skipping.")
            continue
            
        print(f"Scanning folder '{folder}'...")
        
        # Recurse and find files
        for file_path in folder_path.rglob("*"):
            if not file_path.is_file():
                continue
                
            suffix = file_path.suffix.lower()
            
            # Load PDFs
            if suffix == ".pdf":
                print(f"Loading PDF: {file_path}")
                try:
                    loader = PyPDFLoader(str(file_path))
                    pdf_docs = loader.load()
                    # Clean page content and set source metadata
                    for doc in pdf_docs:
                        doc.page_content = clean_text(doc.page_content)
                        # Ensure standard metadata format
                        doc.metadata["source"] = str(file_path.as_posix())
                    documents.extend(pdf_docs)
                except Exception as e:
                    print(f"Error loading PDF {file_path}: {e}")
                    
            # Load Markdown / Text files
            elif suffix in [".md", ".txt"]:
                # Skip helper notes or script files if any
                if file_path.name in ["notes.txt", "requirements.txt"]:
                    continue
                print(f"Loading Text/Markdown: {file_path}")
                try:
                    loader = TextLoader(str(file_path), encoding="utf-8")
                    text_docs = loader.load()
                    for doc in text_docs:
                        doc.page_content = clean_text(doc.page_content)
                        doc.metadata["source"] = str(file_path.as_posix())
                    documents.extend(text_docs)
                except Exception as e:
                    print(f"Error loading text/markdown {file_path}: {e}")
                    
    return documents

def main():
    print("Starting document ingestion process...")
    
    # 1. Load all documents
    docs = load_documents()
    # Filter out empty page contents
    docs = [d for d in docs if d.page_content.strip()]
    print(f"\nLoaded total of {len(docs)} documents/pages.")
    
    if not docs:
        print("No documents loaded. Ingestion cancelled.")
        return
        
    # 2. Split documents into chunks
    print("Splitting documents into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(docs)
    print(f"Created {len(chunks)} chunks.")
    
    # 3. Create or rebuild Chroma DB
    if os.path.exists(PERSIST_DIR):
        print(f"Removing old vector database at '{PERSIST_DIR}'...")
        try:
            shutil.rmtree(PERSIST_DIR)
        except Exception as e:
            print(f"Warning: Failed to delete old DB directory: {e}. Attempting to overwrite.")
            
    print("Creating new Chroma vector store...")
    embedding_model = MistralAIEmbeddings()
    
    try:
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            persist_directory=PERSIST_DIR
        )
        print(f"Successfully created and persisted Chroma DB at '{PERSIST_DIR}'.")
    except Exception as e:
        print(f"Error creating vector store: {e}")

if __name__ == "__main__":
    main()
