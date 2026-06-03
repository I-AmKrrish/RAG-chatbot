# load the pdf
# create into chunks
# create the embeddings 
# store into chroma


from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import Chroma, chroma 

from dotenv import load_dotenv
load_dotenv()


def clean_text(text):
    return text.encode("utf-8", errors="surrogatepass").decode("utf-8", errors="ignore")



data = PyPDFLoader("document_loaders/deeplearning.pdf")
docs = data.load()


# Step 2: Clean BEFORE splitting
for doc in docs:
    doc.page_content = clean_text(doc.page_content) if doc.page_content else ""

docs = [doc for doc in docs if doc.page_content.strip()]
print(f"Loaded {len(docs)} valid pages after cleaning")


splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 100
)


chunks = splitter.split_documents(docs)
print(f"Created {len(chunks)} chunks")
# The print just tells you how many chunks were created so you can verify the splitting worked.


embedding_model = MistralAIEmbeddings()

vectorstore = Chroma.from_documents(
    documents = chunks,
    embedding = embedding_model,
    persist_directory = "chroma_db"
)