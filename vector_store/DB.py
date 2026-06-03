# in this file we havent crated chunks and thats why we had to cleant the document first

from dotenv import load_dotenv
load_dotenv()

from langchain_community.vectorstores import Chroma
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader

def clean_text(text):
    # encode with surrogatepass to handle them, then decode ignoring errors
    return text.encode("utf-8", errors="surrogatepass").decode("utf-8", errors="ignore")


data = PyPDFLoader("document_loaders/deeplearning.pdf")
docs = data.load()

for doc in docs:
    doc.page_content = str(doc.page_content) if doc.page_content is not None else ""

# Clean surrogate characters from all docs
for doc in docs:
    doc.page_content = clean_text(doc.page_content)

docs = [doc for doc in docs if doc.page_content.strip()]
print(f"Loaded {len(docs)} valid pages")


embedding_model = MistralAIEmbeddings()

vectorstore = Chroma.from_documents(
    documents = docs,
    embedding = embedding_model,
    persist_directory = "chroma_db"
)


result = vectorstore.similarity_search("how does RNN work", k=2)

for r in result:
    print(r.page_content)
    print(r.metadata)

retriver = vectorstore.as_retriever()


docs = retriver.invoke("explain RNN")

for d in docs:
    print(d.page_content)