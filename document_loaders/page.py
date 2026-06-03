from langchain_community.document_loaders import WebBaseLoader, url

url = "https://www.apple.com/in/macbook-neo/"

data = WebBaseLoader(url)

docs = data.load()

print(docs[0].page_content)