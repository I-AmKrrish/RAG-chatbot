from dotenv import load_dotenv
load_dotenv()

from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_mistralai import ChatMistralAI 
from langchain_core.prompts import ChatPromptTemplate

embedding_model = MistralAIEmbeddings()

vectorstore = Chroma(
    persist_directory = "chroma_db",
    embedding_function = embedding_model,
)


retriever = vectorstore.as_retriever(
    search_type = "mmr",
    search_kwargs = {
        "k": 4, 
        "fetch_k": 10,
        "lambda_mult": 0.5, #0-1 least diverse results to most diverse results
    }
)


llm = ChatMistralAI(model = "mistral-medium-3-5")


#prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """You are a helpful AI assistant
        use ONLY the provided context to answer the question.
        If the answer is not in the context, say: "I could not find the answer in the given documents" """
        ),
        ("human",
        """Context: {context}
        Question: {question}
        """
        ),
    ]
)

print("RAG system is ready")

print("Press 0 to exit")

while True:
    query = input("YOU: ")
    if query == "0":
        break

    docs = retriever.invoke(query)

    context = "\n\n".join(
        [doc.page_content for doc in docs]
        )


    final_prompt = prompt.invoke({
        "context": context,
        "question": query,
    })

    reponse = llm.invoke(final_prompt)

    print(f"\n \n AI: , {reponse.content}")


    