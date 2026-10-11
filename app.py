import os
import warnings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


os.environ["GROQ_API_KEY"] = "ADD your API KEY"

def main():
    file_path = "chat.txt"
    if not os.path.exists(file_path):
        print(f"Error: '{file_path}' not found. Please make sure the file exists in the current directory.")
        return
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            chat_content = f.read()
        docs = [Document(page_content=chat_content)]
        print("Chat loaded successfully!")
    except Exception as e:
        print(f"File Load Error: {e}")
        return
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory="./chroma_db")

    llm = ChatGroq(
        model_name="openai/gpt-oss-120b",
        temperature=0.3
    )

    system_prompt = (
        "You are Lisa, a smart and witty AI assistant analyzing a WhatsApp chat history.\n"
        "RULES:\n"
        "1. Read the provided CONTEXT very carefully (it contains Marathi, Hinglish, and English).\n"
        "2. Answer the user's question accurately based on the chat history.\n"
        "3. If the answer is present, reply naturally like a close friend in Hinglish/Marathi.\n"
        "4. If the info is NOT in the context, say: 'Bhai, chats mein iska koi zikr nahi hai.'\n\n"
        "CONTEXT:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    rag_chain = (
        {"context": vectorstore.as_retriever(), "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    while True:
        user_input = input("👉 Tum: ")
        if user_input.lower() in ['exit', 'quit']:
            print("good bye 👋")
            break
           
        print("⏳ Lisa is thinking...")
        try:
            response = rag_chain.invoke(user_input)
            print(f"💡 AI: {response.strip()}\n")
        except Exception as e:
            print(f"❌ Error: {e}\n")
    
if __name__ == "__main__":
    main()    