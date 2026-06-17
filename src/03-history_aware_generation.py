import os 
from langchain_chroma import Chroma # vector store for storing and querying embeddings
from dotenv import load_dotenv # load environment variables from a .env file
from langchain_community.embeddings import HuggingFaceEmbeddings
from groq import Groq
load_dotenv()
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={"batch_size": 32}
)
client = Groq(api_key=os.getenv("GROQ_API_KEY")) # initialize the Groq client with the API key from the .env file7

persistent_directory = "db/chroma_db" # directory to persist the vector store

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata= {"hnsw:space" : "cosine"},
    collection_name="rag_collection"   
)

retriever = db.as_retriever(
    search_type="similarity_score_threshold", 
    search_kwargs={
        "k": 3,
        "score_threshold":0.3  # only return chunks with a cosine similarity > 0.3
    }
)

# store the conversation history to rewrite the query if needed 
chat_history =[]

def ask_question(user_question) : 
    print ("You asked : ", user_question)

    if chat_history :
        # ask the llm to reformulate the question based on the history of the conversation 
        message = f"""Given the chat history, rewrite the new question to be standalone and searchable. Just return the rewritten question.
        Chat history: {"\n".join(chat_history)}
        New question: {user_question}"""

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",  # example model
            messages=[
                {
                    "role": "user",
                    "content": message
                }
            ]
        )
        search_question = response.choices[0].message.content.strip()
        print(f"Searching for: {search_question}")
    else :
        search_question = user_question
    
    # next step find the relevant docs 

    docs = retriever.invoke(search_question)
    print(f"Found {len(docs)} relevant documents.")
    for i, doc in enumerate(docs, 1):
        lines = doc.page_content.split('\n')[:2]
        preview = '\n'.join(lines)
        print(f"  Doc {i}: {preview}...")

    # third step : make the final llm call

    combined_input = f"""Based on the following documents, please answer this question: {search_question}

    Documents:
    {"\n".join([f"- {doc.page_content}" for doc in docs])}

    Please provide a clear, helpful answer using only the information from these documents. If you can't find the answer in the documents, say "I don't have enough information to answer that question based on the provided documents."
    """

    # step 4 : get the response from the llm

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",  # example model   
        messages=[
            {
                "role": "user",
                "content": combined_input
            }
        ]
    )
    answer = response.choices[0].message.content
    chat_history.append(f"User: {user_question}")
    chat_history.append(f"Assistant: {answer}")

    print("\n--- Generated Response ---")
    print(answer)

    return answer



def start_chat() : 
    print("Ask me questions type 'quit' to exit .")

    while True :
        question = input("\nYour question : ")

        if question.lower() == 'quit' :
            print("Goodbye!")
            break

        ask_question(question)

if __name__ == "__main__":
    start_chat() 