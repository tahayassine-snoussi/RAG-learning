from langchain_chroma import Chroma # vector store for storing and querying embeddings
from dotenv import load_dotenv # load environment variables from a .env file
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={"batch_size": 32}
)

persistent_directory = "db/chroma_db" # directory to persist the vector store

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata= {"hnsw:space" : "cosine"},
    collection_name="rag_collection"   
)

query = "How much did Microsoft pay to acquire GitHub?"
 
retriever = db.as_retriever(search_kwargs={"k": 5}) # retrieve the top 5 most relevant chunks for the query

#retriever = db.as_retriever(
#    search_type="similarity_score_threshold", 
#    search_kwargs={
#        "k": 3,
#        "score_threshold":0.3  # only return chunks with a cosine similarity > 0.3
#    }
#)

print("Collection count:", db._collection.count())
relevant_docs = retriever.invoke(query)

print(f"\nUser Query: {query}")
# Display results
print("--- Context ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")



# Synthetic Questions: 

# 1. "What was NVIDIA's first graphics accelerator called?"
# 2. "Which company did NVIDIA acquire to enter the mobile processor market?"
# 3. "What was Microsoft's first hardware product release?"
# 4. "How much did Microsoft pay to acquire GitHub?"
# 5. "In what year did Tesla begin production of the Roadster?"
# 6. "Who succeeded Ze'ev Drori as CEO in October 2008?"
# 7. "What was the name of the autonomous spaceport drone ship that achieved the first successful sea landing?"
# 8. "What was the original name of Microsoft before it became Microsoft?"