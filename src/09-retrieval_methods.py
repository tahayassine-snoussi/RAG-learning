from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
load_dotenv()

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={"batch_size": 32}
)

persistent_directory = "db/chroma_db"
db = Chroma(
    persist_directory = persistent_directory,
    embedding_function = embedding_model,
    collection_name="rag_collection",
    collection_metadata = {"hnsw :space" : "cosine"}
)

query = "How much did Microsoft pay to acquire GitHub?"
print(f"Query : {query}\n")
print("*"*100)
#----------------------------------------------------
# method 1 : basic similarity search
#----------------------------------------------------
# returns the top k most similar documents

print("=== method 1: similarity search {k=3} ===")
retriever = db.as_retriever(search_kwargs={"k":3})

docs = retriever.invoke(query)
print(f"retrieved {len(docs)} documents : \n")

for i,doc in enumerate(docs,1) : 
    print(f"Document {i} :")
    print(f"{doc.page_content}\n")


#----------------------------------------------------
# method 2 : similarity with threshold score search
#---------------------------------------------------- 
# returns documents with similarity score bigger than the threshold
print("*"*100)
print("=== method 2: similarity with score threshold ===")

retriever2 = db.as_retriever(
    search_type = "similarity_score_threshold",
    search_kwargs={
        "k":3,
        "score_threshold":0.3 # docs with simlarity score >= 0.3
    }
)

docs = retriever2.invoke(query)
print(f"retrieved {len(docs)} documents with similarity score threshold = 0.3 : \n")

for i,doc in enumerate(docs,1) : 
    print(f"Document {i} :")
    print(f"{doc.page_content}\n")


#----------------------------------------------------
# method 3 : Maximum marginal relevance "MMR"
#---------------------------------------------------- 
# finds chunks relevant to the query and then picks the most diverse ones to get differnet relevant infos not just repeated infos 
print("*"*100)
print("=== method 3: Maximum marginal relevance (MMR) ===")

retriever3 = db.as_retriever(
    search_type = "mmr",
    search_kwargs={
        "k":3,                  # final number of docs 
        "fetch_k":10,           # initial pool to select from 
        "lambda_mult" : 0.5     # 0 = max diversity , 1 = max relevance
    }
)

docs = retriever3.invoke(query)
print(f"retrieved {len(docs)} documents (lambda = 0.5) : \n")

for i,doc in enumerate(docs,1) : 
    print(f"Document {i} :")
    print(f"{doc.page_content}\n")
