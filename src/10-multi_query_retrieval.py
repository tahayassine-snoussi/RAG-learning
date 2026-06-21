import os
import json
from typing import List
from dotenv import load_dotenv
from pydantic import BaseModel
from groq import Groq
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

# Native Groq client initialization
llm = Groq(api_key=os.getenv("GROQ_API_KEY"))

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={"batch_size": 32}
)

persistent_directory = "db/chroma_db"
db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_name="rag_collection",
    collection_metadata={"hnsw:space": "cosine"} # Note: Ensure your DB initialization doesn't have a space ("hnsw:space")
)

# 1. Define the Pydantic schema
class QueryVariations(BaseModel):
    queries: List[str]

original_query = "How does Tesla make money ?"
print(f"Original Query : {original_query}\n")

#---------------------------------------------------------
# Step 1 : generate multiple query variations using native Groq JSON mode
#---------------------------------------------------------

prompt = f"""Generate 3 different variations of this query that would help retrieve relevant documents:

Original query: {original_query}

Return exactly 3 alternative queries that rephrase or approach the same question from different angles.
You must respond with a JSON object that matches this schema:
{{
  "queries": ["query 1", "query 2", "query 3"]
}}"""

# Use native chat completion with response_format set to JSON object
response = llm.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ],
    # Enforce JSON output mode natively
    response_format={"type": "json_object"}
)

# 2. Extract and parse the text content
json_string = response.choices[0].message.content

# 3. Validate and load it into your Pydantic model
try:
    parsed_data = json.loads(json_string)
    structured_output = QueryVariations(**parsed_data)
    
    print("Generated Query Variations:")
    for query in structured_output.queries:
        print(f"- {query}")
        
except json.JSONDecodeError:
    print("Failed to decode JSON response from Groq.")



#---------------------------------------------------------
# Step 2 : search with each query variation and store results 
#---------------------------------------------------------

retriever = db.as_retriever(search_kwargs={"k":5})
all_retrieved_results = []
for i, query in enumerate(structured_output.queries,1) : 
    print(f"\n=== RESULTS FOR QUERY {i} : {query} ===")

    docs = retriever.invoke(query)
    all_retrieved_results.append(docs)

    print(f"retrieved {len(docs)} documents : \n")

    for j, doc in enumerate(docs,1) :
        print(f"Document {j} : \n")
        print(f"{doc.page_content[:100]}...\n")
    print("-"*50)

print("\n"+"="*60)
print("Multi-query retrieval completed")