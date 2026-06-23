from collections import defaultdict
import os
import json
from typing import List
from dotenv import load_dotenv
from pydantic import BaseModel
from groq import Groq
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv()

# Native Groq client initialization
llm = Groq(api_key=os.getenv("GROQ_API_KEY"))

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={"batch_size": 32}
)

persistent_directory = BASE_DIR / "db" / "chroma_db"
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



#---------------------------------------------------------
# Step 3 : apply reciprocal rank fusion (RRF)
#---------------------------------------------------------

print("\n"+"="*60)


def reciprocal_rank_fusion(chunk_lists, k=60, verbose=True) : 
    if verbose:
        print("\n" + "="*60)
        print("APPLYING RECIPROCAL RANK FUSION")
        print("="*60)
        print(f"\nUsing k={k}")
        print("Calculating RRF scores...\n")
    
    # Data structures for RRF calculation
    rrf_scores = defaultdict(float)  # Will store: {chunk_content: rrf_score}
    all_unique_chunks = {}  # Will store: {chunk_content: actual_chunk_object}
    
    # For verbose output - track chunk IDs
    chunk_id_map = {}
    chunk_counter = 1
    
    # Go through each retrieval result
    for query_idx, chunks in enumerate(chunk_lists, 1):
        if verbose:
            print(f"Processing Query {query_idx} results:")
        
        # Go through each chunk in this query's results
        for position, chunk in enumerate(chunks, 1):  # position is 1-indexed
            # Use chunk content as unique identifier
            chunk_content = chunk.page_content

            if chunk_content not in chunk_id_map : 
                chunk_id_map[chunk_content] = f"Chunk_{chunk_counter}"
                chunk_counter +=1
            
            chunk_id = chunk_id_map[chunk_content]
            
            # Store the chunk object (in case we haven't seen it before)
            all_unique_chunks[chunk_content] = chunk
            
            # Calculate position score: 1/(k + position)
            position_score = 1 / (k + position)

            # Add to RRF score
            rrf_scores[chunk_content] += position_score
            
            if verbose:
                print(f"  Position {position}: {chunk_id} +{position_score:.4f} (running total: {rrf_scores[chunk_content]:.4f})")
                print(f"    Preview: {chunk_content[:80]}...")
        
        if verbose:
            print()
    
    # Sort chunks by RRF score (highest first)
    sorted_chunks = sorted(
        [(all_unique_chunks[chunk_content], score) for chunk_content, score in rrf_scores.items()],
        key=lambda x: x[1],  # Sort by RRF score
        reverse=True  # Highest scores first
    )
    
    if verbose:
        print(f"✅ RRF Complete! Processed {len(sorted_chunks)} unique chunks from {len(chunk_lists)} queries.")
    
    return sorted_chunks

# Apply RRF to our retrieval results
fused_results = reciprocal_rank_fusion(all_retrieved_results, k=60, verbose=True)

# ──────────────────────────────────────────────────────────────────
# Step 4: Display Final Fused Results
# ──────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("FINAL RRF RANKING")
print("="*60)

print(f"\nTop {min(10, len(fused_results))} documents after RRF fusion:\n")

for rank, (doc, rrf_score) in enumerate(fused_results[:10], 1):
    print(f"🏆 RANK {rank} (RRF Score: {rrf_score:.4f})")
    print(f"{doc.page_content[:200]}...")
    print("-" * 50)

print(f"\n✅ RRF Complete! Fused {len(fused_results)} unique documents from {len(structured_output.queries)} query variations.")
print("\n💡 Key benefits:")
print("   • Documents appearing in multiple queries get boosted scores")
print("   • Higher positions contribute more to the final score") 
print("   • Balanced fusion using k=60 for gentle position penalties")

# ──────────────────────────────────────────────────────────────────
# Optional: Quick Usage Examples
# ──────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("USAGE EXAMPLES")
print("="*60)
