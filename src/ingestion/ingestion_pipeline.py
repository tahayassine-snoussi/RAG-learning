import os 
from langchain_community.document_loaders import TextLoader, DirectoryLoader #read text files from a directory
from langchain_text_splitters import CharacterTextSplitter # split text into chunks
from langchain_openai import OpenAIEmbeddings # generate embeddings using OpenAI's API
from langchain_chroma import Chroma # vector store for storing and querying embeddings
from dotenv import load_dotenv # load environment variables from a .env file
from langchain_community.embeddings import HuggingFaceEmbeddings


embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={"batch_size": 32}
)
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY") # load the GROQ API key from the .env file


def load_files(directory_path):
    """
    Load text files from a directory and return a list of documents.
    """
    loader = DirectoryLoader(
        directory_path,
        glob="*.txt", 
        loader_cls=lambda path: TextLoader(path, encoding="utf-8")
        )
    documents = loader.load()

    if len(documents) == 0:
        raise ValueError(f"No text files found in the directory: {directory_path}")
    
    for i , doc in enumerate(documents[:2]) : 
        print(f"""\ndocument {i+1} : \n 
              Source : {doc.metadata['source']} \n 
              Content length : {len(doc.page_content)} chars \n 
              Content preview : {doc.page_content[:50]}... \n 
              Metadata : {doc.metadata} \n """)
 
    return documents

def chunk_documents(documents, chunk_size=1000, chunk_overlap=0):
    """
    Split documents into smaller chunks.
    """
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(documents)

    if chunks:
        for i, chunk in enumerate(chunks[:5]):
            print(f"\n--- Chunk {i+1} ---")
            print(f"Source: {chunk.metadata['source']}")
            print(f"Length: {len(chunk.page_content)} characters")
            print(f"Content:")
            print(chunk.page_content)
            print("-" * 50)
        
        if len(chunks) > 5:
            print(f"\n... and {len(chunks) - 5} more chunks") 
        
    return chunks

def create_vector_store(chunks, persist_directory="db/chroma_db"):
    """
    Create a vector store from the document chunks and persist it to disk.
    """

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_name="rag_collection",   # ADD THIS
        collection_metadata={"hnsw:space": "cosine"}
    )
    
    print (f"finished creating the vector store and persisting it to disk at {persist_directory}")
    return vector_store


def main() : 
    
    print("************************\n Starting the ingestion pipeline...\n************************")

    # 1. loading the files 
    print("****\n Loading files...\n****")
    documents = load_files("./data")

    # 2 chuncking the files
    print("****\n Chunking files...\n****")
    chunks = chunk_documents(documents)

    # 3 Embedding the chunks and storing them in a vector DB
    print("****\n Embedding files...\n****")
    vector_store = create_vector_store(chunks)




if __name__ == "__main__":
    main()





# documents = [
#    Document(
#        page_content="Google LLC is an American multinational corporation and technology company focusing on online advertising, search engine technology, cloud computing, computer software, quantum computing, e-commerce, consumer electronics, and artificial intelligence (AI).",
#        metadata={'source': 'docs/google.txt'}
#    ),
#    Document(
#        page_content="Microsoft Corporation is an American multinational corporation and technology conglomerate headquartered in Redmond, Washington.",
#        metadata={'source': 'docs/microsoft.txt'}
#    ),
#    Document(
#        page_content="Nvidia Corporation is an American technology company headquartered in Santa Clara, California.",
#        metadata={'source': 'docs/nvidia.txt'}
#    ),
#    Document(
#        page_content="Space Exploration Technologies Corp., commonly referred to as SpaceX, is an American space technology company headquartered at the Starbase development site in Starbase, Texas.",
#        metadata={'source': 'docs/spacex.txt'}
#    ),
#    Document(
#        page_content="Tesla, Inc. is an American multinational automotive and clean energy company headquartered in Austin, Texas.",
#        metadata={'source': 'docs/tesla.txt'}
#    )
# ]