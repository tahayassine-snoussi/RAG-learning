import os 
from langchain_community.document_loaders import TextLoader, DirectoryLoader #read text files from a directory
from langchain_text_splitters import CharacterTextSplitter # split text into chunks
from langchain_openai import OpenAIEmbeddings # generate embeddings using OpenAI's API
from langchain_chroma import Chroma # vector store for storing and querying embeddings
from dotenv import load_dotenv # load environment variables from a .env file

load_dotenv()