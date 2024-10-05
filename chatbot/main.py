# import pandas as pd
# import os
# import streamlit as st


# file_formats = {
#     "csv": pd.read_csv,
#     "xls": pd.read_excel,
#     "xlsx": pd.read_excel,
#     "xlsm": pd.read_excel,
#     "xlsb": pd.read_excel,
# }
# @st.cache_data(ttl="2h")
# def load_data(uploaded_file):
#     try:
#         ext = os.path.splitext(uploaded_file.name)[1][1:].lower()
#     except:
#         ext = uploaded_file.split(".")[-1]
#     if ext in file_formats:
#         return file_formats[ext](uploaded_file)
#     else:
#         st.error(f"Unsupported file format: {ext}")
#         return None

# # Read the Pandas DataFrame
# df = load_data(uploaded_file)

from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
import chromadb
from tqdm import tqdm

# Initialize MongoDB connection
client = MongoClient('mongodb://localhost:27017/')  # Change as per your setup
db = client['yelp']  # Database name
collection = db['businesses']  # Collection name

# Initialize the sentence transformer model
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

# Initialize ChromaDB client
chroma_client = chromadb.Client()

# Create a new collection in ChromaDB (if it doesn't exist)
# You can specify an embedding dimension if needed. MPNet base produces 768-dimension embeddings.
collection_name = 'business_embeddings'
chroma_collection = chroma_client.create_collection(name=collection_name)

# Fetch data from MongoDB
businesses = list(collection.find())
total_businesses = len(businesses)

# Loop through businesses and create embeddings
for business in tqdm(businesses, desc="Processing businesses", total=total_businesses):
    business_id = str(business['_id'])  # Use MongoDB _id as identifier
    business_name = business.get('name', '')
    business_description = business.get('description', '')  # Modify as per your collection schema
    
    # Concatenate name and description (or modify as needed)
    text_data = business_name + ' ' + business_description
    
    # Generate embeddings
    embedding = model.encode(text_data)
    
    # Store embeddings in ChromaDB
    chroma_collection.add(
        ids=[business_id],
        embeddings=[embedding.tolist()],  # Convert numpy array to list for JSON compatibility
        metadatas=[business]  # Store the original document as metadata (optional)
    )

print(f"Stored embeddings for {chroma_collection.count()} businesses in ChromaDB.")
