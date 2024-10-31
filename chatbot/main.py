# # import pandas as pd
# # import os
# # import streamlit as st


# # file_formats = {
# #     "csv": pd.read_csv,
# #     "xls": pd.read_excel,
# #     "xlsx": pd.read_excel,
# #     "xlsm": pd.read_excel,
# #     "xlsb": pd.read_excel,
# # }
# # @st.cache_data(ttl="2h")
# # def load_data(uploaded_file):
# #     try:
# #         ext = os.path.splitext(uploaded_file.name)[1][1:].lower()
# #     except:
# #         ext = uploaded_file.split(".")[-1]
# #     if ext in file_formats:
# #         return file_formats[ext](uploaded_file)
# #     else:
# #         st.error(f"Unsupported file format: {ext}")
# #         return None

# # # Read the Pandas DataFrame
# # df = load_data(uploaded_file)

# from pymongo import MongoClient
# from sentence_transformers import SentenceTransformer
# import chromadb
# from tqdm import tqdm

# # Initialize MongoDB connection
# client = MongoClient('mongodb://localhost:27017/')  # Change as per your setup
# db = client['yelp']  # Database name
# collection = db['businesses']  # Collection name

# # Initialize the sentence transformer model
# model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

# # Initialize ChromaDB client
# chroma_client = chromadb.Client()

# # Create a new collection in ChromaDB (if it doesn't exist)
# # You can specify an embedding dimension if needed. MPNet base produces 768-dimension embeddings.
# collection_name = 'business_embeddings'
# chroma_collection = chroma_client.create_collection(name=collection_name)

# # Fetch data from MongoDB
# businesses = list(collection.find())
# total_businesses = len(businesses)

# # Loop through businesses and create embeddings
# for business in tqdm(businesses, desc="Processing businesses", total=total_businesses):
#     business_id = str(business['_id'])  # Use MongoDB _id as identifier
#     business_name = business.get('name', '')
#     business_description = business.get('description', '')  # Modify as per your collection schema
    
#     # Concatenate name and description (or modify as needed)
#     text_data = business_name + ' ' + business_description
    
#     # Generate embeddings
#     embedding = model.encode(text_data)
    
#     # Store embeddings in ChromaDB
#     chroma_collection.add(
#         ids=[business_id],
#         embeddings=[embedding.tolist()],  # Convert numpy array to list for JSON compatibility
#         metadatas=[business]  # Store the original document as metadata (optional)
#     )

# print(f"Stored embeddings for {chroma_collection.count()} businesses in ChromaDB.")

import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
# from langchain.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory
from pymongo import MongoClient
import random


load_dotenv()

OPENAI_KEY=os.environ.get("OPENAI_KEY")

class MongoRetriver:
    def __init__(self) -> None:
        pass

    def get_sample_from_mongodb(self, db_name, collection_name, sample_size=5, query_filter=None):
        print(f"Connecting to MongoDB database: {db_name}, collection: {collection_name}")
        
        # Establish MongoDB connection
        client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=5000, socketTimeoutMS=5000)  
        db = client[db_name]
        collection = db[collection_name]
        
        print(f"Connection established. Fetching sample of size: {sample_size}")
        
        # Initialize the aggregation pipeline
        pipeline = []
        
        # If a filter is provided, apply it
        if query_filter:
            print(f"Applying filter: {query_filter}")
            pipeline.append({"$match": query_filter})
        else:
            print("No filter applied.")
        
        # Use $sample to randomly sample documents
        pipeline.append({"$sample": {"size": sample_size}})
        print(f"Pipeline for aggregation: {pipeline}")

        # Execute the aggregation pipeline
        sample = list(collection.aggregate(pipeline))
        
        # print(f"Sample retrieved: {sample}")
        
        return sample
    
    
    def mongochain(self, sample, user_question, db_name):
        user_message = f"""
        Here is the User question: {user_question}
        You are a Data Analytics expert specializing in generating MongoDB queries based on natural language input, with a deep understanding of MongoDB schemas, business requirements, and data structures. 
        You are provided with the following sample data:

        {sample}

        Your task is to:
        1. Analyze the schema and structure of the provided data.
        2. Understand the business context and intent from the user's query—whether it's for retrieving insights, generating reports, filtering specific data, tracking key performance indicators (KPIs), or manipulating datasets for decision-making purposes.
        3. Construct optimized MongoDB queries based on the business objectives. These queries should:
        - Align with the specified business goals (e.g., decision-making, operational efficiency, trend identification).
        - Be efficient and scalable for handling large datasets.
        4. Ensure that the query retrieves relevant data accurately and in a timely manner.
        5. Design the query to be adaptable to potential future business needs and scaling requirements.

        ### Error Handling Instructions:
        - Ensure that the query gracefully handles cases where requested data is missing or unavailable.
        - Implement mechanisms to handle schema changes over time, such as optional fields or new document structures.
        - Incorporate validation checks to prevent issues such as invalid data types or incorrect field references.
        - Log or report any errors encountered during the execution of the query, providing sufficient detail for debugging.

        Use case examples:
        - Fetch customer behavior data for analytics and targeted marketing.
        - Generate sales reports grouped by product category, region, or time period.
        - Filter customer reviews for sentiment analysis to inform product development.
        - Perform trend analysis on business KPIs such as revenue growth or churn rate.
        """

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", user_message),
                MessagesPlaceholder(variable_name="history"),  # Keep history if required
                ("human", "{question}"),
            ]
        )

        chain = prompt | ChatOpenAI(openai_api_key=OPENAI_KEY)

        # Construct the input for the invoke call
        input_data = {"question": user_question}

        try:
            # Invoke the chain with the question
            response = chain.invoke(input_data)
        except Exception as e:
            print(f"Error invoking chain: {e}")
            response = {"error": str(e)}

        return response





    # def mongochain(self,sample,user_question,db_name):

    #     user_message = f"""
    #     Here is the User question {user_question}
    #     You are a Data Analytics expert specializing in generating MongoDB queries based on natural language input, with a deep understanding of MongoDB schemas, business requirements, and data structures. 
    #     You are provided with the following sample data:

    #     {sample}

    #     Your task is to:
    #     1. Analyze the schema and structure of the provided data.
    #     2. Understand the business context and intent from the user's query—whether it's for retrieving insights, generating reports, filtering specific data, tracking key performance indicators (KPIs), or manipulating datasets for decision-making purposes.
    #     3. Construct optimized MongoDB queries based on the business objectives. These queries should:
    #     - Align with the specified business goals (e.g., decision-making, operational efficiency, trend identification).
    #     - Be efficient and scalable for handling large datasets.
    #     4. Ensure that the query retrieves relevant data accurately and in a timely manner.
    #     5. Design the query to be adaptable to potential future business needs and scaling requirements.

    #     ### Error Handling Instructions:
    #     - Ensure that the query gracefully handles cases where requested data is missing or unavailable.
    #     - Implement mechanisms to handle schema changes over time, such as optional fields or new document structures.
    #     - Incorporate validation checks to prevent issues such as invalid data types or incorrect field references.
    #     - Log or report any errors encountered during the execution of the query, providing sufficient detail for debugging.

    #     Use case examples:
    #     - Fetch customer behavior data for analytics and targeted marketing.
    #     - Generate sales reports grouped by product category, region, or time period.
    #     - Filter customer reviews for sentiment analysis to inform product development.
    #     - Perform trend analysis on business KPIs such as revenue growth or churn rate.
    #     """


    #     prompt = ChatPromptTemplate.from_messages(
    #     [
    #         ("system", user_message),
    #         # MessagesPlaceholder(variable_name="history"),
    #         # ("human", "{question}"),
    #     ]
    #     )

    #     chain = prompt | ChatOpenAI(openai_api_key=OPENAI_KEY)


    #     # chain_with_history = RunnableWithMessageHistory(
    #     #         chain,
    #     #         lambda session_id: MongoDBChatMessageHistory(
    #     #                     session_id=session_id,
    #     #                     connection_string="mongodb://localhost:27017/",
    #     #                     database_name = db_name,
    #     #                     collection_name="chat_histories",
    #     #                 ),
    #     #                 input_messages_key="question",
    #     #                 history_messages_key="history",
    #     #             )
    #     # config = {"configurable": {"session_id": "<SESSION_ID>"}}

    #     response=chain.invoke({"question": "{user_question}"})

    #     return response

if __name__ == "__main__":
    db_name="yelp"
    collection_name="businesses"
    user_question= "give me business name from city tucson  and postal code 85705"
    mong=MongoRetriver()
    sample=mong.get_sample_from_mongodb(db_name=db_name,collection_name=collection_name)
    response =  mong.mongochain(sample=sample,user_question=user_question,db_name=db_name)
    