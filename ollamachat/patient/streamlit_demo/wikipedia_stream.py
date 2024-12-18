# import streamlit as st
# import wikipediaapi
# from langchain.schema import Document
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_ollama import OllamaEmbeddings
# from langchain.vectorstores import FAISS


# class WikipediaMunich:
#     def __init__(self):
#         pass

#     def fetch_wikipages(self, page):
#         wiki_wiki = wikipediaapi.Wikipedia(
#             user_agent='Testproj',
#             language='en',
#             extract_format=wikipediaapi.ExtractFormat.WIKI
#         )

#         p_wiki = wiki_wiki.page(page)
#         document = Document(
#             page_content=p_wiki.text,
#             metadata={"source": "Wikipedia", "title": p_wiki.title}
#         )
#         return document

#     def document_splitter(self, document):
#         text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=1000,
#             chunk_overlap=100
#         )
#         chunks = text_splitter.split_documents([document])
#         return chunks

#     def query_munich_page(self, query, chunks):
#         embeddings = OllamaEmbeddings(
#             model="nomic-embed-text",
#         )
#         vectorstore = FAISS.from_documents(chunks, embeddings)
#         results = vectorstore.similarity_search(query, k=3)
#         combined = [result.page_content for result in results]
#         return combined


# def main():
#     st.title("Wikipedia Munich Information Search")

#     # User input for question
#     question = st.text_input("Ask a question about Munich:", "Do people use bicycles in Munich?")

#     # User input for page name
#     page = st.text_input("Enter the Wikipedia page:", "Munich")

#     if st.button("Get Answer"):
#         wm = WikipediaMunich()
#         # Fetch the page and split it into chunks
#         document = wm.fetch_wikipages(page=page)
#         chunks = wm.document_splitter(document=document)

#         # Get similarity-based answers
#         answer = wm.query_munich_page(query=question, chunks=chunks)

#         # Display the results
#         st.write(f"### Results for the question: '{question}'")
#         for i, ans in enumerate(answer):
#             st.write(f"**Result {i + 1}:**")
#             st.write(ans)
#             st.markdown("---")


# if __name__ == "__main__":
#     main()
import streamlit as st
import wikipediaapi
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain.vectorstores import FAISS

class WikipediaMunich:
    def __init__(self):
        pass

    def fetch_wikipages(self, page):
        print("fetching pages")
        wiki_wiki = wikipediaapi.Wikipedia(
            user_agent='Testproj',
            language='en',
            extract_format=wikipediaapi.ExtractFormat.WIKI
        )

        p_wiki = wiki_wiki.page(page)
        document = Document(
            page_content=p_wiki.text,
            metadata={"source": "Wikipedia", "title": p_wiki.title}
        )
        return document

    def document_splitter(self, document):
        print("splitting docs")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        chunks = text_splitter.split_documents([document])
        return chunks

    def query_munich_page(self, query, chunks):
        print("chunking and querying")
        embeddings = OllamaEmbeddings(
            model="nomic-embed-text",
        )
        vectorstore = FAISS.from_documents(chunks, embeddings)
        results = vectorstore.similarity_search(query, k=3)
        print("answers chunking and querying")
        combined = [result.page_content for result in results]

        return combined

def main():
    # Streamlit title
    st.title("Wikipedia Munich Information Search")

    # Add a sidebar for additional information or help
    st.sidebar.title("About the App")
    st.sidebar.info("This app allows you to ask questions about Munich, and it will fetch the relevant information from Wikipedia, split the data into chunks, and use embeddings for similarity-based answers.")

    # User input for question
    question = st.text_input("Ask a question about Munich:", "Do people use bicycles in Munich?")

    # User input for page name
    page = st.text_input("Enter the Wikipedia page:", "Munich")

    # Inform the user if no question or page name is entered
    if not question or not page:
        st.warning("Please provide a valid question and Wikipedia page name.")
    
    # Button to trigger the answer search
    if st.button("Get Answer"):
        with st.spinner('Fetching data and processing your request...'):
            wm = WikipediaMunich()

            # Fetch the page and split it into chunks
            document = wm.fetch_wikipages(page=page)
            chunks = wm.document_splitter(document=document)

            # Get similarity-based answers
            answer = wm.query_munich_page(query=question, chunks=chunks)

            # Display the results with nice formatting
            st.success("Here are the top answers for your query:")
            for i, ans in enumerate(answer):
                st.write(f"**Result {i + 1}:**")
                st.write(ans)
                st.markdown("---")

    # Footer to keep the user engaged
    st.markdown("""
    ---
    **Tip**: You can try different questions or explore other Wikipedia pages related to Munich!
    """)

    # Provide more helpful information if needed
    st.sidebar.markdown("If you'd like more detailed help or have any questions, feel free to explore the documentation or contact support.")

if __name__ == "__main__":
    main()
