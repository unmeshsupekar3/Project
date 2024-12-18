import wikipediaapi
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from nomic import EmbedText
from langchain_ollama import OllamaEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_core.vectorstores import InMemoryVectorStore

class WikipediaMunich:
    def __init__(self):
        pass
    def fetch_wikipages(self,page):
        wiki_wiki = wikipediaapi.Wikipedia(
            user_agent='Testproj',
                language='en',
                extract_format=wikipediaapi.ExtractFormat.WIKI
        )


        # p_wiki = wiki_wiki.page("Munich")
        p_wiki = wiki_wiki.page(page)
        # print(p_wiki.text)
        document = Document(page_content=p_wiki.text,
                            metadata={"source": "Wikipedia",
                                        "title": p_wiki.title}
                                )
        # print(type(document))
        return document

    def document_splitter(self,document):

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=100
        )
        chunks = text_splitter.split_documents([document])

        # print(len(chunks))
        return chunks


    def query_munich_page(self,query,chunks):
        print("Embedding started")
        embeddings = OllamaEmbeddings(
            model="nomic-embed-text",
        )
        print("Vectorstore started")
        vectorstore = FAISS.from_documents(chunks, embeddings)
        print("Similarity retriever active")
        # retriever = vectorstore.as_retriever()
        results = vectorstore.similarity_search(query, k=3)
        combined = []
        for i, result in enumerate(results):
            print(f"Result {i + 1}:\n{result}\n")
            combined.append(result.page_content)
        # retrieved = retriever.invoke(query)
        # print(retrieved)
        return combined




if __name__ == "__main__":
    question = "Do people use bicycles in Munich?"
    page = 'Munich'
    wm=WikipediaMunich()
    document=wm.fetch_wikipages(page=page)
    chunks=wm.document_splitter(document=document)
    answer = wm.query_munich_page(query=question,
                         chunks=chunks)

    # answer = query_munich_page(question)
    for i in answer:

        print(f"Q: {question}\nA: {i}")

# # embeddings = EmbedText()
# # ollama.embeddings(model='nomic-embed-text', prompt='The sky is blue because of rayleigh scattering')

# vectorstore = InMemoryVectorStore.from_texts(
#     [text],
#     embedding=embeddings,
# )