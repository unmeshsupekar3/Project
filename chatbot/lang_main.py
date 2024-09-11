from langchain_community.document_loaders.mongodb import MongodbLoader

loader = MongodbLoader(
    connection_string="mongodb://localhost:27017/",
    db_name="yelp",
    collection_name="businesses",
)
print("k")
docs = loader.load()

print(len(docs))