from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from routes import ( chat_ollama_routes,
                    wikipedia_routes)

app = FastAPI()

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, 
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"], 
)

print("~~~~~~~~~~~APPLICATION READY~~~~~~~~~~~~~~~~~~")

app.include_router(chat_ollama_routes.router)

app.include_router(wikipedia_routes.router)

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=953, reload=True)