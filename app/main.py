# September 10, 2025
# from fastapi import FastAPI

#app = FastAPI()

#@app.get("/")
#def read_root():
#    return {"message": "Hello, FastAPI with UV!"}
    

from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel
from app.bigram_model import BigramModel
import spacy

app = FastAPI()

# Sample corpus for the bigram model
corpus = [
"The Count of Monte Cristo is a novel written by Alexandre Dumas. \
It tells the story of Edmond Dantès, who is falsely imprisoned and later seeks revenge.",
"this is another example sentence",
"we are generating text based on bigram probabilities",
"bigram models are simple but effective"
]
bigram_model = BigramModel(corpus)
nlp = spacy.load("en_core_web_lg")

class TextGenerationRequest(BaseModel):
    start_word: str
    length: int

class EmbeddingRequest(BaseModel):
    word: str

@app.get("/")
def read_root():
    return {"Hello": "World"}
@app.post("/generate")
def generate_text(request: TextGenerationRequest):
    generated_text = bigram_model.generate_text(request.start_word, request.length)
    return {"generated_text": generated_text}

@app.post("/embeddings")
def embeddings(request: EmbeddingRequest):
    print("LOOK FOR REQUEST WORD", request.word)
    print(nlp)
    emb_word = nlp(request.word)
    print("LOOK FOR WORD", emb_word)
    return emb_word.vector
