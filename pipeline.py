import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import PyPDF2
import mimetypes
import pyttsx3
from transformers import WhisperProcessor, WhisperForConditionalGeneration, EncoderDecoderCache
import soundfile as sf
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
import uuid
from dotenv import load_dotenv
import os
import google.generativeai as genai


def extract_text_from_pdf(pdf_path):
    text = ""
    mime_type, _ = mimetypes.guess_type(pdf_path)

    if mime_type == 'text/plain':
        # Handle text files
        with open(pdf_path, 'r', encoding='utf-8') as file:
            text = file.read()
    else:
        # Handle PDF files
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text()
    return text


# Clean the text
def clean_text(text):
    text = re.sub(r'\s+', ' ', text) 
    text = re.sub(r'\n', ' ', text) 
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  
    return text

# Split the text into chunks
def chunk_text(text, chunk_size=500, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_text(text)
    print(f"Chunks: {chunks}")
    return chunks


def embed(texts):
    EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return EMBED_MODEL.encode(texts, normalize_embeddings=True).astype("float32")

chroma_client = chromadb.PersistentClient("chroma_db")
c_collection = chroma_client.get_or_create_collection("rag_chroma")

def create_vector_db(chunks):
    c_collection.add(
        ids=[str(uuid.uuid4()) + f"-{i}" for i in range(len(chunks))],
        documents=chunks,
        embeddings=embed(chunks),
    )

def retrieve_context_response(query: str, k: int = 3):
    res = c_collection.query(query_texts=[query], n_results=k)
    full_prompt = f"Context: {res["documents"][0]}\n\nQuestion: {query}\n\nAnswer:"
    result = model.generate_content(full_prompt)
    return result.text.strip()

def speech_to_text(audio_file, language='en'):
   
    # Speech to text conversion for input query
    processor = WhisperProcessor.from_pretrained("openai/whisper-small")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
    audio_input, _ = sf.read(audio_file)
    input_features = processor(audio_input, sampling_rate=16000, return_tensors="pt", language=language).input_features
    past_key_values = None
    encoder_decoder_cache = EncoderDecoderCache.from_legacy_cache(past_key_values)
    predicted_ids = model.generate(input_features, past_key_values=encoder_decoder_cache)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    return transcription[0]

def text_to_speech(text):
    engine = pyttsx3.init()  
    engine.say(text)  
    engine.runAndWait()  


load_dotenv()
api_key = os.getenv("API_KEY")
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-1.5-flash")


# Executing AI pipeline
def ai_pipeline_development(pdf_path):

    # Load and preprocess the PDF
    book_text = extract_text_from_pdf(pdf_path)
    cleaned_text = clean_text(book_text)
    chunks = chunk_text(cleaned_text)
    
    # Create or load vector database
    vector_db = create_vector_db(chunks)
    
    return vector_db


# Response generation
def response_generation(audio_file):    
    
    query_text = speech_to_text(audio_file)
    response = retrieve_context_response(query_text)
    print(f"Response: {response}")
    text_to_speech(response)
    return response


def main():

    pdf_path = 'D:/Indiabe/bio.pdf'
    ai_pipeline_development(pdf_path)

    audio_file = 'D:/Indiabe/test.wav'
    response_generation(audio_file)


if __name__ == '__main__':
    main()
    

