import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import PyPDF2
import mimetypes
import pyttsx3
from transformers import WhisperProcessor, WhisperForConditionalGeneration, EncoderDecoderCache
import soundfile as sf
import requests
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline

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

# Load the data for vector database
def load_book(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

# Clean the text
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)  
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  
    return text

# Split the text into chunks
def chunk_text(text, chunk_size=1000, chunk_overlap=100):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_text(text)
    print(chunks)
    return chunks

# Create vector database
def create_vector_db(chunks, embedding_model):
    vector_db = FAISS.from_texts(chunks, embedding_model)
    return vector_db

# Speech to text conversion for input query
def speech_to_text(audio_file, language='en'):
    
    processor = WhisperProcessor.from_pretrained("openai/whisper-small")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
    audio_input, _ = sf.read(audio_file)
    input_features = processor(audio_input, sampling_rate=16000, return_tensors="pt", language=language).input_features
    past_key_values = None
    encoder_decoder_cache = EncoderDecoderCache.from_legacy_cache(past_key_values)
    predicted_ids = model.generate(input_features, past_key_values=encoder_decoder_cache)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    
    return transcription[0]

# Text to speech for speech response   
def text_to_speech(text):
    engine = pyttsx3.init()  
    engine.say(text)  
    engine.runAndWait() 


# retrieving response from LLM 
# Define the API endpoint and API key
API_KEY = "key"  
BASE_URL = "http://127.0.0.1:8080"  
MODEL_ENDPOINT = "/completion"

url = f"{BASE_URL}{MODEL_ENDPOINT}"

class AnacondaLLM:
    def __init__(self, api_endpoint, api_key):
        self.api_endpoint = api_endpoint
        self.api_key = api_key

    def query(self, prompt):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "prompt": prompt
        }
        response = requests.post(self.api_endpoint, headers=headers, json=data)
        response.json()
        return response.json()["content"] 

# Instantiate the AnacondaLLM class
llm = AnacondaLLM(url, API_KEY)

def retrieve_context_response(vector_db, query_text):
    retriever = vector_db.as_retriever()
    relevant_docs = retriever.get_relevant_documents(query_text)
    context = " ".join([doc.page_content for doc in relevant_docs])
    print(context)
    full_prompt = f"Context: {context}\n\nQuestion: {query_text}\n\nAnswer:"
    result = llm.query(full_prompt)
    
    return result

def ai_pipeline_development(pdf_path):

    # Load and preprocess the PDF
    book_text = extract_text_from_pdf(pdf_path)
    cleaned_text = clean_text(book_text)
    chunks = chunk_text(cleaned_text)
    
    # Create or load vector database
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = create_vector_db(chunks, embedding_model)
    
    return vector_db


def response_generation(vector_db, audio_file):    
    
	  query_text = speech_to_text(audio_file)
    response = retrieve_context_response(vector_db, query_text)
    print(response)
    text_to_speech(response)
    return response


def main():

pdf_path = 'D:/path/file.pdf'
db = ai_pipeline_development(pdf_path)

audio_file = 'D:/path/test_audio.wav'
response = response_generation(db, audio_file)


if __name__ == '__main__':
    main()
    
