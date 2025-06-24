import streamlit as st
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.prompts import PromptTemplate
from deep_translator import GoogleTranslator
from langchain.chains import RetrievalQA
from PyPDF2 import PdfReader
from audio_recorder_streamlit import audio_recorder
import speech_recognition as sr
import os

# Streamlit Layout Configuration
st.set_page_config(layout="wide")

# Sidebar
st.sidebar.header("📄 Upload Travel Preferences")
uploaded_file = st.sidebar.file_uploader("Upload Travel Preferences (PDF)", type="pdf")

# Chatbot Title
st.markdown("<h1 style='text-align: center;'>✈️ AI-Driven Virtual Travel Agent 🤖</h1>", unsafe_allow_html=True)

# Define Folder Paths
folder_path = "db"
os.makedirs(folder_path, exist_ok=True)

# Initialize Models and Embeddings
cached_llm = Ollama(model="wizardlm2:7b")
embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
text_splitters = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=100)

# Define Prompt for Travel Itinerary Generation
raw_prompt = PromptTemplate.from_template(
"""<s>[INST] You are a virtual travel agent designed to create personalized travel itineraries based on user preferences such as budget, interests, and location. Your goal is to provide detailed trip plans, including flights, accommodations, activities, and dining options.

#### When responding:
1. **Greetings**:
   - If the user greets you (e.g., "Hello," "Hi," "How are you?"), respond warmly and ask for their travel preferences.
   - Examples:
     - "Hello! Where are you planning to travel?"
     - "Hi there! What’s your budget and interests for the trip?"
     - "Good morning! Let’s plan your dream vacation. What are your preferences?"

2. **Travel Preferences**:
   - Ask for the user’s budget, preferred location, and interests (e.g., hills, beaches, temples, adventure, etc.).
   - Use the context provided (e.g., uploaded PDF or ingested data) to generate a personalized itinerary.

3. **Itinerary Generation**:
   - Provide a detailed trip plan, including:
     - Flights (if applicable)
     - Accommodations (hotels, resorts, etc.)
     - Activities (sightseeing, adventure, etc.)
     - Dining options (restaurants, cafes, etc.)
   - Ensure the plan aligns with the user’s budget and interests.

4. **Contextual Responses**:
   - Only use the context (e.g., uploaded PDF or ingested data) if the user’s query explicitly relates to it. For example:
     - If the user asks, "What are the best hotels in Goa?" and the context contains this information, provide a personalized response.
     - If the user asks a general question like "How are you?", do not pull unrelated context.
     

Remember, your role is to be a helpful and efficient travel assistant. [/INST] </s>
[INST] {input} Context: {context} Answer: [/INST]"""
)

def is_greeting(query):
    greetings = ["hi", "hello", "hey", "good morning", "good afternoon", "how are you", "what's up"]
    return any(greeting in query.lower() for greeting in greetings)

# Function to generate a greeting response
def generate_greeting_response():
    return "Hello! Where are you planning to travel? Let’s plan your dream vacation! 😊"

def handle_query(query):
    if is_greeting(query):
        # Return a predefined greeting response
        return generate_greeting_response(), []
    else:
        # Proceed with the RAG pipeline for travel-related queries
        return retrieve_answer(query)

# Process Uploaded PDF (Travel Preferences)
if uploaded_file:
    # Create Folder if it is not present
    if not os.path.exists("./pdf"):
        os.makedirs("./pdf")

    save_file = f"pdf/{uploaded_file.name}"
    with open(save_file, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Change to use PdfReader instead of PDFPlumberLoader
    pdf_reader = PdfReader(save_file)
    text = "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
    
    # Split text directly instead of documents
    chunks = text_splitters.split_text(text)
    
    # Create vector store from texts instead of documents
    vector_store = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory=folder_path
    )
    vector_store.persist()
    st.sidebar.success("Travel preferences embedded successfully!")

# Chat Section
st.markdown("""
    <style>
        .chat-container {
            border: 2px solid #ddd;
            border-radius: 10px;
            padding: 15px;
            height: 500px;
            overflow-y: auto;
        }
    </style>
""", unsafe_allow_html=True)

# Initialize chat history if not present
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Function to handle chat response
def retrieve_answer(query):
    vector_store = Chroma(persist_directory=folder_path, embedding_function=embeddings)
    retriever = vector_store.as_retriever()
    # Use RetrievalQA instead of custom chains
    qa_chain = RetrievalQA.from_chain_type(cached_llm, retriever=retriever)
    result = qa_chain.run(query)
    return result, [] 

# Chat Input
query = st.chat_input("Enter your message here...")

# Process Query
if query:
    with st.spinner("Planning your trip..."):
        response, sources = handle_query(query)
        if response:
            # Append to Chat History
            st.session_state.chat_history.append((query, response))

    # Display chat history
    for q, r in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(q)
        with st.chat_message("assistant"):
            st.write(r)