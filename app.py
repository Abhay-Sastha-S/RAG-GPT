import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from docx import Document
import streamlit as st
import requests
import json
import io

# Initialize Sentence Transformer model for embeddings
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Function to load a Word document and return the text
def load_docx(file):
    doc = Document(io.BytesIO(file.getvalue()))  # Read the bytes from the uploaded file
    text = [para.text for para in doc.paragraphs if para.text.strip() != '']
    return text

# Chunk text into larger blocks (e.g., ~500 words or 2000 characters per chunk)
def chunk_text(text_list, chunk_size=2000):
    chunks = []
    current_chunk = ""

    for paragraph in text_list:
        # Add paragraph to the current chunk
        if len(current_chunk) + len(paragraph) <= chunk_size:
            current_chunk += f"{paragraph} "
        else:
            # Save the current chunk and start a new one
            chunks.append(current_chunk.strip())
            current_chunk = f"{paragraph} "

    # Add the last chunk if not empty
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

# Convert text to embeddings
def create_embeddings(text_list):
    embeddings = model.encode(text_list)
    return embeddings

# Save the embeddings and index to files
def save_embeddings_and_index(embeddings, index, embeddings_file="embeddings.pkl", index_file="index.faiss"):
    with open(embeddings_file, 'wb') as f:
        pickle.dump(embeddings, f)
    faiss.write_index(index, index_file)

# Load embeddings and index from files
def load_embeddings_and_index(embeddings_file="embeddings.pkl", index_file="index.faiss"):
    with open(embeddings_file, 'rb') as f:
        embeddings = pickle.load(f)
    index = faiss.read_index(index_file)
    return embeddings, index

# Function to process and save the document (only run once for the document)
def process_and_save_document(file):
    # Load and chunk the document text
    doc_text = load_docx(file)
    doc_chunks = chunk_text(doc_text, chunk_size=2000)  # Adjust `chunk_size` as needed

    # Create embeddings for the chunks
    doc_embeddings = create_embeddings(doc_chunks)
    
    # Create FAISS index
    index = faiss.IndexFlatL2(doc_embeddings.shape[1])
    index.add(np.array(doc_embeddings).astype(np.float32))
    
    # Save embeddings and index
    save_embeddings_and_index(doc_embeddings, index)

    # Optionally return the chunks for debugging
    return doc_chunks

# Function to call the ArliAI API for generating responses
def call_arli_api(user_input, context, api_key):
    url = "https://api.arliai.com/v1/chat/completions"
    
    # Ensure context is directly related to the document
    system_message = (
        "You are a helpful assistant. Please answer the user's question using only the information from the provided extract from the new Indian Penal Code called the Bharatiya Nyaaya Samhita. "
        "forget About indian penal code that you know before answering this"
        "Do not invent any information or hallucinate. Stick strictly to the content in the document."
    )
    
    # Prepare the API payload
    payload = json.dumps({
        "model": "Mistral-Nemo-12B-Instruct-2407",  # Replace with the desired model
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": "How can I help you with that?"},  # Placeholder assistant response
            {"role": "user", "content": context}  # Provide the relevant context from the document
        ],
        "temperature": 0,
        "max_tokens": 1500,
        "top_p": 0.3,
        "top_k": 45,
        "stream": False
    })
    
    # Set headers with API key for authorization
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f"Bearer {api_key}"
    }

    # Make the request to the ArliAI API
    response = requests.post(url, headers=headers, data=payload)

    # Check if the response was successful
    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']
    else:
        return "Error: Unable to fetch response from ArliAI."

# Streamlit interface
def run_streamlit_app():
    st.title("RAG-powered Chatbot with ArliAI API")
    
    # Sidebar for document upload and API key input
    with st.sidebar:
        st.header("Upload Your Document & Enter API Key")
        uploaded_file = st.file_uploader("Upload Document (Word .docx)", type=["docx"])
        api_key = st.text_input("Enter your ArliAI API Key:", type="password")
        submit_button = st.button("Submit")
    
    # Initialize session state variables
    if "chat_enabled" not in st.session_state:
        st.session_state.chat_enabled = False
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "doc_chunks" not in st.session_state:
        st.session_state.doc_chunks = None

    # Handle document processing and API key submission
    if uploaded_file is not None and api_key and submit_button:
        doc_chunks = process_and_save_document(uploaded_file)
        st.session_state.api_key = api_key
        st.session_state.chat_enabled = True
        st.session_state.chat_history = []
        st.session_state.doc_chunks = doc_chunks
        st.success("Document processed and chat enabled!")
    
    # Check if chat is enabled
    if st.session_state.chat_enabled:
        doc_embeddings, index = load_embeddings_and_index()
        doc_chunks = st.session_state.doc_chunks

        # Display chat history
        if st.session_state.chat_history:
            for user_msg, bot_msg in st.session_state.chat_history:
                st.markdown(f"<div style='background-color: #525151; padding: 10px; border-radius: 10px; margin-bottom: 10px;'>"
                            f"<strong>User:</strong> {user_msg}</div>", unsafe_allow_html=True)
                st.markdown(f"<div style='background-color: #364e6b; padding: 10px; border-radius: 10px; margin-bottom: 10px;'>"
                            f"<strong>Bot:</strong> {bot_msg}</div>", unsafe_allow_html=True)
        
        user_input = st.text_input("Ask a question:", placeholder="Type your question here...", key="user_input")
        
        if user_input:
            relevant_texts = retrieve_relevant_texts(user_input, index, doc_chunks, top_k=3)
            combined_context = "\n".join(relevant_texts)

            with st.spinner('Bot is thinking...'):
                answer = call_arli_api(user_input, combined_context, st.session_state.api_key)
            
            st.session_state.chat_history.append((user_input, answer))
            st.markdown(f"<div style='background-color:#525151; padding: 10px; border-radius: 10px;'>"
                        f"<strong>User:</strong> {user_input}</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='background-color: #364e6b; padding: 10px; border-radius: 10px;'>"
                        f"<strong>Bot:</strong> {answer}</div>", unsafe_allow_html=True)

            if st.button("Show source of truth"):
                st.write(f"Source: {combined_context}")
    else:
        st.warning("Please upload a document and enter the API key to enable chat.")

# Function to retrieve relevant texts based on a query
def retrieve_relevant_texts(query, index, doc_text, top_k=1):
    query_embedding = model.encode([query])
    _, indices = index.search(np.array(query_embedding).astype(np.float32), k=top_k)
    L = (doc_text[idx] for idx in indices[0] if idx < len(doc_text))
    print(L)
    return L

# Run the Streamlit app
if __name__ == "__main__":
    run_streamlit_app()
