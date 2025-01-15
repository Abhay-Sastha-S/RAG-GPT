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

# Function to load a Word document from uploaded file
def load_docx(file):
    doc = Document(io.BytesIO(file.getvalue()))  # Read the bytes from the uploaded file
    text = [para.text for para in doc.paragraphs if para.text.strip() != '']
    return text

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
    doc_text = load_docx(file)
    doc_embeddings = create_embeddings(doc_text)
    
    # Create FAISS index
    index = faiss.IndexFlatL2(doc_embeddings.shape[1])
    index.add(np.array(doc_embeddings).astype(np.float32))
    
    # Save embeddings and index
    save_embeddings_and_index(doc_embeddings, index)

# Function to call the ArliAI API for generating responses
def call_arli_api(user_input, context, api_key):
    url = "https://api.arliai.com/v1/chat/completions"
    
    # Ensure context is directly related to the document, guiding the assistant to provide fact-based answers
    system_message = (
        "You are a helpful assistant. Please answer the user's question using only the information from the provided document. "
        "Do not invent any information or hallucinate. Stick strictly to the content in the document."
    )
    
    # Prepare the API payload
    payload = json.dumps({
        "model": "Mistral-Nemo-12B-Instruct-2407",  # Replace with the model you want to use
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": "How can I help you with that?"},  # Placeholder assistant response
            {"role": "user", "content": context}  # Provide the relevant context from the document
        ],
        "temperature": 0.01,  # Lower temperature for deterministic responses
        "max_tokens": 150,  # Set a reasonable limit for response length
        "top_p": 0.1,  # Reduce sampling to make the response more focused
        "top_k": 20,  # Reduce the number of highest probability tokens considered
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
        # Extract and return the assistant's response
        return response.json()['choices'][0]['message']['content']
    else:
        # If the request failed, return an error message
        return "Error: Unable to fetch response from ArliAI."

# Streamlit interface
def run_streamlit_app():
    st.title("RAG-powered Chatbot with ArliAI API")
    
    # Sidebar for document upload and API key input
    with st.sidebar:
        st.header("Upload Your Document & Enter API Key")
        
        # Upload the document
        uploaded_file = st.file_uploader("Upload Document (Word .docx)", type=["docx"])
        
        # Input for ArliAI API key
        api_key = st.text_input("Enter your ArliAI API Key:", type="password")
        
        # Submit button to process the document
        submit_button = st.button("Submit")
    
    # Handle document processing and API key submission
    if uploaded_file is not None and api_key and submit_button:
        # Process the document and create embeddings
        process_and_save_document(uploaded_file)
        st.session_state.api_key = api_key  # Save the API key in session
        st.session_state.chat_enabled = True
        st.session_state.chat_history = []  # Initialize chat history

        st.success("Document processed and chat enabled!")
    
    # Check if chat is enabled
    if 'chat_enabled' in st.session_state and st.session_state.chat_enabled:
        # Load embeddings and index for subsequent runs
        doc_embeddings, index = load_embeddings_and_index()
        doc_text = load_docx(uploaded_file)  # Load text from the document

        # Display chat history with styling
        if 'chat_history' in st.session_state:
            for user_msg, bot_msg in st.session_state.chat_history:
                st.markdown(f"<div style='background-color: #f1f1f1; padding: 10px; border-radius: 10px; margin-bottom: 10px;'>"
                            f"<strong>User:</strong> {user_msg}</div>", unsafe_allow_html=True)
                st.markdown(f"<div style='background-color: #d8eaff; padding: 10px; border-radius: 10px; margin-bottom: 10px;'>"
                            f"<strong>Bot:</strong> {bot_msg}</div>", unsafe_allow_html=True)
        
        # Input box to ask questions with placeholder
        user_input = st.text_input("Ask a question:", placeholder="Type your question here...", key="user_input", help="Type your question and press Enter.")
        
        if user_input:
            # Retrieve relevant text from the document
            relevant_text = retrieve_relevant_text(user_input, index, doc_text)

            # Show loading message while waiting for API response
            with st.spinner('Bot is thinking...'):
                # Generate answer using ArliAI API
                answer = call_arli_api(user_input, relevant_text, st.session_state.api_key)
            
            # Display the generated answer with styled chat bubbles
            st.session_state.chat_history.append((user_input, answer))  # Save to chat history
            st.markdown(f"<div style='background-color: #f1f1f1; padding: 10px; border-radius: 10px; margin-bottom: 10px;'>"
                        f"<strong>User:</strong> {user_input}</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='background-color: #d8eaff; padding: 10px; border-radius: 10px; margin-bottom: 10px;'>"
                        f"<strong>Bot:</strong> {answer}</div>", unsafe_allow_html=True)

            # Button to show the source of the answer
            if st.button("Show source of truth"):
                st.write(f"Source: {relevant_text}")
    
    else:
        st.warning("Please upload a document and enter the API key to enable chat.")

# Function to retrieve relevant text based on a query
def retrieve_relevant_text(query, index, doc_text):
    query_embedding = model.encode([query])
    _, indices = index.search(np.array(query_embedding).astype(np.float32), k=1)
    return doc_text[indices[0][0]]  # Retrieve the most relevant paragraph

# Run the Streamlit app
if __name__ == "__main__":
    run_streamlit_app()
