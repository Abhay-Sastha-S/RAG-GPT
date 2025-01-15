# RAG-Powered Legal Chatbot

## Overview
This project is a **Retrieval-Augmented Generation (RAG)** chatbot designed for precise question-answering using legal documents, specifically the *Bharatiya Nyaaya Samhita*. The system combines **semantic search** with **generative AI** to provide factually accurate and context-specific answers, ensuring that responses are grounded in the uploaded source material.

## Key Features
1. **Semantic Document Embedding**:
   - Uses **Sentence Transformers (paraphrase-MiniLM-L6-v2)** to encode document text into embeddings.
   - Dynamically splits documents into contextually meaningful chunks to preserve semantic integrity.

2. **Efficient Retrieval**:
   - Implements **FAISS (Facebook AI Similarity Search)** for scalable and high-speed similarity searches on the embeddings.

3. **Generative AI Integration**:
   - Queries the **ArliAI API** to generate detailed responses based on retrieved document chunks.
   - Avoids hallucination by grounding the generative model in retrieved, fact-based content.

4. **User-Friendly Interface**:
   - Built with **Streamlit** for an intuitive, interactive user experience.
   - Supports document uploads, query inputs, and real-time responses with contextual references.

## System Architecture
1. **Document Processing**:
   - Uploaded `.docx` files are processed into paragraphs and encoded into embeddings using Sentence Transformers.
   - The embeddings are stored in a FAISS index for efficient retrieval.

2. **Query Handling**:
   - User queries are encoded into embeddings and matched against the FAISS index.
   - The top-k most relevant document chunks are retrieved for context.

3. **Response Generation**:
   - The retrieved context is sent to the ArliAI API along with the user query.
   - A deterministic, factually grounded response is generated and displayed in the UI.

## Installation
### Prerequisites
- Python 3.8+
- Libraries:
  ```bash
  pip install -r requirements.txt
  ```
- ArliAI API Key (Register at [ArliAI](https://api.arliai.com/))

### Required Python Libraries
- `streamlit`
- `faiss`
- `sentence-transformers`
- `python-docx`
- `numpy`
- `pickle`
- `requests`

### Clone the Repository
```bash
git clone https://github.com/your-repo/rag-legal-chatbot.git
cd rag-legal-chatbot
```

## Usage
### Step 1: Run the Application
Start the Streamlit app:
```bash
streamlit run app.py
```

### Step 2: Upload a Document
- Upload a `.docx` file containing legal content (e.g., *Bharatiya Nyaaya Samhita*).

### Step 3: Enter Your ArliAI API Key
- Input your API key in the sidebar to enable chatbot functionality.

### Step 4: Ask Questions
- Type queries related to the uploaded document in the input box.
- The chatbot will retrieve relevant content and provide detailed, context-aware responses.

## Technical Details
### Document Chunking
- Documents are split into paragraphs, ensuring that each chunk is semantically coherent and contextually rich.
- This improves the accuracy of retrieval and ensures meaningful answers.

### FAISS Index
- Embeddings are stored in a **FlatL2** FAISS index for similarity search.
- Queries are matched to the top-k most relevant chunks using cosine similarity.

### Generative AI
- The **ArliAI API** powers the generative responses by grounding them in the retrieved chunks.
- Hyperparameters such as `temperature`, `top_p`, and `max_tokens` are optimized for deterministic and concise answers.

## Example Workflow
1. Upload a document containing legal sections.
2. Ask: *"Under what sections is private defense allowed?"*
3. The chatbot retrieves relevant sections and responds:
   - *"Based on the provided extract from the Bharatiya Nyaaya Samhita, the sections under which private defense is okay and extends to causing death are:
Right of Private Defense of Property (Section 40):

Robbery
House-breaking after sunset and before sunrise
Mischief by fire or any explosive substance committed on any building, tent, or vessel used as a human dwelling or for property custody
Theft, mischief, or house-trespass under circumstances that may reasonably cause apprehension of death or grievous hurt
Right of Private Defense of Body (Section 38):

Assault that may reasonably cause apprehension of death or grievous hurt
Assault with the intention of committing rape, gratifying unnatural lust, kidnapping or abducting, or wrongfully confining a person
Act of throwing or administering acid that may reasonably cause apprehension of grievous hurt"*

4. View the retrieved source content by clicking **Show Source of Truth**.

## Future Improvements
- Support for additional document formats (e.g., PDF).
- Advanced query expansion techniques for better contextual understanding.
- Integration with other generative AI models for improved response quality.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contributing
Contributions are welcome! Please create a pull request or open an issue for any enhancements or bug fixes.
---
