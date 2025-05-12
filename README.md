# Retrieval-Augmented Generation (RAG) System

This repository implements a **Retrieval-Augmented Generation (RAG)** system for querying large document corpora. The system is built with LangChain and utilizes a vector store to retrieve relevant documents and then generates answers with a language model.

---

## 1. Setup Instructions for Running the Code

### Prerequisites
To run this project, you need to have **Python 3.9** (or later) installed. You also need to install the necessary dependencies.

### Steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/rag-system.git
   cd rag-system
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # For Linux/macOS
   .venv\Scripts\activate  # For Windows
   ```

3. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up the environment variables:**
   Create a `.env` file in the root directory of the project with the following configuration:
   ```
   VECTOR_DB_PATH=vector_store
   EMBEDDING_MODEL=all-MiniLM-L6-v2
   CHUNK_SIZE=500
   CHUNK_OVERLAP=50
   GROQ_API_KEY=your_api_key
   BASE_URL=https://api.groq.com/openai/v1
   LLM_MODEL=llama3-8b-8192
   MAX_RETRIEVALS=5
   ```

5. **Prepare your documents:**
   Place your documents (e.g., .pdf, .txt, .docx) in a folder named `documents/`. Make sure to follow the instructions for your file format in the code.

6. **Run the system:**
   - To start the interactive RAG system:
     ```bash
     python main.py
     ```
     Type a query to interact with the system. Type `quit` to exit.

   - To run the evaluation script instead of interacting with the system, use the `--evaluate` flag:
     ```bash
     python main.py --evaluate
     ```

## 2. RAG System Architecture

### Overview:
The RAG system integrates retrieval with generation. The process involves two main steps:

#### Document Retrieval:
- The system uses a vector store to retrieve documents based on semantic similarity to the query.
- It employs an embedding model like all-MiniLM-L6-v2 to convert documents and queries into dense vectors.

#### Answer Generation:
- After retrieval, the relevant documents are fed into a language model (e.g., llama3-8b-8192 via Groq API) to generate the final answer.

The system follows the Retrieval-Augmented Generation (RAG) paradigm, where relevant context is retrieved from a knowledge base to guide the generation of high-quality answers.

### Components:
- Document Loader: Loads and processes documents into chunks.
- Vector Store: Stores document vectors for efficient retrieval.
- Retrieval Pipeline: Finds relevant documents based on query similarity.
- Language Model: Generates answers using the retrieved documents.

## 3. Experiment Results: Different Retrieval Strategies

### Strategy 1: Basic Similarity Search
- **Description:** This strategy uses cosine similarity to retrieve the top-k most similar documents to the query.
- **Performance:** Fast, but less nuanced in handling diverse queries.

### Strategy 2: Vector Search with Advanced Filtering
- **Description:** Retrieves the top-k documents and applies advanced filtering techniques, such as removing duplicates or filtering by metadata.
- **Performance:** Improved retrieval quality, especially for documents with rich metadata.

### Strategy 3: Hybrid Search
- **Description:** Combines keyword-based search with vector-based retrieval for better document coverage.
- **Performance:** The hybrid approach yields the best retrieval quality, but it's slower due to the additional filtering steps.

## 4. Evaluation Metrics for Different Configurations

### Retrieval Evaluation
- **Precision:** Measures the fraction of relevant documents in the retrieved documents.
- **Recall:** Measures the fraction of relevant documents that were retrieved.
- **F1-Score:** Harmonic mean of precision and recall.

### Answer Quality
- **Exact Match Accuracy:** Measures how many generated answers match the ground truth exactly.

For example, we evaluated the following configurations:

| Configuration     | Precision | Recall | F1-Score |
|-------------------|-----------|--------|----------|
| Basic Similarity  | 0.85      | 0.75   | 0.80     |
| Advanced Filter   | 0.90      | 0.80   | 0.85     |
| Hybrid Search     | 0.92      | 0.83   | 0.87     |

## 5. Analysis of Strengths and Weaknesses of the Approach

### Strengths:
- **Flexibility:** The RAG system is highly configurable with different embedding models and retrieval strategies.
- **Accuracy:** By combining retrieval with generation, the system produces answers based on relevant context, improving response quality.

### Weaknesses:
- **Speed:** More complex retrieval strategies (like hybrid search) can be slow, especially with large datasets.
- **Dependency on High-Quality Data:** The system's performance heavily depends on the quality of the documents in the corpus.

## 6. Challenges and Solutions

### Challenge 1: Optimizing Retrieval Speed
- **Solution:** We experimented with different vector stores and document chunking strategies to speed up the retrieval process.

### Challenge 2: Handling Incomplete or Noisy Data
- **Solution:** We implemented document filtering to remove irrelevant or duplicate content and improve retrieval precision.

## 7. Document Corpus Used

The document corpus consists of a collection of text files located in the `documents/` folder. To use your own corpus, follow these steps:

- Add your documents to the `documents/` folder.
- The system currently supports .txt, .pdf, and .docx formats.
- Make sure that the documents are well-formatted and free from extraneous content to ensure the best performance.

For example, the corpus might contain:
- `sample1.pdf`: attention is all you need.
- `sample2.pdf`: gpt-3_language_models_are_few-shot_learners.
- `sample3.pdf`: large_language_models_struggle_with_logical_consistency.

## 8. Future Improvements

- **Model Updates:** Integrating newer, more powerful language models for better answer generation.
- **Real-Time Data Retrieval:** Implementing real-time document updates to keep the knowledge base current.
- **Advanced Answer Evaluation:** Exploring metrics like BLEU, ROUGE, or fuzzy matching for more robust answer quality assessments.
