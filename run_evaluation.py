from rag_pipeline import setup_rag_chain, run_query
from document_processor import load_vector_store
from langchain_huggingface import HuggingFaceEmbeddings
from evaluation import evaluate_retrieval, evaluate_answers
import os

# Your test cases
evaluation_data = [
    {
        "query": "What is the objective of Lab 6?",
        "expected_doc_ids": ["documents/lab6.pdf"],
        "expected_answer": "To build a retrieval-augmented generation (RAG) system."
    },
    {
        "query": "Which model is used for embedding?",
        "expected_doc_ids": ["documents/config.txt"],
        "expected_answer": "all-MiniLM-L6-v2"
    }
]

# Load system
embedding_model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
vector_store_path = os.getenv("VECTOR_DB_PATH", "vector_store")
embeddings = HuggingFaceEmbeddings(model_name=f"sentence-transformers/{embedding_model_name}")
vector_store = load_vector_store(vector_store_path, embeddings)
rag_chain = setup_rag_chain(vector_store)

# Evaluation loop
retrieval_scores = []
predicted_answers = []
expected_answers = []

for item in evaluation_data:
    query = item["query"]
    expected_doc_ids = item["expected_doc_ids"]
    expected_answer = item["expected_answer"]

    # Run query
    result = run_query(rag_chain, query)
    predicted_answer = result['result']
    retrieved_docs = [doc.metadata.get("source") for doc in result["source_documents"]]

    # Collect
    predicted_answers.append(predicted_answer)
    expected_answers.append(expected_answer)

    retrieval_result = evaluate_retrieval(expected_doc_ids, retrieved_docs)
    retrieval_scores.append(retrieval_result)

# Output
print("\n=== Retrieval Metrics ===")
for i, scores in enumerate(retrieval_scores):
    print(f"Query {i+1}: Precision={scores['precision']:.2f}, Recall={scores['recall']:.2f}, F1={scores['f1']:.2f}")

print("\n=== Answer Accuracy ===")
answer_metrics = evaluate_answers(predicted_answers, expected_answers)
print(f"Exact Match Accuracy: {answer_metrics['exact_match_accuracy']:.2f}")
