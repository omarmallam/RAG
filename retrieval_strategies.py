def basic_similarity_search(vector_store, query, k=4):
    docs = vector_store.similarity_search(query, k=k)
    for i, doc in enumerate(docs):
        print(f"\n--- Document {i+1} ---")
        print(f"Source: {doc.metadata.get('source', 'Unknown')}")
        print(f"Content: {doc.page_content[:150]}...")
    return docs
