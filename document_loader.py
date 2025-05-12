import glob
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader

def load_documents(documents_dir="documents"):
    all_docs = []

    # Load PDFs
    for pdf_file in glob.glob(f"{documents_dir}/**/*.pdf", recursive=True):
        try:
            loader = PyPDFLoader(pdf_file)
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = pdf_file
                doc.metadata["file_type"] = "pdf"
            all_docs.extend(docs)
        except Exception as e:
            print(f"Error loading PDF {pdf_file}: {e}")

    # Load DOCX files
    for docx_file in glob.glob(f"{documents_dir}/**/*.docx", recursive=True):
        try:
            loader = Docx2txtLoader(docx_file)
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = docx_file
                doc.metadata["file_type"] = "docx"
            all_docs.extend(docs)
        except Exception as e:
            print(f"Error loading DOCX {docx_file}: {e}")

    # Load TXT files
    for txt_file in glob.glob(f"{documents_dir}/**/*.txt", recursive=True):
        try:
            loader = TextLoader(txt_file)
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = txt_file
                doc.metadata["file_type"] = "txt"
            all_docs.extend(docs)
        except Exception as e:
            print(f"Error loading TXT {txt_file}: {e}")

    # Final log of the number of documents loaded
    print(f"Loaded {len(all_docs)} document(s) in total")
    return all_docs
