from langchain_community.document_loaders import DirectoryLoader, UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

# Define the directory where your documents are stored
# Note: The folder name has a trailing space!
DOCS_DIR = "/Users/arunmsn/Desktop/Notebooks/StoryBooks"

print("Loading Privacy Impact Assessment documents...")
print(f"Checking directory: {DOCS_DIR}")

# Check if directory exists
if os.path.exists(DOCS_DIR):
    print("✅ Directory exists")
else:
    print("❌ Directory does not exist")
    # Let's try without the trailing space
    DOCS_DIR_NO_SPACE = DOCS_DIR.rstrip()
    print(f"Trying without trailing space: {DOCS_DIR_NO_SPACE}")
    if os.path.exists(DOCS_DIR_NO_SPACE):
        print("✅ Directory found without trailing space")
        DOCS_DIR = DOCS_DIR_NO_SPACE
    else:
        print("❌ Directory still not found")

# Load various document types using UnstructuredFileLoader
# Unstructured is great for handling PDFs, DOCX, TXT, HTML, etc.
# It requires additional dependencies like `pypdf`, `python-docx`, `markdown` etc.
# Install them with: pip install unstructured[all-docs]

documents = []
total_files_scanned = 0

for root, dirs, files in os.walk(DOCS_DIR):
    print(f"\nScanning: {root}")
    print(f"Found {len(files)} files: {files[:5]}{'...' if len(files) > 5 else ''}")
    
    for file in files:
        total_files_scanned += 1
        file_path = os.path.join(root, file)
        file_ext = os.path.splitext(file)[1].lower()
        
        # Filter for file types Unstructured can handle. Add more as needed.
        if file.lower().endswith(('.pdf', '.txt', '.docx', '.html', '.md')):
            try:
                print(f"  Loading: {file}")
                loader = UnstructuredFileLoader(file_path)
                docs = loader.load()
                documents.extend(docs)
                print(f"  ✅ Loaded {len(docs)} document(s)")
            except Exception as e:
                print(f"  ❌ Error loading {file}: {e}")
        else:
            print(f"  ⏭️ Skipped: {file} (extension: {file_ext})")

print(f"\nTotal files scanned: {total_files_scanned}")
print(f"Total documents loaded: {len(documents)}")

# Split Documents into Chunks:
# This is vital for fitting content into LLM context windows.
# RecursiveCharacterTextSplitter is good because it tries to keep chunks meaningful.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,      # Max characters per chunk
    chunk_overlap=200,    # Overlap between chunks to maintain context
    length_function=len,  # Use standard length function
    add_start_index=True, # Add start index to metadata
)

chunks = text_splitter.split_documents(documents)
print(f"Total chunks created: {len(chunks)}")

print("\n✅ Document loading complete! Your RAG system is ready with these chunks.")
# Uncomment the line below to inspect a sample chunk:
# print(f"\nSample chunk:\n{chunks[0].page_content[:500]}..." if chunks else "No chunks available")
