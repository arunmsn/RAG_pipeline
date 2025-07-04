from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from tqdm import tqdm  # For progress bar

# Import the document loading and chunking functionality
print("Loading documents and creating chunks...")
exec(open('LoadDocument.py').read())

print(f"Loaded {len(documents)} documents and created {len(chunks)} chunks")

# 3.1 Initialize the local embedding model
# 'all-MiniLM-L6-v2' is small, fast, and performs well for its size.
# 'BAAI/bge-small-en-v1.5' is another excellent small model, but might be slightly slower.
print("Initializing embedding model...")
embeddings_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# 3.2 Initialize ChromaDB
# This will create a 'chroma_db' folder in your current directory to store the data.
persist_directory = "chroma_db_binder"  # <--- This is where your vector DB will live
print(f"Initializing ChromaDB at: {persist_directory}")
vectorstore = Chroma(
    collection_name="my_binder_collection",
    embedding_function=embeddings_model,
    persist_directory=persist_directory
)

# 3.3 Add chunks to ChromaDB
# For tens of thousands of pages, this will take a *long* time.
# Consider batching if you run into memory issues or want to resume.
# Also, ensure you only add new chunks if you rerun the script.

if len(chunks) == 0:
    print("❌ No chunks found! Make sure LoadDocument.py is working properly.")
else:
    print(f"Adding {len(chunks)} chunks to ChromaDB...")
    # Use tqdm for a progress bar
    batch_size = 1000  # Process in batches to manage memory and network calls (if using remote API)
    for i in tqdm(range(0, len(chunks), batch_size), desc="Adding chunks"):
        batch = chunks[i:i + batch_size]
        # You might want to check if the chunk already exists before adding
        # For now, we'll just add them.
        vectorstore.add_documents(batch)

    # Persist the database to disk
    vectorstore.persist()
    print(f"✅ Finished adding chunks. ChromaDB persisted at: {persist_directory}")

# To load an existing vectorstore:
# vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings_model)