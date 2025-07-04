import os
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama

# === Added imports for hybrid retrieval and prompt ===
from langchain_community.retrievers import BM25Retriever
from langchain.prompts import PromptTemplate
from langchain.schema import Document

def setup_rag_system():
    """Set up the RAG system by loading the existing ChromaDB"""
    print("🔄 Loading your Privacy Impact Assessment RAG system...")

    # Load the same embedding model used to create the database
    embeddings_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    # Load the existing ChromaDB
    persist_directory = "chroma_db_binder"
    vectorstore = Chroma(
        collection_name="my_binder_collection",
        embedding_function=embeddings_model,
        persist_directory=persist_directory
    )

    print(f"✅ Loaded vector database with {vectorstore._collection.count()} chunks")

    return vectorstore

# === New function to setup BM25 retriever from vectorstore docs ===
def setup_bm25_retriever(vectorstore):
    docs = vectorstore.get(include=["documents"])["documents"]
    return BM25Retriever.from_documents([Document(page_content=doc) for doc in docs])

def setup_llm():
    """Set up the language model - tries multiple options"""
    print("\n🤖 Setting up language model...")

    ollama_models = [
        "llama3.2",
        "llama3.2:1b",
        "llama3",
        "mistral",
        "codellama",
        "llama2",
        "phi3"
    ]

    for model_name in ollama_models:
        try:
            print(f"Trying Ollama model: {model_name}...")
            llm = Ollama(model=model_name, base_url="http://localhost:11434")
            # Test if Ollama is running with a simple test
            test_response = llm.invoke("Hello")
            print(f"✅ Using Ollama model: {model_name}")
            return llm
        except Exception as e:
            print(f"❌ Model {model_name} failed: {str(e)[:100]}...")
            continue

    print("\n" + "="*60)
    print("⚠️  No LLM available. Please set up one of these options:")
    print("="*60)
    print("Option 1 - OpenAI API:")
    # You can add fallback here if desired
    return None

# === New hybrid retrieval function combining dense + sparse retrieval ===
def hybrid_retrieve(vectorstore, bm25_retriever, query, k_embed=5, k_bm25=5):
    dense_hits = vectorstore.similarity_search(query, k=k_embed)
    sparse_hits = bm25_retriever.invoke(query)[:k_bm25]

    seen = set()
    combined = []
    for doc in dense_hits + sparse_hits:
        if doc.page_content not in seen:
            combined.append(doc)
            seen.add(doc.page_content)
    return combined

# === New prompt template encouraging reasoning and extrapolation ===
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are provided with the following retrieved information:
{context}

Based on this information:
- Identify key facts.
- Combine these facts logically.
- Deduce any implications or conclusions that follow, even if not explicitly stated.
- Show your reasoning step-by-step.

Question: {question}
Answer:
"""
)

def main():
    vectorstore = setup_rag_system()
    bm25_retriever = setup_bm25_retriever(vectorstore)
    llm = setup_llm()
    if llm is None:
        print("No LLM available. Exiting.")
        return

    print("\n💬 You can now ask questions! Type 'exit' to quit.\n")
    while True:
        user_input = input("Your question: ").strip()
        if user_input.lower() == "exit":
            print("👋 Goodbye!")
            break

        # Use hybrid retrieval
        retrieved_docs = hybrid_retrieve(vectorstore, bm25_retriever, user_input)
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        final_prompt = prompt_template.format(context=context, question=user_input)

        try:
            response = llm.invoke(final_prompt)
            print("\n📝 Answer:\n")
            print(response)
        except Exception as e:
            print(f"❌ LLM failed: {str(e)}")

if __name__ == "__main__":
    main()
