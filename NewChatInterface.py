import os
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.schema import Document

def setup_vectorstore():
    print("🔄 Loading your Privacy Impact Assessment RAG system...")

    embeddings_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    persist_directory = "chroma_db_binder"

    vectorstore = Chroma(
        collection_name="my_binder_collection",
        embedding_function=embeddings_model,
        persist_directory=persist_directory
    )

    print(f"✅ Loaded vector database with {vectorstore._collection.count()} chunks")
    return vectorstore

def setup_bm25_retriever(vectorstore):
    docs = vectorstore.get(include=["documents"])["documents"]
    return BM25Retriever.from_documents([Document(page_content=doc) for doc in docs])

def setup_llm():
    print("\n🤖 Setting up language model...")

    ollama_models = [
        "llama3",
        "mistral",
        "codellama",
        "phi3"
    ]

    for model_name in ollama_models:
        try:
            print(f"Trying Ollama model: {model_name}...")
            llm = Ollama(model=model_name, base_url="http://localhost:11434")
            llm.invoke("Hello")  # Simple test
            print(f"✅ Using Ollama model: {model_name}")
            return llm
        except Exception as e:
            print(f"❌ Model {model_name} failed: {str(e)[:100]}...")

    print("\n⚠️ No LLM available. Please set up Ollama or another provider.")
    return None

def hybrid_retrieve(vectorstore, bm25_retriever, query, k_embed=5, k_bm25=5):
    dense_hits = vectorstore.similarity_search(query, k=k_embed)
    sparse_hits = bm25_retriever.invoke(query)[:k_bm25]

    # Deduplicate
    seen = set()
    combined = []
    for doc in dense_hits + sparse_hits:
        if doc.page_content not in seen:
            combined.append(doc)
            seen.add(doc.page_content)
    return combined

def build_prompt(context, question):
    template = """
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
    prompt = PromptTemplate(input_variables=["context", "question"], template=template)
    return prompt.format(context=context, question=question)

def main():
    vectorstore = setup_vectorstore()
    bm25_retriever = setup_bm25_retriever(vectorstore)
    llm = setup_llm()
    if llm is None:
        return

    print("\n💬 You can now ask questions! Type 'exit' to quit.\n")
    while True:
        user_input = input("Your question: ").strip()
        if user_input.lower() == "exit" or user_input.lower() == "bye":
            print("👋 Goodbye!")
            break

        retrieved_docs = hybrid_retrieve(vectorstore, bm25_retriever, user_input)
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        final_prompt = build_prompt(context, user_input)

        try:
            response = llm.invoke(final_prompt)
            print("\n📝 Answer:\n")
            print(response)
        except Exception as e:
            print(f"❌ LLM failed: {str(e)}")

if __name__ == "__main__":
    main()



