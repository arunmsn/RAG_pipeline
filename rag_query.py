from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama  # Assuming you're using LangChain wrapper for Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.retrievers import BM25Retriever
from langchain.schema import Document

# Load vector DB (Chroma + embeddings)
embeddings_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
persist_directory = "chroma_db_binder"

vectorstore = Chroma(
    collection_name="my_binder_collection",
    embedding_function=embeddings_model,
    persist_directory=persist_directory
)

# Build BM25 retriever using Chroma's documents
docs = vectorstore.get(include=["documents"])["documents"]
bm25_retriever = BM25Retriever.from_documents([Document(page_content=doc) for doc in docs])

# Build hybrid retrieval function
def hybrid_retrieve(query, k_embed=5, k_bm25=5):
    dense_hits = vectorstore.similarity_search(query, k=k_embed)
    sparse_hits = bm25_retriever.invoke(query)[:k_bm25]
    
    # Combine + deduplicate
    seen = set()
    combined = []
    for doc in dense_hits + sparse_hits:
        if doc.page_content not in seen:
            combined.append(doc)
            seen.add(doc.page_content)
    return combined

# Initialize Ollama3 LLM
ollama_llm = Ollama(model="ollama3")  # Adjust if your setup calls it differently

# Prompt template that encourages extrapolation
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

# Build QA chain with custom retriever + prompt
def rag_extrapolate(query):
    retrieved_docs = hybrid_retrieve(query)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    formatted_prompt = prompt.format(context=context, question=query)
    
    response = ollama_llm.invoke(formatted_prompt)
    return response

# Example usage
if __name__ == "__main__":
    user_query = input("Ask your question: ")
    answer = rag_extrapolate(user_query)
    print("\n---ANSWER---\n")
    print(answer)
