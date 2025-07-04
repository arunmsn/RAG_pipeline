from langchain.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Load the vector store (if not already loaded from the previous step)
# embeddings_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2") # Use the SAME embedding model
# persist_directory = "chroma_db_binder"
# vectorstore = Chroma(
#     collection_name="my_binder_collection",
#     embedding_function=embeddings_model,
#     persist_directory=persist_directory
# )

# 4.1 Initialize the local LLM using Ollama
# Ensure 'ollama run mistral' (or your chosen model) is running in a separate terminal
llm = Ollama(model="mistral") # <--- Use the same model name you pulled with Ollama

# 4.2 Set up the RAG Chain
# This chain will:
# 1. Take your question.
# 2. Embed your question using the embedding_function.
# 3. Search the vectorstore for relevant chunks.
# 4. Pass the retrieved chunks and your question to the LLM.
# 5. Get the answer from the LLM.

# A more specific prompt template can help the LLM focus
prompt_template = """
Use the following pieces of context from the documents to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Please cite the source document(s) by their file name if available, otherwise by their chunk index.
If there are multiple sources, cite all of them.

Context:
{context}

Question: {question}

Helpful Answer:"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff", # 'stuff' puts all retrieved docs in one prompt. Other types exist for more docs.
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}), # Retrieve top 5 most relevant chunks
    return_source_documents=True, # Important for citing sources
    chain_type_kwargs={"prompt": PROMPT}
)

# 4.3 Run the e-Discovery Query Loop
print("\n--- E-Discovery Assistant ---")
print("Type 'exit' to quit.")

while True:
    query = input("\nYour question: ")
    if query.lower() == 'exit':
        break

    result = qa_chain({"query": query})
    answer = result["result"]
    source_documents = result["source_documents"]

    print(f"\nAnswer: {answer}")

    if source_documents:
        print("\nSources:")
        for i, doc in enumerate(source_documents):
            source_name = doc.metadata.get('source', f"Chunk {i+1}") # Try to get filename
            start_index = doc.metadata.get('start_index', 'N/A')
            print(f"- {source_name} (from index {start_index})")
            # print(f"  Content snippet: {doc.page_content[:200]}...") # Uncomment to see snippets
    else:
        print("No specific sources found for this answer.")