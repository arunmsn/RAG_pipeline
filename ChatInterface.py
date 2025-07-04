import os
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama

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

def setup_llm():
    """Set up the language model - tries multiple options"""
    print("\n🤖 Setting up language model...")
    
#   # Option 1: Try OpenAI (if API key is set)
#   if os.getenv("OPENAI_API_KEY"):
#       print("Using OpenAI GPT model")
#        return ChatOpenAI(
#            model="gpt-3.5-turbo",
#            temperature=0.1,
#            max_tokens=1000

    
    # Option 2: Try Ollama (local model) - try common model names
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
    
    # Option 3: Fallback message
    print("\n" + "="*60)
    print("⚠️  No LLM available. Please set up one of these options:")
    print("="*60)
    print("Option 1 - OpenAI API:")
    print("  export OPENAI_API_KEY='your-api-key-here'")
    print("\nOption 2 - Ollama (free, local):")
    print("  1. Make sure Ollama is running: ollama serve")
    print("  2. Pull a model: ollama pull llama3.2")
    print("  3. Or run directly: ollama run llama3.2")
    print("  4. Check what models you have: ollama list")
    print("="*60)
    return None

def simple_chat():
    """Simple command-line chat interface"""
    print("\n" + "="*60)
    print("🎯 Privacy Impact Assessment Chat Interface")
    print("="*60)
    print("Ask questions about your Privacy Impact Assessment documents!")
    print("Type 'quit', 'exit', or 'bye' to end the conversation.")
    print("="*60)
    
    # Set up the RAG system
    vectorstore = setup_rag_system()
    llm = setup_llm()
    
    if llm is None:
        print("\n❌ Cannot start chat without a language model. Please set up OpenAI API or Ollama first.")
        return
    
    # Create the RAG chain
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 5}  # Retrieve top 5 most relevant chunks
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        verbose=False
    )
    
    print("\n🚀 Chat is ready! Ask your first question:")
    
    while True:
        # Get user input
        user_question = input("\n🤔 You: ").strip()
        
        # Check for exit commands
        if user_question.lower() in ['quit', 'exit', 'bye', 'q']:
            print("\n👋 Goodbye! Your RAG system is always here when you need it.")
            break
        
        if not user_question:
            print("Please ask a question about your Privacy Impact Assessment documents.")
            continue
        
        try:
            print("\n🔍 Searching documents...")
            
            # Get response from RAG system
            result = qa_chain.invoke({"query": user_question})
            
            # Display the answer
            print(f"\n🤖 Assistant: {result['result']}")
            
            # Optionally show sources
            if result.get('source_documents'):
                print(f"\n📚 Sources: Found information in {len(result['source_documents'])} documents")
                for i, doc in enumerate(result['source_documents'][:2], 1):
                    source = doc.metadata.get('source', 'Unknown source')
                    print(f"  {i}. {os.path.basename(source)}")
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Try rephrasing your question or check your setup.")

def test_retrieval():
    """Test the retrieval system with sample questions"""
    print("\n🧪 Testing retrieval system...")
    
    vectorstore = setup_rag_system()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    test_questions = [
        "privacy impact assessment",
        "data processing",
        "vendor security",
        "compliance requirements"
    ]
    
    for question in test_questions:
        print(f"\n🔍 Testing query: '{question}'")
        docs = retriever.get_relevant_documents(question)
        print(f"  Found {len(docs)} relevant chunks")
        if docs:
            # Show first few words of the most relevant document
            preview = docs[0].page_content[:100].replace('\n', ' ')
            print(f"  Preview: {preview}...")

if __name__ == "__main__":
    print("🎯 Privacy Impact Assessment RAG System")
    print("\nChoose an option:")
    print("1. Start chat interface")
    print("2. Test retrieval system")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    if choice == "1":
        simple_chat()
    elif choice == "2":
        test_retrieval()
    else:
        print("Invalid choice. Starting chat interface...")
        simple_chat() 