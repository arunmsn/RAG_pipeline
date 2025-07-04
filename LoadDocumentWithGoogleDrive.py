from langchain_community.document_loaders import DirectoryLoader, UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

# Define the directory where your documents are stored
DOCS_DIR = "/Users/sathia.m/sathia.m@reddit.com - Google Drive/Shared drives/SPACE/People & Teams/SPACE Consulting/Privacy Impact Assessment Questionnaires "

# To get the folder ID, go to your Google Drive folder and copy the ID from the URL:
# https://drive.google.com/drive/folders/FOLDER_ID_HERE
# You'll need to replace this with your actual folder ID
GOOGLE_DRIVE_FOLDER_ID = "YOUR_FOLDER_ID_HERE"  # Replace with your actual folder ID

print("Loading Privacy Impact Assessment documents (with Google Drive support)...")

# Initialize documents list
documents = []

# Option 1: Load Google Docs (.gdoc files) using GoogleDriveLoader
print("Loading Google Docs (.gdoc files)...")
try:
    from langchain_google_community import GoogleDriveLoader
    
    # Load Google Docs from the specified folder
    google_loader = GoogleDriveLoader(
        folder_id=GOOGLE_DRIVE_FOLDER_ID,
        # Credentials will be automatically managed
        # First run will prompt for authentication
        recursive=True,  # Include subfolders
        file_types=["document", "sheet"],  # Load Google Docs and Sheets
    )
    
    google_docs = google_loader.load()
    documents.extend(google_docs)
    print(f"Loaded {len(google_docs)} Google Docs")
    
    # Print the names of loaded Google Docs
    for doc in google_docs:
        print(f"  - {doc.metadata.get('title', 'Unnamed Google Doc')}")
        
except ImportError:
    print("❌ langchain-google-community not installed")
    print("Install with: pip install langchain-google-community[drive]")
except Exception as e:
    print(f"❌ Could not load Google Docs: {e}")
    print("Make sure you have set up Google Drive API credentials")

# Option 2: Load local files (including .docx files)
print("\nLoading local files...")
local_count = 0
for root, _, files in os.walk(DOCS_DIR):
    for file in files:
        file_path = os.path.join(root, file)
        if file.lower().endswith(('.pdf', '.txt', '.docx', '.html', '.md')):
            try:
                loader = UnstructuredFileLoader(file_path)
                docs = loader.load()
                documents.extend(docs)
                local_count += 1
                print(f"  - {file}")
            except Exception as e:
                print(f"❌ Error loading {file}: {e}")

print(f"\nLoaded {local_count} local files")
print(f"Total documents loaded: {len(documents)}")

# Split Documents into Chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    add_start_index=True,
)

chunks = text_splitter.split_documents(documents)
print(f"Total chunks created: {len(chunks)}")

print("\n✅ Complete document loading finished!")
print(f"📊 Summary:")
print(f"   • Google Docs: {len([d for d in documents if 'google' in d.metadata.get('source', '').lower()])}")
print(f"   • Local files: {local_count}")
print(f"   • Total chunks: {len(chunks)}")

# Uncomment to see a sample chunk:
# if chunks:
#     print(f"\n📄 Sample chunk preview:")
#     print(f"{chunks[0].page_content[:300]}...")
#     print(f"Source: {chunks[0].metadata.get('source', 'Unknown')}") 