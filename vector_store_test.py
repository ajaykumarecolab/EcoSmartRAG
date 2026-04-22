import os
from pypdf import PdfReader
from openai import AzureOpenAI
import chromadb

# Azure OpenAI client
client = AzureOpenAI(
    api_version="2024-12-01-preview",
    azure_endpoint="https://cds-ds-openai-001-x.openai.azure.com/",
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
)

# Read PDF
reader = PdfReader("data/corpus/631_water drinking standards.pdf")
text = ""
for page in reader.pages:
    text += page.extract_text()

# Chunking
def chunk_text(text, chunk_size=800, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

chunks = chunk_text(text)

# Create ChromaDB client (persistent)
chroma_client = chromadb.Client(
    chromadb.config.Settings(
        persist_directory="chroma_db"
    )
)

collection = chroma_client.get_or_create_collection(
    name="water_docs"
)

# Embed and store chunks
for i, chunk in enumerate(chunks):
    embedding = client.embeddings.create(
        model="text-embedding-3-small",
        input=chunk
    ).data[0].embedding

    collection.add(
        documents=[chunk],
        embeddings=[embedding],
        ids=[str(i)]
    )

print("Stored", collection.count(), "chunks in ChromaDB")
