from pypdf import PdfReader

reader = PdfReader("data/corpus/631_water drinking standards.pdf")

full_text = ""
for page in reader.pages:
    full_text += page.extract_text()

print("Total characters:", len(full_text))

def chunk_text(text, chunk_size=800, overlap=100):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap

    return chunks


chunks = chunk_text(full_text)

print("Number of chunks:", len(chunks))
print("\n--- Example chunk ---\n")
print(chunks[0])