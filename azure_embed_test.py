import os
from openai import AzureOpenAI

client = AzureOpenAI(
    api_version="2024-12-01-preview",
    azure_endpoint="https://cds-ds-openai-001-x.openai.azure.com/",
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
)


text = "Water disinfection is essential for safe drinking water."

response = client.embeddings.create(
    model="text-embedding-3-small",
    input=text,
)

embedding = response.data[0].embedding

print("Embedding length:", len(embedding))
print("First 10 values:", embedding[:10])
