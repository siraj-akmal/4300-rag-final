import chromadb
import ollama
import os
from transformers import AutoModel

# Initialize ChromaDB connection
chroma_client = chromadb.HttpClient(host="localhost", port=8000)

# Set constants
COLLECTION_NAME = "ds4300-rag"
TEXT_FOLDER = "processed_texts"  
selected_model = None
jina_model = None
selected_llm_model = None

# Ensure collection exists and clear it at the start of each run
def get_or_create_collection():
    try:
        chroma_client.delete_collection(COLLECTION_NAME)  # Clear existing data
    except:
        pass  # Collection might not exist yet
    return chroma_client.create_collection(COLLECTION_NAME)

collection = get_or_create_collection()

def get_embedding(text: str) -> list:
    """
    Generate an embedding for the given text using the selected embedding model.
    """
    if EMBEDDING_MODEL == "jina-embeddings-v2-base-en":
        return jina_model.encode([text])[0].tolist()
    else:
        response = ollama.embeddings(model=EMBEDDING_MODEL, prompt=text)
        return response["embedding"]

def store_embedding(doc_id: str, text: str, embedding: list):
    """
    Store the document and its embedding in ChromaDB.
    """
    collection.add(ids=[doc_id], embeddings=[embedding], documents=[text])
    print(f"Stored embedding for: {doc_id}")

def process_text_files():
    """
    Reads text files, generates embeddings, and stores them in ChromaDB.
    """
    if not os.path.exists(TEXT_FOLDER):
        print(f"Folder '{TEXT_FOLDER}' not found.")
        return

    text_files = [f for f in os.listdir(TEXT_FOLDER) if f.endswith(".txt")]
    if not text_files:
        print("No text files found.")
        return

    for filename in text_files:
        filepath = os.path.join(TEXT_FOLDER, filename)
        with open(filepath, "r", encoding="utf-8") as file:
            text = file.read()
            embedding = get_embedding(text)
            store_embedding(filename, text, embedding)

def query_llm(query: str, matching_chunks: list) -> str:
    """
    Query the LLM with a given question and relevant context.
    """
    context = "\n\n".join([f"Chunk {i+1}: {chunk}" for i, chunk in enumerate(matching_chunks)])
    prompt_to_send = (
        f"User's Question: {query}\n\n"
        f"Relevant Context:\n{context}\n\n"
        "Your task: Answer the user's question as clearly as possible."
    )
    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": "You are an AI assistant with expertise in computer science."},
            {"role": "user", "content": prompt_to_send}
        ],
    )
    return response["message"]["content"]

def perform_knn_search(query_text: str, k: int = 2):
    """
    Perform a KNN similarity search in ChromaDB.
    """
    embedding = get_embedding(query_text)
    results = collection.query(query_embeddings=[embedding], n_results=k)

    if not results['documents'][0]:
        print("No relevant matches found.")
        return

    matching_chunks = results['documents'][0]
    print(f"\nTop {len(matching_chunks)} matching chunks retrieved:")
    for i, chunk in enumerate(matching_chunks):
        print(f"\nChunk {i+1}: {chunk[:300]}...")

    response = query_llm(query_text, matching_chunks)
    print(f"\nResponse from {LLM_MODEL}:\n{response}\n")

# Prompt user to select an embedding model
embedding_models = {
    "1": "nomic-embed-text",
    "2": "jina-embeddings-v2-base-en",
    "3": "granite-embedding:278m",
}

print("Select an embedding model:")
for key, model in embedding_models.items():
    print(f"{key}: {model}")

while selected_model not in embedding_models:
    selected_model = input("Enter the number corresponding to your choice: ")

EMBEDDING_MODEL = embedding_models[selected_model]

# If Jina embeddings are selected, load the model
if EMBEDDING_MODEL == "jina-embeddings-v2-base-en":
    jina_model = AutoModel.from_pretrained("jinaai/jina-embeddings-v2-base-en", trust_remote_code=True)

# Prompt user to select an LLM model
llm_models = {
    "1": "llama3.2:latest",
    "2": "mistral",
}

print("Select an LLM model:")
for key, model in llm_models.items():
    print(f"{key}: {model}")

while selected_llm_model not in llm_models:
    selected_llm_model = input("Enter the number corresponding to your choice: ")

LLM_MODEL = llm_models[selected_llm_model]
print(f"Using LLM model: {LLM_MODEL}")

if __name__ == "__main__":
    # process text files loads the parsed notes into the database
    #process_text_files()
    query = input("What question do you want to ask? ")
    # acctually performs the semantic search and queries the LLM
    perform_knn_search(query)
