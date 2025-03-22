import ollama
import redis
import numpy as np
import os
from redis.commands.search.query import Query

# Initialize Redis connection
redis_client = redis.Redis(host="localhost", port=6379, db=0)

VECTOR_DIM = 768
INDEX_NAME = "embedding_index"
DOC_PREFIX = "doc:"
DISTANCE_METRIC = "COSINE"
TEXT_FOLDER = "processed_texts"  # Folder containing .txt files

# Prompt user to select an embedding model
embedding_models = {
    "1": "nomic-embed-text",
    "2": "jina/jina-embeddings-v2-base-en",
    "3": "granite-embedding:278m",
}

print("Select an embedding model:")
for key, model in embedding_models.items():
    print(f"{key}: {model}")

selected_model = None
while selected_model not in embedding_models:
    selected_model = input("Enter the number corresponding to your choice: ")

EMBEDDING_MODEL = embedding_models[selected_model]


# Prompt user to select an LLM model
llm_models = {
    "1": "llama3.2:latest",
    "2": "mistral",
}

print("Select an LLM model:")
for key, model in llm_models.items():
    print(f"{key}: {model}")

selected_llm_model = None
while selected_llm_model not in llm_models:
    selected_llm_model = input("Enter the number corresponding to your choice: ")

LLM_MODEL = llm_models[selected_llm_model]
print(f"Using LLM model: {LLM_MODEL}")

def create_hnsw_index():
    try:
        redis_client.execute_command(f"FT.DROPINDEX {INDEX_NAME} DD")
    except redis.exceptions.ResponseError:
        pass
    
    redis_client.execute_command(
        f"""
        FT.CREATE {INDEX_NAME} ON HASH PREFIX 1 {DOC_PREFIX}
        SCHEMA text TEXT
        embedding VECTOR HNSW 6 DIM {VECTOR_DIM} TYPE FLOAT32 DISTANCE_METRIC {DISTANCE_METRIC}
        """
    )
    print("Index created successfully.")


def get_embedding(text: str) -> list:
    response = ollama.embeddings(model=EMBEDDING_MODEL, prompt=text)
    return response["embedding"]


def store_embedding(doc_id: str, text: str, embedding: list):
    key = f"{DOC_PREFIX}{doc_id}"
    redis_client.hset(
        key,
        mapping={
            "text": text,
            "embedding": np.array(embedding, dtype=np.float32).tobytes(),  # Store as byte array
        },
    )
    print(f"Stored embedding for: {doc_id}")


def process_text_files():
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
    """Use Llama 3.2 to generate a response based on retrieved text."""
    context = "\n\n".join([f"Chunk {i+1}: {chunk}" for i, chunk in enumerate(matching_chunks)])

    prompt_to_send = (
        f"User's Question: {query}\n\n"
        f"Relevant Context (if applicable):\n{context}\n\n"
        "Your task: Answer the user's question as clearly and accurately as possible."
        "If the question is unclear or not actually a question, state that explicitly."
    )

    print("\n=== Final Prompt Sent to Llama ===\n")
    print(prompt_to_send)
    print("\n=== End of Prompt ===\n")

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an  assistant with computer science expertise. Your primary goal is to answer the user's question clearly and accurately."
                    "If the input does not contain a valid question, explicitly state: 'It looks like your input is not a question.'"
                    "Use retrieved text chunks **only if relevant**; otherwise, ignore them and rely on your own knowledge."
                ),
            },
            {"role": "user", "content": prompt_to_send}
        ],
    )
    return response["message"]["content"]



def perform_knn_search(query_text: str, k: int = 2):
    """Retrieve top k matching chunks and pass them to the LLM for final response generation."""
    embedding = get_embedding(query_text)
    q = (
        Query(f"*=>[KNN {k} @embedding $vec AS vector_distance]")
        .sort_by("vector_distance")
        .return_fields("text", "vector_distance")
        .dialect(2)
    )
    res = redis_client.ft(INDEX_NAME).search(
        q, query_params={"vec": np.array(embedding, dtype=np.float32).tobytes()}
    )

    matching_chunks = [doc.text for doc in res.docs]

    if not matching_chunks:
        print("No relevant matches found.")
        return

    print(f"\nTop {len(matching_chunks)} matching chunks retrieved:")
    for i, chunk in enumerate(matching_chunks):
        print(f"\nChunk {i+1}: {chunk[:300]}...")  # Display first 300 characters

    response = query_llm(query_text, matching_chunks)
    print(f"\nResponse from {LLM_MODEL}:\n{response}\n")


if __name__ == "__main__":
    create_hnsw_index()
    process_text_files()
    query = "What is redis?"
    perform_knn_search(query)
