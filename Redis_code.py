import ollama
import redis
import numpy as np
import os
from redis.commands.search.query import Query
from transformers import AutoModel

# Initialize Redis connection
redis_client = redis.Redis(host="localhost", port=6379, db=0)

# set constants
VECTOR_DIM = 768
INDEX_NAME = "embedding_index"
DOC_PREFIX = "doc:"
DISTANCE_METRIC = "COSINE"
TEXT_FOLDER = "processed_texts"  
selected_model = None
jina_model = None
selected_llm_model = None

#clear redis database if reindexing
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
    """
    Generate an embedding for the given text using the selected embedding model.

    This function uses either the Jina embeddings model or the Ollama embeddings
    model based on the global EMBEDDING_MODEL setting.

    Parameters:
    text (str): The input text to be embedded.

    Returns:
    list: A list of floats representing the embedding vector for the input text.
    """
    if EMBEDDING_MODEL == "jina-embeddings-v2-base-en":
        return jina_model.encode([text])[0].tolist()
    else:
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
    """
    This function processes all text files in the specified folder, reads their content,
    generates embeddings for the text using the selected embedding model, and stores the
    embeddings along with the text content in Redis.

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
    Query the Language Model (LLM) with a given question and relevant context.

    This function prepares a prompt by combining the user's query and relevant context
    from matching chunks. It then sends this prompt to the LLM for processing and returns
    the model's response.

    Parameters:
    query (str): The user's question or input to be answered by the LLM.
    matching_chunks (list): A list of text chunks that provide relevant context for the query.
    """
    context = "\n\n".join([f"Chunk {i+1}: {chunk}" for i, chunk in enumerate(matching_chunks)])
    prompt_to_send = (
        f"User's Question: {query}\n\n"
        f"Relevant Context (if applicable):\n{context}\n\n"
        "Your task: Answer the user's question as clearly and accurately as possible."
        "If the question is unclear or not actually a question, state that explicitly."
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
    Perform a K-Nearest Neighbors (KNN) search on the Redis index using the given query text.

    This function embeds the query text, searches for similar embeddings in the Redis index,
    retrieves matching text chunks, and generates a response using a language model.

    Parameters:
    query_text (str): The text query to search for in the index.
    k (int, optional): The number of nearest neighbors to retrieve. Defaults to 2.

    """
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
    create_hnsw_index()
    process_text_files()
    query = input("What question do you want to ask? ")
    perform_knn_search(query)
