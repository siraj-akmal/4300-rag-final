import ollama
import faiss
import numpy as np
import os
import time
import psutil

# Define constants
VECTOR_DIM = 768
TEXT_FOLDER = "processed_texts"  # Folder containing .txt files
k = 2  # Number of nearest neighbors to retrieve
filenames = [f for f in os.listdir(TEXT_FOLDER) if f.endswith(".txt")]

# Initialize FAISS index (using L2 distance)
index = faiss.IndexFlatL2(VECTOR_DIM)

# Function to get the current memory usage of the system in MB
def get_system_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)  # Convert bytes to MB

def get_stable_memory_usage():
    """
    Ensures that memory readings are stable by taking multiple samples
    and ensuring no large fluctuations occur before accepting the reading.
    """
    initial_memory = get_system_memory_usage()
    time.sleep(0.1)  # Delay for a short time to allow system processes to stabilize
    stable_memory = get_system_memory_usage()
    
    # If the memory fluctuates within 5% of the first reading, we take it as stable
    while abs(stable_memory - initial_memory) > 0.05 * initial_memory:
        time.sleep(0.1)  # Wait a bit before trying again
        stable_memory = get_system_memory_usage()

    return stable_memory

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


def get_embedding(text: str) -> list:
    """Get the embedding for the given text using Ollama."""
    try:
        response = ollama.embeddings(model=EMBEDDING_MODEL, prompt=text)
        return response["embedding"]
    except Exception as e:
        print(f"Error generating embedding for text: {e}")
        return None  # Return None if there's an error, and skip storing the embedding


def store_embedding(doc_id: str, text: str, embedding: list):
    """Store the text and its embedding in FAISS."""
    if embedding is not None:  # Only store if embedding is valid
        embedding = np.array(embedding, dtype=np.float32).reshape(1, VECTOR_DIM)
        index.add(embedding)  # Add embedding to FAISS index
        print(f"Stored embedding for: {doc_id}")
    else:
        print(f"Skipping file {doc_id} due to embedding generation error.")


def process_text_files():
    """Process .txt files in the folder, generate embeddings, and store them in FAISS."""
    if not os.path.exists(TEXT_FOLDER):
        print(f"Folder '{TEXT_FOLDER}' not found.")
        return

    text_files = [f for f in os.listdir(TEXT_FOLDER) if f.endswith(".txt")]
    if not text_files:
        print("No text files found.")
        return

    # Record start time for embedding process
    start_embedding_time = time.time()

    # Get initial system memory usage
    initial_system_memory = get_stable_memory_usage()

    for i, filename in enumerate(text_files):
        filepath = os.path.join(TEXT_FOLDER, filename)
        try:
            with open(filepath, "r", encoding="utf-8") as file:
                text = file.read()
                embedding = get_embedding(text)
                store_embedding(i, text, embedding)  # Use index i as the document ID
        except Exception as e:
            print(f"Error processing file {filename}: {e}")
            continue  # Skip this file and continue with the next one

    # Record end time for embedding process
    end_embedding_time = time.time()

    # Calculate total embedding time
    embedding_time = end_embedding_time - start_embedding_time
    print(f"\nTotal embedding time: {embedding_time:.2f} seconds")
    
    # Get final system memory usage
    final_system_memory = get_stable_memory_usage()

    # Calculate memory usage in MB
    memory_used_for_embeddings = final_system_memory - initial_system_memory
    print(f"Memory used for embeddings: {memory_used_for_embeddings:.2f} MB")


def query_llm(query: str, matching_chunks: list) -> str:
    """Use Llama 3.2 to generate a response based on retrieved text."""
    context = "\n\n".join([f"Chunk {i + 1}: {chunk}" for i, chunk in enumerate(matching_chunks)])

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
        messages=[{
            "role": "system",
            "content": (
                "You are an assistant with computer science expertise. Your primary goal is to answer the user's question clearly and accurately."
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
    if embedding is None:
        print("Embedding generation failed, cannot perform search.")
        return

    query_embedding = np.array(embedding, dtype=np.float32).reshape(1, VECTOR_DIM)

    # Get initial system memory before performing the LLM query
    initial_system_memory = get_stable_memory_usage()

    # Start timer for KNN search and LLM query execution
    start_query_time = time.time()

    # Perform the KNN search on FAISS index
    D, I = index.search(query_embedding, k)  # D = distances, I = indices of the closest vectors
    # Retrieve the corresponding filenames using the indices
    matching_chunks = []
    for idx in I[0]:
        if idx == -1:
            continue  # Skip invalid indices
        filename = filenames[idx]  # Use the index to retrieve the filename
        file_path = os.path.join(TEXT_FOLDER, filename)

        # Open and read the corresponding text file
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding="utf-8") as f:
                matching_chunks.append(f.read())
        else:
            print(f"Warning: File {file_path} not found.")

    if not matching_chunks:
        print("No relevant matches found.")
        return

    print(f"\nTop {len(matching_chunks)} matching chunks retrieved:")
    for i, chunk in enumerate(matching_chunks):
        print(f"\nChunk {i + 1}: {chunk[:300]}...")  # Display first 300 characters

    # Generate the response from LLM
    response = query_llm(query_text, matching_chunks)
    print(f"\nResponse from {LLM_MODEL}:\n{response}\n")

    # Record end time for the LLM query execution
    end_query_time = time.time()

    # Calculate time taken for KNN search and LLM query
    query_time = end_query_time - start_query_time
    print(f"LLM query execution time: {query_time:.2f} seconds")

    # Get final system memory after performing the LLM query
    final_system_memory = get_stable_memory_usage()

    # Calculate memory usage for the LLM query in MB
    memory_used_for_query = final_system_memory - initial_system_memory
    print(f"Memory used for LLM query: {memory_used_for_query:.2f} MB")


if __name__ == "__main__":
    process_text_files()
    query = "When is it better to use AVL tree vs BST tree?"
    perform_knn_search(query)