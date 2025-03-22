import ollama
import faiss
import numpy as np
import os

# Define constants
VECTOR_DIM = 768
TEXT_FOLDER = "processed_texts"  # Folder containing .txt files
k = 2  # Number of nearest neighbors to retrieve
filenames = [f for f in os.listdir(TEXT_FOLDER) if f.endswith(".txt")]

# Initialize FAISS index (using L2 distance)
index = faiss.IndexFlatL2(VECTOR_DIM)

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
    response = ollama.embeddings(model=EMBEDDING_MODEL, prompt=text)
    return response["embedding"]


def store_embedding(doc_id: str, text: str, embedding: list):
    """Store the text and its embedding in FAISS."""
    embedding = np.array(embedding, dtype=np.float32).reshape(1, VECTOR_DIM)
    index.add(embedding)  # Add embedding to FAISS index
    print(f"Stored embedding for: {doc_id}")


def process_text_files():
    """Process .txt files in the folder, generate embeddings, and store them in FAISS."""
    if not os.path.exists(TEXT_FOLDER):
        print(f"Folder '{TEXT_FOLDER}' not found.")
        return

    text_files = [f for f in os.listdir(TEXT_FOLDER) if f.endswith(".txt")]
    if not text_files:
        print("No text files found.")
        return

    for i, filename in enumerate(text_files):
        filepath = os.path.join(TEXT_FOLDER, filename)
        with open(filepath, "r", encoding="utf-8") as file:
            text = file.read()
            embedding = get_embedding(text)
            store_embedding(i, text, embedding)  # Use index i as the document ID


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
    query_embedding = np.array(embedding, dtype=np.float32).reshape(1, VECTOR_DIM)

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


if __name__ == "__main__":
    process_text_files()
    query = "When is it better to use AVL tree vs BST tree?"
    perform_knn_search(query)
