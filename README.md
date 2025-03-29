# Setup Instructions for Ollama Models and Docker Servers

This guide will walk you through installing Ollama models and running Redis and ChromaDB servers using Docker.

## Prerequisites

- You need to have [Ollama](https://ollama.com/) installed on your system.
- You need to have [Docker](https://www.docker.com/) installed for running Redis and ChromaDB servers.

## Step 1: Install Ollama Models

To install the required models from Ollama, open your terminal and run the following commands:

1. **Install `nomic-embed-text` model:**
   ```bash
   ollama pull nomic-embed-text
   ```
2. **Install `granite-embedding:278m` model:**
   ```bash
   ollama pull granite-embedding:278m
   ```

3. **Install `jina/jina-embeddings-v2-base-en` model:**
   ```bash
   ollama pull jina/jina-embeddings-v2-base-en
   ```

4. **Install `llama3.2:latest` model:**
   ```bash
   ollama pull llama3.2:latest
   ```
5. **Install `mistral` model:**
   ```bash
   ollama pull mistral
   ```
## Step 2: Run Redis Server via Docker

If you don't have a Redis server running, you can use Docker to pull and run the official Redis image.

1. **Pull the `redis` Docker Image:**
   ```bash
   docker pull redis
   ```
2. **Pull the `chromaDB` Docker Image:**
   ```bash
   docker pull ghcr.io/chroma-core/chroma
   ```

2. **Run the Docker Images in a server:**
   ```bash
   docker run --name chromadb-server -p 8000:8000 -d ghcr.io/chroma-core/chroma
   ```
   ```bash
   docker run --name redis-server -p 6379:6379 -d redis
   ```
3. **Verify that the servers are running:**
   ```bash
   docker ps
   ```
## Step 3: Install Required Python Libraries**
```bash
pip install -r requirements.txt
```
## Step 4:

Run the Python File of the Database you want or, for Timing/Storage demonstrations, run the Timing and Storage versions!


   
