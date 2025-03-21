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
2. **Install `ranite-embedding:278m` model:**
   ```bash
   ollama pull ranite-embedding:278m
   ```
3. **Install `llama3.2:latest` model:**
   ```bash
   ollama pull llama3.2:latest
   ```
4. **Install `mistral` model:**
   ```bash
   ollama pull mistral
   ```
