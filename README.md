# QA System with Docker using RAG

## Overview

This project provides a QA system that uses Retrieval-Augmented Generation (RAG) and semantic retrieval methods, based on cosine similarities of embeddings. The service is containerized using Docker, allowing for easy deployment and management.

The `RAG system` uses `gpt-4o` LLM by default. It works by creating a vector database hosted in [Elastic Cloud](https://www.elastic.co/cloud), in order to store the documents and using them for Retrieval-Augmented Generation. It returns the LLM-generated answer along with the sources, ie. document name and page number. The prompt used for this RAG is also given in the prompt.txt file in the utils folder.

The `Semantic_retrieval system` uses the `MiniLM-L6-v2` model by default. It works by embedding the document chunks as well as the queries. Then, it finds the `cosine similarities` between the documents and queries and finds the nearest document chunks matching the query. You can set the threshold for this similarity measure as well as the number of document chunks to return. It returns the chunk of documents as answer along with the sources, ie. document name.

If the system cannot find an answer to the question, it returns `Sorry this information is not present in our documents.` Sources are empty in that case.

## Project Structure

- `main.py`: Contains the FastAPI application and endpoint definitions.
- `Dockerfile`: Defines the Docker image for running the FastAPI application.
- `utils/`: Contains utility scripts for text extraction, embedding, and more.
- `models/`: Contains scripts for model management and vector storage.
- `data/`: Contains all the documents required for RAG.

## Getting Started

Follow these steps to run the application using Docker.

### Prerequisites

1. **Docker**: Ensure Docker is installed on your system. You can download Docker from [here](https://www.docker.com/get-started).

### Building the Docker Image

1. **Navigate to the Project Directory**

   Open a terminal or command prompt and navigate to the directory containing your `Dockerfile`:

   ```bash
   cd /path/to/your/project

2. **Build the Docker Image**

   Run the following command to build the Docker image. Replace `my-fastapi-app` with a name of your choice for the image:
    ```bash
   docker build -t my-fastapi-app .
### Running the Docker Container

1. **Run the Docker Container**
    Use the following command to start a Docker container from the image you built. This will expose port 8000 on your host machine:

    ```bash
   docker run -p 8080:8080 `                                                                                                --env USER_QUERY="How many personal leaves do I have ?" `
   --env CHUNK_SIZE=500 `
   --env CHUNK_OVERLAP=0 `
   --env METHOD="RAG" `
   my-fastapi-app
   ```
### Understanding the Dockerfile

```Dockerfile
# Use the official Python image from the Docker Hub
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose port 8000 for the FastAPI application
EXPOSE 8000

# Command to run the FastAPI application using uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Here's a brief overview of what each section in `Dockerfile` does:

- `FROM python:3.9-slim`: This line specifies the base image for the container, which is a minimal version of Python 3.9. Using a slim image reduces the overall size of the Docker image and minimizes unnecessary packages.

- `WORKDIR /app`: Sets the working directory inside the container to /app. All subsequent commands will be run from this directory.

- `COPY requirements.txt .`: Copies the requirements.txt file from your local directory to the container's working directory. This file typically lists the Python packages required by your application.

- `RUN pip install --no-cache-dir -r requirements.txt`: Installs the Python dependencies specified in requirements.txt. The --no-cache-dir option prevents caching of package files, which helps keep the Docker image size smaller.

- `COPY . .`: Copies the rest of your application code into the container's working directory.

- `EXPOSE 8000`: Informs Docker that the container will listen on port 8000. This is necessary to make the port accessible when the container is running.

- `CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]`: Specifies the default command to run when the container starts. This command uses `uvicorn` to run the FastAPI application, binding to all network interfaces (`0.0.0.0`) and exposing port 8000.




### Arguments in Docker run
Here's a brief overview of what each argument in the `Docker run` command does:

- `USER_QUERY`: Query for which you need the answer.

- `CHUNK_SIZE`: Size of chunks in which to split each documents for looking up. Default = `500`

- `CHUNK_OVERLAP`: Overlap between the chunks in which the documents are split. Default = `0`

- `METHOD`: can take value `RAG` or `Semantic_retrieval`. Default = `RAG`

- `RETURN_METADATA`: Whether to return metadata when getting the ouput from the `RAG system`. Default = `true`

- `TEMPERATURE`: Temperature that regulates the randomness of the LLM responses, in case of `RAG system`. Default = `0.0`

- `MODEL_NAME`: Name of the LLM for `RAG system`. Default = `gpt-4o`

- `EMBEDDING_MODEL`: Name of the model which is used for embedding the documents for the `Semantic_retrieval method`. Default = `sentence-transformers/all-MiniLM-L6-v2`

- `THRESHOLD`: Threshold for similarity score between document embedding and query embedding for the `Semantic_retrieval method`. Default = `0.1`

- `TOP_K`: number of document chunks with similarity more than threshold value to show in the output of the `Semantic_retrieval method`. Default = `3`


   
