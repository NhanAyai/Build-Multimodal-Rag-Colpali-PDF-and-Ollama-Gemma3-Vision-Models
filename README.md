# 📁🧠👀 Build-Multimodal-Rag-Colpali-PDF-and-Ollama-Gemma3-Vision-Models

## 📝 Description
This project demonstrates how to build a powerful Multimodal Retrieval-Augmented Generation (RAG) system capable of understanding both text and images within PDF documents. It leverages the Colpali library to index PDF files, extracting both text and image information. Subsequently, it uses an Ollama-hosted Gemma3 vision model (specifically the 4 billion parameter version or larger) to answer user queries based on the indexed content of these PDFs. This system allows you to ask questions that require understanding not just the text but also the visual information present in your PDF documents.

<div align="center">
  
  ![Image](https://github.com/user-attachments/assets/82750688-90af-407c-9881-b41fa75c1f1b)
  
</div>

## ✨ Key Features
* **Multimodal Understanding:** Processes and understands both textual and visual information from PDF documents.
* **PDF Indexing with Colpali:** Utilizes the `byaldi` library to load a pre-trained RAG model and index PDF documents, storing both text and base64 representations of images.
* **Ollama Integration:** Seamlessly integrates with Ollama to run the Gemma3 vision model locally.
* **Gemma3 Vision Inference:** Employs the state-of-the-art Gemma3 model (4b or larger) for reasoning about the combined text and image content retrieved by the ColPali model.
* **RAG Architecture:** Implements a Retrieval-Augmented Generation pipeline where Colpali handles the retrieval of relevant multimodal information, and Ollama Gemma3 performs the generation of the answer.

## 🛠️ Installation
Follow these steps to set up the project environment:
1.  **Install Dependencies:**
    Run the following commands in a Google Colab notebook or a similar environment to install the necessary libraries and system utilities:
    ```bash
    pip install -q byaldi
    sudo apt-get install -y poppler-utils
    pip install -q git+https://github.com/huggingface/transformers.git qwen-vl-utils flash-attn optimum auto-gptq bitsandbytes
    pip install -q ollama
    pip install -q colab-xterm
    pip install -q triton
    sudo apt-get update
    sudo apt-get install poppler-utils
    ```
2.  **Load ColPali RAG Model:**
    The project uses a pre-trained ColPali RAG model for indexing. This is loaded using the following Python code:
    ```python
    import os
    import base64
    from byaldi import RAGMultiModalModel
    RAG = RAGMultiModalModel.from_pretrained("vidore/colpali-v1.2", verbose=1)
    ```
3.  **Get a Sample Document:**
    A sample PDF document ("Attention is all you need" paper) is downloaded for demonstration:
    ```bash
    wget https://arxiv.org/pdf/1706.03762
    mkdir docs
    mv 1706.03762 docs/attention.pdf
    ```
    You can replace this with your own PDF documents.
4.  **Index the PDF:**
    The ColPali RAG model is used to index the PDF document. This process extracts text and stores base64 representations of images:
    ```python
    RAG.index(
        input_path="./docs/attention.pdf",
        index_name="attention",
        store_collection_with_index=True,  # Store base64 representation of images
        overwrite=True
    )
    ```
5.  **Set Up Ollama and Gemma3 Vision:**
    Ollama is installed and the Gemma3 vision model (4b) is pulled:
    ```bash
    sudo apt update
    sudo apt install -y pciutils
    curl -fsSL https://ollama.com/install.sh | sh
    import threading
    import subprocess
    import time
    def run_ollama_serve():
        subprocess.Popen(["ollama", "serve"])
    thread = threading.Thread(target=run_ollama_serve)
    thread.start()
    time.sleep(5)
    ollama pull gemma3:4b
    ```
    This sets up the local Ollama server and downloads the necessary Gemma3 model.

## 🚀 Usage
The core functionality for performing inference is encapsulated in the `inference` function:
```python
from IPython.display import Image
import ollama
import base64

def see_image(image_base64):
    """Decodes and displays a base64 encoded image."""
    image_bytes = base64.b64decode(image_base64)
    filename = 'image.jpg'
    with open(filename, 'wb') as f:
        f.write(image_bytes)
    display(Image(filename))

def inference(question: str):
    """
    Performs inference using the Gemma3 vision model.
    It retrieves relevant data using the loaded ColPali RAG model,
    displays the retrieved image, and then queries the Ollama Gemma3 model
    with the question and the local image.
    """
    results = RAG.search(question, k=1)  # Retrieve relevant data using ColPali
    see_image(results[0]['base64'])  # Display the retrieved image
    response = ollama.chat(
        model='gemma3:4b',
        messages=[{
            'role': 'user',
            'content': question,
            'images': ['image.jpg']
        }]
    )
    return response['message']['content']

# Example query
inference_result = inference("Hãy giải thích bằng tiếng Việt figure 1.")
print(inference_result)
```

## 🔄 Workflow
  
```mermaid
graph LR
    A[User Query] --> B(Retrieval)
    B -- Text Snippets --> C{Gemma3 Vision Model}
    B -- Image Data --> C
    D[PDF Document] --> E(Colpali Image Extraction)
    D --> F(Text Extraction)
    E --> B
    F --> B
    C -- Answer --> G[User Response]
