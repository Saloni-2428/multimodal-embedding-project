# 🎯 Multimodal Embedding Project

## 📌 Overview
This project demonstrates how to build a **multimodal embedding pipeline** that integrates text, PDF documents, audio, and images into a unified vector space using **FAISS** for similarity search.  
It leverages **Sentence Transformers**, **OpenAI Whisper**, and **CLIP** to generate embeddings across modalities, enabling powerful search and retrieval.

---

## 🚀 Features
- **Text embeddings** using SentenceTransformer (`all-MiniLM-L6-v2`)
- **PDF parsing** with PyPDF2
- **Audio transcription** with Whisper
- **Image embeddings** with CLIP
- **Vector search** using FAISS
- Unified storage and retrieval of multimodal content

---

## 🛠️ Installation
Clone the repository and install dependencies:

```bash
git clone https://github.com/Saloni-2428/multimodal-embedding-project.git
cd multimodal-embedding-project
pip install -r requirements.txt


---

## 🧪 Usage / Example Code
You can run the main script (`train.py`) to add text, PDF, audio, and image embeddings into FAISS and perform semantic search.

### Quick Example
```python
from sentence_transformers import SentenceTransformer
import faiss

# Load text model
text_model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize FAISS index
dimension = 384
index = faiss.IndexFlatL2(dimension)
documents = []

def add_text(text, source):
    vec = text_model.encode([text]).astype("float32")
    index.add(vec)
    documents.append(f"{source}: {text[:80]}")

# Add a sample text
add_text("Artificial Intelligence is transforming healthcare.", "Text file")

# Search
def search(query, k=3):
    qvec = text_model.encode([query]).astype("float32")
    _, idx = index.search(qvec, k)
    return [documents[i] for i in idx[0]]

results = search("How is AI used in healthcare?")
print("🔎 Search Results:", results)




📂 Project Structure
multimodal-embedding-project/
│── README.md                # Project documentation
│── requirements.txt         # Python dependencies
│── train.py                 # Main script for embeddings & search
│── data/                    # Sample data folder
│   ├── sample_text.txt
│   ├── sample_document.pdf
│   ├── sample_audio.wav
│   └── sample_image.jpg
│── utils/                   # Helper functions (optional)
│   └── preprocessing.py
│── models/                  # Pretrained or fine-tuned models
│── notebooks/               # Jupyter notebooks for experiments
│   └── demo.ipynb



🧠 Models Used
SentenceTransformer: Text embeddings
Whisper: Audio transcription
CLIP: Image embeddings
FAISS: Vector similarity search


---

## 📊 Results
- Integrates **text, PDF, audio, and image embeddings** into a single FAISS index.
- Enables **semantic search** across modalities.
- Example query: *“How is AI used in healthcare?”* returns relevant snippets from text, PDF, audio transcription, and image content.

---

## 📚 Datasets
You can experiment with:
- [COCO Dataset](https://cocodataset.org/)
- [Flickr30k Dataset](https://www.bing.com/search?q=Flickr30k+dataset)

---

## 🤝 Contributing
Contributions are welcome!  
1. Fork the repo  
2. Create a new branch  
3. Commit changes  
4. Submit a pull request  

--

## 📬 Contact
Created by **Saloni-2428**  
For questions or collaborations, reach out via [GitHub profile](https://github.com/Saloni-2428).
