# Swirl-ai-engineering-bootcamp-02

This is part of the third sprint of the SwirlAI Engineering Bootcamp. It is a hybrid RAG pipeline that uses Qdrant for vector storage and retrieval, OpenAI for embeddings and generation, LangSmith for observability, instructor for structured outputs, and streamlit for the UI.

---

## 🚀 Getting Started

If you do need to run the code, follow these steps:

### 1. Clone the Repository

```bash
git clone https://github.com/SL2226799/swirlai-ai-engineering-bootcamp-02.git
cd swirlai-ai-engineering-bootcamp-02
```

### 2. Copy Environment File

```bash
cp env.example .env
```

### 3. Edit `.env` and Add Your API Keys

```env
OPENAI_API_KEY=your_openai_api_key
GOOGLE_API_KEY=your_google_api_key
GROQ_API_KEY=your_groq_api_key
QDRANT_URL=your_qdrant_url
QDRANT_COLLECTION_NAME=your_qdrant_collection_name
EMBEDDING_MODEL=your_embedding_model
EMBEDDING_MODEL_PROVIDER=your_embedding_model_provider
GENERATION_MODEL=your_generation_model
GENERATION_MODEL_PROVIDER=your_generation_model_provider
LANGSMITH_API_KEY=your_langsmith_api_key
```

> ✅ Keep the remaining configuration as per `.env.example`.

---

## 🐳 Run the Project

```bash
make run-docker-compose
```


## 📚 Citation

This repository uses data provided by the authors of the following paper.  
If you use this work, please cite:

```bibtex
@article{hou2024bridging,
  title={Bridging Language and Items for Retrieval and Recommendation},
  author={Hou, Yupeng and Li, Jiacheng and He, Zhankui and Yan, An and Chen, Xiusi and McAuley, Julian},
  journal={arXiv preprint arXiv:2403.03952},
  year={2024}
}
```