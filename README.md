# Electronics & AI Research Assistant

An intelligent Streamlit app designed to **answer technical questions based on research papers** in the fields of Electronics and Artificial Intelligence. Powered by **Retrieval-Augmented Generation (RAG)**, this project combines advanced language models, semantic search, and document embeddings to deliver precise, context-aware answers grounded in real scientific literature.

---

## Project Aim

The goal of this project is to **bridge the gap between dense academic research papers and accessible knowledge** by enabling users to interactively query complex Electronics and AI research documents. Inspired by the paper *“Overview of Emerging Electronics Technologies for AI / LLM Ideas”*, the app aims to:

- Empower researchers, engineers, and enthusiasts to quickly extract meaningful insights without manually reading voluminous texts.
- Facilitate faster decision-making and learning by providing accurate answers grounded in cutting-edge Electronics and AI technologies.
- Demonstrate how Retrieval-Augmented Generation can be applied to domain-specific research, enhancing accessibility and usability of technical knowledge.

---

## Why I Built This

Reading and understanding technical research papers, especially in fast-evolving fields like Electronics and AI, can be overwhelming and time-consuming. I built this assistant to:

- **Automate knowledge extraction** from complex scientific texts using state-of-the-art AI.
- Explore the practical integration of LangChain, Groq's LLM, and OpenAI embeddings in a real-world research setting.
- Showcase the potential of **RAG techniques** for building intelligent assistants that combine deep language understanding with document retrieval.
- Create a tool that helps myself and others stay up-to-date with emerging AI-driven electronics technologies by asking natural language questions instead of sifting through pages of dense content.

---

##  How It Works

1. **PDF Loading**: Research papers stored in the `research_papers` folder are loaded using `PyPDFDirectoryLoader`.
2. **Chunking**: Documents are split into overlapping text chunks (e.g., 1000 characters with 200 overlap) by `RecursiveCharacterTextSplitter` to improve retrieval precision.
3. **Embedding**: Each chunk is transformed into a vector representation using `OpenAIEmbeddings`, capturing semantic meaning.
4. **Vector Store**: These vectors are indexed and stored in a FAISS similarity search database (`FAISS`).
5. **Retrieval & Generation**: Upon receiving a question, the system retrieves relevant chunks and generates answers using Groq’s Llama 3.1 model via LangChain’s `create_retrieval_chain` and `ChatGroq`.
6. **Prompting**: `ChatPromptTemplate` structures the query with context for accurate, source-grounded answers.

>  **Note:** Vector embeddings are created once per session, ensuring subsequent queries are answered instantly with minimal latency.

---

##  Technical Components

| Module/Library | Role |
| -------------- | ---- |
| `ChatGroq` (from `langchain_groq`) | Groq's optimized LLM for answer generation. |
| `OpenAIEmbeddings` | Converts text chunks into semantic vectors. |
| `RecursiveCharacterTextSplitter` | Splits documents into smaller, overlapping chunks. |
| `create_stuff_documents_chain` | Combines retrieved document chunks into a unified prompt. |
| `create_retrieval_chain` | Integrates retrieval and generation into a seamless pipeline. |
| `FAISS` | High-speed vector search engine for similarity matching. |
| `PyPDFDirectoryLoader` | Loads PDFs from a directory into document objects. |
| `ChatPromptTemplate` | Defines the prompt structure sent to the LLM. |

---

## Setup & Usage

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/electronics-ai-research-assistant.git
   cd electronics-ai-research-assistant
   ```
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Configure API keys**:
   ```bash
   Create a .env file with your API keys:
    OPENAI_API_KEY=your_openai_api_key
    GROQ_API_KEY=your_groq_api_key
   ```
4. **Add your PDFs**:
   ```bash
   Place your Electronics and AI research papers in the research_papers/ folder.
   ```
5. **Run the app**:
  ``` bash
   streamlit run app.py
  ```
---

## Folder Structure:
```bash
project/
│
├── app.py                      # Streamlit application code
├── research_papers/           # Place your PDF research papers here
├── .env                       # Environment variables for API keys
├── requirements.txt           # Python dependencies
└── README.md                  # This documentation file

```
---
## Impact and Use Cases

This project makes Electronics and AI research **more accessible and actionable**:

- Researchers can get instant answers to specific technical questions, speeding up literature reviews.
- Engineers and developers gain a quick-reference assistant to validate concepts or explore new tech without exhaustive reading.
- Students and learners get an AI-powered tutor for complex subjects by querying trusted scientific sources.
- The underlying RAG architecture serves as a blueprint for building **domain-specific AI assistants** in other fields.
---

## Acknowledgments
 - Special thanks to the authors of “Overview of Emerging Electronics Technologies for AI and LLM Ideas” for their invaluable research and insights that made this project possible.
 - Thanks to the teams behind LangChain, Groq, OpenAI, and Streamlit for the incredible tools and APIs.

---

























