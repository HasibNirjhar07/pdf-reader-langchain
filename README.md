# 🧠 LangChain Semantic Search Engine

This project is one of the best ways to **learn LangChain step-by-step**, as it demonstrates how LLMs *retrieve*, *understand*, and *respond* with real data.

---

## 🌍 Overview

We're building a **Semantic Search Engine**.

### 🚀 What it does

- You upload a document (e.g., a company report or article).  
- Then you can ask natural language questions like:  
  > "How many distribution centers does Nike have in the US?"  
- The system retrieves the **most relevant part** of the document and answers based on it.

---

## 💡 Why "Semantic" Search?

Because it understands **meaning**, not just **keywords**.

| Query              | Keyword Search                                  | Semantic Search                                                   |
| ------------------ | ----------------------------------------------- | ----------------------------------------------------------------- |
| "Nike US branches" | Might not find "distribution centers in the US" | ✅ Understands both mean similar things and retrieves correct info |

Semantic search uses **embeddings** (vector representations of meaning) instead of simple word matching.

---

## 🧩 Core Concepts

You'll learn five main components:

1. **Document Loader** — Reads your files (PDF, txt, etc.)  
2. **Text Splitter** — Breaks long text into smaller chunks  
3. **Embeddings** — Converts text into numeric vectors  
4. **Vector Store** — Saves and searches those vectors  
5. **Retriever** — Finds relevant chunks for your query  

---

## 🪜 Step-by-Step Guide

### 🧾 Step 1: Document Loader

A **document loader** reads your file and turns it into structured `Document` objects.

Each document has:
- `page_content`: the actual text
- `metadata`: details like file name or page number
```python
from langchain_core.documents import Document

documents = [
    Document(page_content="Dogs are loyal pets.", metadata={"source": "pets.pdf"}),
    Document(page_content="Cats are independent.", metadata={"source": "pets.pdf"})
]
```

To load a PDF:
```python
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("nke-10k-2023.pdf")
docs = loader.load()
```

Now `docs[0]` is Page 1, `docs[1]` is Page 2, etc.

✅ **Purpose:** Extracts and structures raw text from your file.

---

### ✂️ Step 2: Text Splitter

Large pages are divided into smaller, overlapping **chunks** for better search precision.
```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

all_splits = splitter.split_documents(docs)
```

✅ **Why overlap?**  
It preserves context between chunks — so related sentences stay connected.

✅ **Purpose:** Makes text searchable and improves retrieval accuracy.

---

### 🔢 Step 3: Embeddings

Embeddings are **numeric vectors** representing text meaning.

| Sentence             | Embedding (shortened)      |
| -------------------- | -------------------------- |
| "Dogs are friendly." | `[0.12, -0.04, 0.89, ...]` |
| "Cats are friendly." | `[0.11, -0.03, 0.88, ...]` |

They're close in vector space because they mean similar things.
```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vector = embeddings.embed_query("Dogs are great companions")

print(len(vector))  # 1536
```

✅ **Purpose:** Converts text into numerical form so meaning can be compared mathematically.

---

### 🧠 Step 4: Vector Store

Stores and searches embeddings efficiently.
```python
from langchain_core.vectorstores import InMemoryVectorStore

vector_store = InMemoryVectorStore(embeddings)
vector_store.add_documents(all_splits)
```

**Searching:**
```python
results = vector_store.similarity_search(
    "How many distribution centers does Nike have in the US?"
)
```

Output:
```
"NIKE has eight significant distribution centers in the United States..."
```

✅ **Purpose:** Quickly finds chunks with meanings closest to the question.

---

### 🧩 Step 5: Retriever

A **retriever** wraps the vector store to easily fetch relevant text for any query.
```python
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 1})

retriever.batch([
    "How many distribution centers does Nike have in the US?",
    "When was Nike incorporated?",
])
```

✅ **Purpose:** Provides an interface for question-based retrieval — the foundation for RAG (Retrieval-Augmented Generation).

---

### 🤖 Step 6: Minimal RAG (Retrieval + LLM)

RAG combines:

1. **Retriever** → fetches relevant text
2. **LLM** → uses that text to answer the question

Example:
```python
context = retriever.invoke("When was Nike founded?")
prompt = f"Answer the question using the context:\n{context}\n\nQuestion: When was Nike founded?"

response = llm.invoke(prompt)
print(response)
```

✅ **Purpose:** The LLM responds with factual, document-based answers.

---

## 🧭 Summary Table

| Step | Concept         | Purpose                    | Example                         |
| ---- | --------------- | -------------------------- | ------------------------------- |
| 1    | Document Loader | Load text from files       | Read PDF pages                  |
| 2    | Text Splitter   | Break text into chunks     | Split 1000 chars with overlap   |
| 3    | Embeddings      | Convert text to numbers    | `[0.12, -0.04, 0.89, ...]`      |
| 4    | Vector Store    | Save & search embeddings   | `similarity_search()`           |
| 5    | Retriever       | Fetch relevant text chunks | `.batch()`                      |
| 6    | RAG             | Combine retriever + LLM    | "Answer using document context" |

---

## 📚 Real-Life Analogy

Imagine your PDF is a **library**:

| LangChain Concept | Library Analogy                                                   |
| ----------------- | ----------------------------------------------------------------- |
| Document Loader   | Scanning each book page                                           |
| Text Splitter     | Breaking pages into paragraphs                                    |
| Embeddings        | Creating unique "meaning fingerprints" for each paragraph         |
| Vector Store      | A searchable card catalog of meanings                             |
| Retriever         | The librarian who finds the right info                            |
| RAG               | The librarian + expert who explains the answer in simple language |

---

## 🧩 Next Steps

You can extend this project by:

* Adding **UI with Streamlit** for file upload & question input
* Using **FAISS or Chroma** instead of in-memory store
* Integrating with **OpenAI or Anthropic models** for responses

---

## 🧑‍💻 Run the Project

### 1️⃣ Install dependencies
```bash
pip install langchain langchain-community langchain-openai langchain-text-splitters
```

### 2️⃣ Set your API key
```bash
export OPENAI_API_KEY="your_api_key_here"
```

### 3️⃣ Run your script
```bash
python app.py
```

---

## 🧩 Author

**Hasibul Islam Nirjhar**  
Software Engineering Student at IUT 🇧🇩  
☕ Debugs with coffee, solves bugs with pizza 🍕

---
