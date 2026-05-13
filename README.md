# rag-tutorial

Companion repository for the PyCon US 2026 tutorial, **"Implementing RAG in Python: Build a Retrieval-Augmented Generation System"** by Isabel Michel.

This project is a small, local-first introduction to Retrieval-Augmented Generation (RAG) in Python. The goal is to show how to build a grounded question-answering pipeline over a document corpus using lightweight open-source tools, rather than relying on a hosted black-box stack.

## What This Tutorial Covers

Large language models are good at generating fluent text, but they are weak when the answer depends on:

- recent information
- private documents
- domain-specific material
- verifiable, source-grounded answers

RAG addresses that by retrieving relevant context first and then passing that context into the model prompt.

In the tutorial, participants incrementally build a Python pipeline that:

- ingests and indexes a document corpus
- chunks text into retrieval-friendly units
- computes embeddings for those chunks
- retrieves relevant chunks with cosine similarity
- injects the retrieved context into an LLM prompt
- generates answers that are more grounded and explainable than a raw model response

The emphasis is not just on making a demo work. The tutorial focuses on the design choices that matter in real systems: chunk size, retrieval strategy, prompt construction, and evaluation tradeoffs.

## Who This Is For

This material is aimed at Python users with basic programming familiarity. Prior experience with LLMs, embeddings, vector databases, or RAG is not required.

If you understand Python scripts, packages, and notebooks, this repo is intended to be approachable.

## Project Goal

By the end of the tutorial, you should have:

- a working Python RAG prototype
- a mental model for the major moving parts in a RAG system
- a baseline implementation you can extend with better embeddings, larger corpora, or different LLMs

## What Is In This Repository Right Now

This repository is currently a minimal scaffold for the tutorial. At the moment it contains:

- `README.md`: project overview and usage notes
- `world-history-text.pdf`: the sample document corpus used for retrieval
- `.drive-setup/setup.py`: a bootstrap script that downloads the models and the sample PDF for offline use

The setup script downloads:

- `Qwen/Qwen2.5-0.5B-Instruct` for local generation
- `sentence-transformers/all-MiniLM-L6-v2` for text embeddings
- `world-history-text.pdf` if it is not already present

The script is intended to preload the assets needed to run the tutorial offline.

## Expected Architecture

The tutorial workflow follows a standard RAG pipeline:

1. Load the source document.
2. Extract or parse raw text from the corpus.
3. Split the text into chunks that are small enough to retrieve effectively but large enough to preserve meaning.
4. Convert each chunk into an embedding vector.
5. Store those vectors in a simple retrieval index.
6. Embed an incoming user question.
7. Compare the question embedding against stored chunk embeddings using cosine similarity.
8. Select the top matching chunks.
9. Build a prompt that includes both the user question and the retrieved context.
10. Ask the language model to answer using that retrieved evidence.

In compact form, the data flow looks like this:

```text
PDF/document corpus
  -> text extraction
  -> chunking
  -> embeddings
  -> similarity index
  -> retrieval
  -> prompt assembly
  -> LLM answer
```

## Why These Models

The current bootstrap script uses small, practical open models so the tutorial stays understandable and runnable on a local machine.

- `all-MiniLM-L6-v2` is a common baseline embedding model for semantic similarity tasks.
- `Qwen2.5-0.5B-Instruct` is small enough to use as an instructional local model while still demonstrating the retrieval-plus-generation pattern.

These are teaching defaults, not hard requirements. Once the pipeline works, you can swap in stronger embedding models, alternative retrievers, or a different LLM.

## Setup

### 1. Create a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install the bootstrap dependencies

At minimum, the local setup script needs:

```powershell
pip install huggingface_hub requests
```

If you plan to work from a notebook, you will also want Jupyter installed:

```powershell
pip install jupyter
```

### 3. Download the tutorial assets

```powershell
python .drive-setup\setup.py
```

This will cache the Hugging Face model artifacts locally and ensure that `world-history-text.pdf` exists in the repository root.

## Suggested Implementation Steps

If you are building out the rest of the tutorial code, a clean progression is:

1. Read the PDF and extract plain text.
2. Normalize and chunk the text.
3. Generate embeddings for each chunk.
4. Build a simple in-memory index.
5. Retrieve top-k chunks for a user query.
6. Construct a grounded prompt.
7. Generate an answer with the local LLM.
8. Inspect retrieval quality and refine chunking or prompting.

That progression keeps each part understandable and testable before moving to the next layer.

## What To Pay Attention To

The tutorial is most useful when you pay attention to the tradeoffs, not just the output.

- Smaller chunks may improve retrieval precision but lose context.
- Larger chunks preserve context but can reduce retrieval quality.
- Top-k retrieval changes how much evidence the model sees.
- Prompt wording affects whether the model stays grounded in the retrieved text.
- A working answer is not automatically a trustworthy answer; you still need to inspect retrieved evidence.

## Possible Extensions

Once the baseline pipeline works, useful next steps include:

- trying different chunking strategies
- replacing in-memory retrieval with a vector database
- swapping in a larger or more specialized embedding model
- comparing multiple LLMs for answer quality
- adding citations to returned answers
- measuring retrieval quality with a small evaluation set
- indexing multiple documents instead of a single PDF

## Status

This repository currently looks like a tutorial scaffold rather than a finished reference implementation. The core assets for an offline RAG exercise are present, but the main notebook or Python application that wires the full pipeline together is not currently checked in.

That means this README documents both:

- what the tutorial is intended to teach
- how the current repository is set up to support that work

## Source

This repository is based on the PyCon US 2026 tutorial page:

- [Implementing RAG in Python: Build a Retrieval-Augmented Generation System](https://us.pycon.org/2026/schedule/presentation/56/)

Session details from that page:

- Event: PyCon US 2026 Tutorial
- Presenter: Isabel Michel
- Experience level: Some experience
- Time: Wednesday, May 13, 2026, 1:30 p.m.-5:00 p.m.
