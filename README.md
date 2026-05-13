# rag-tutorial

Companion repository for the PyCon US 2026 tutorial, **"Implementing RAG in Python: Build a Retrieval-Augmented Generation System"** by Isabel Michel.

This project is a local-first wrapper around the instructor's tutorial materials. It vendors the upstream repo as a git submodule and provides a `uv`-managed environment plus local setup notes so the tutorial can run reproducibly from this repository.

The goal is to show how to build a grounded question-answering pipeline over a document corpus using lightweight open-source tools, rather than relying on a hosted black-box stack.

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
- `pyproject.toml`: the `uv`-managed dependency definition for the tutorial environment
- `uv.lock`: the locked dependency set for reproducible installs
- `vendor/rag_tutorial`: the upstream tutorial materials as a git submodule
- `world-history-text.pdf`: the sample document corpus used for retrieval
- `.drive-setup/setup.py`: a bootstrap script that downloads the models and the sample PDF for offline use

The setup script downloads:

- `Qwen/Qwen2.5-0.5B-Instruct` for local generation
- `sentence-transformers/all-MiniLM-L6-v2` for text embeddings
- `world-history-text.pdf` if it is not already present

The script is intended to preload the assets needed to run the tutorial offline.

For PDF parsing, this repository now uses `PyMuPDF`, which is commonly imported in tutorial code as `fitz`.

The upstream materials in the submodule are now:

- `vendor/rag_tutorial/python_tutorial.py`
- `vendor/rag_tutorial/requirements.txt`
- `vendor/rag_tutorial/setup.py`

This wrapper repo still uses `pyproject.toml` and `uv.lock` as its source of truth for dependency management.

One upstream inconsistency to be aware of:

- `python_tutorial.py` still mentions `warmup.py`
- the actual download/bootstrap script checked into the submodule is `setup.py`

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

### 1. Fetch the upstream tutorial files

If you cloned this repository yourself, initialize the submodule first:

```powershell
git submodule update --init --recursive
```

The upstream entrypoint now lives at:

- `vendor/rag_tutorial/python_tutorial.py`

### 2. Install `uv`

This repository now uses `uv` and a checked-in [`pyproject.toml`](./pyproject.toml) as the source of truth for Python dependencies.

If you do not already have `uv` installed, install it with one of the official methods documented here:

- <https://docs.astral.sh/uv/getting-started/installation/>

### 3. Create or sync the project environment

To install the tutorial dependencies and the notebook tooling:

```powershell
uv sync --extra notebook
```

`uv` will create and manage the project's `.venv` automatically from `pyproject.toml` and `uv.lock`.

The environment is aligned to the upstream tutorial imports and includes:

- `PyMuPDF` / `fitz`
- `pandas`
- `spaCy`
- `tqdm`
- `matplotlib`
- `torch`
- `sentence-transformers`
- `transformers` pinned to `4.46.x`
- `accelerate`

If you only want the base Python dependencies and not the notebook tooling:

```powershell
uv sync
```

### 4. Download the tutorial assets

```powershell
uv run python .drive-setup\setup.py
```

This will cache the Hugging Face model artifacts locally and ensure that `world-history-text.pdf` exists in the repository root.

If you have already run the setup script successfully, you do not need to run it again.

The emailed setup script downloads:

- `Qwen/Qwen2.5-0.5B-Instruct`
- `sentence-transformers/all-MiniLM-L6-v2`
- `world-history-text.pdf`

### 5. Understand the offline model-selection caveat

The upstream tutorial script forces offline mode and then chooses an LLM based on detected hardware:

- CPU: `Qwen/Qwen2.5-0.5B-Instruct`
- Apple Silicon / MPS: `Qwen/Qwen2.5-0.5B-Instruct`
- CUDA with under ~8 GB VRAM: `Qwen/Qwen2.5-3B-Instruct`
- CUDA with ~8 GB+ VRAM: `google/gemma-3-4b-it`

That means the emailed warmup assets are enough for the CPU and Apple-Silicon defaults, but not necessarily enough for the CUDA defaults in the notebook.

If you only cached the emailed assets and want guaranteed offline behavior, either:

- keep the notebook on the default CPU path, or
- manually set `model_id = "Qwen/Qwen2.5-0.5B-Instruct"` in the script before loading the LLM, or
- pre-cache the larger CUDA-selected model yourself

### 6. Launch the tutorial environment

```powershell
uv run jupyter lab
```

The upstream tutorial now ships as a Python script with `# %%` cells rather than a checked-in notebook:

- `vendor/rag_tutorial/python_tutorial.py`

Use it in an editor that understands `# %%` cells, or open it from JupyterLab as a source file if that fits your workflow.

If you want to use the upstream download helper instead of the wrapper copy, the submodule also includes:

- `vendor/rag_tutorial/setup.py`

Run scripts inside the managed environment with `uv run`, for example:

```powershell
uv run python some_script.py
```

If the tutorial code uses `fitz`, the installed package is `PyMuPDF`. In your own code, prefer:

```python
import pymupdf as fitz
```

This matches the PyMuPDF docs and avoids confusion with the unrelated `fitz` package on PyPI.

### 7. Dependency workflow

Dependency management now lives in [`pyproject.toml`](./pyproject.toml).

- Add or change dependencies in `pyproject.toml` or with `uv add`.
- Regenerate the lockfile with `uv lock` or `uv sync`.
- Keep setup changes documented in this README when the workflow changes.

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

## Questions

### Can I version the models so repeated runs keep the same weights?

Yes. If you are downloading models from Hugging Face, the reliable way to keep the same weights is to pin each model to an exact `revision` (ideally a commit SHA) when calling `snapshot_download()` or `from_pretrained()`.

For example:

```python
snapshot_download("Qwen/Qwen2.5-0.5B-Instruct", revision="<commit-sha>")
snapshot_download("sentence-transformers/all-MiniLM-L6-v2", revision="<commit-sha>")
```

That gives you a fixed snapshot even if the model author later updates the repo. Once cached locally, offline runs will keep using that same snapshot.

This repository does **not** currently pin Hugging Face model revisions in `.drive-setup/setup.py`; it downloads those model repos by name only. That is usually fine for a tutorial, but it is not the strongest reproducibility guarantee.

Also note that not every chunking step uses model weights. In this tutorial, chunking is mainly sentence splitting and text grouping, so reproducibility there mostly comes from pinning Python package versions in `pyproject.toml` and `uv.lock`. But if you switch to a model-backed parser, chunker, reranker, or embedder, the same revision-pinning rule applies there too.

### If I want a larger embedding model, what are the cost concerns?

There are two separate costs to think about:

- a larger embedding **model**, which usually means slower embedding generation and higher RAM or VRAM usage
- a larger embedding **dimension**, which makes every stored vector bigger and increases retrieval/index cost

For a local RAG setup like this one, the main costs are usually:

- slower corpus indexing and re-indexing
- higher memory use while generating embeddings
- larger on-disk vector storage
- larger in-memory retrieval indexes
- slower similarity search as the corpus grows

As a rule of thumb, raw vector storage grows roughly linearly with dimension. A float32 embedding uses about `4 bytes * embedding_dimension` per chunk, so:

- `384` dimensions is about `1.5 KB` per chunk
- `768` dimensions is about `3 KB` per chunk
- `1536` dimensions is about `6 KB` per chunk

That means one million chunks would require roughly:

- `~1.5 GB` of raw vectors at `384` dimensions
- `~3 GB` of raw vectors at `768` dimensions
- `~6 GB` of raw vectors at `1536` dimensions

Those numbers do not include metadata or approximate-nearest-neighbor index overhead, so the real footprint will be higher.

For this tutorial-sized corpus, a larger model is usually manageable. The cost becomes more important when you have enough chunks that re-embedding time, RAM pressure, or vector storage dominate the system.

### How do I choose a good balance so the model is not too heavy but still good?

Do not optimize for size alone. The practical goal is to use the smallest embedding model that still meets your retrieval-quality target.

A good workflow is:

1. Set simple limits up front for acceptable indexing time, query latency, and storage.
2. Build a small evaluation set of real questions and the chunks you would expect to retrieve.
3. Start with a small baseline model such as `sentence-transformers/all-MiniLM-L6-v2`.
4. Compare it against one medium-size candidate rather than jumping straight to a very large model.
5. Measure retrieval quality with something concrete such as `Recall@k` or by manually checking whether the right chunks appear near the top.
6. Keep the smallest model that stays close to the best-performing candidate on your evaluation set.

In many cases, a smaller well-matched model is better than a larger generic one. Domain fit and training quality often matter more than raw embedding dimension.

It is also often cheaper to improve the system before upgrading the model:

- use better chunk boundaries
- tune chunk size
- tune `top-k`
- add reranking for only the top retrieved chunks

For this repository, a reasonable default is to keep the current embedding model first, test retrieval quality on real questions, and only move up if the baseline clearly misses relevant passages.

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

This repository is now best understood as a wrapper around the upstream tutorial materials.

- The source tutorial files are checked in via the `vendor/rag_tutorial` submodule.
- This repo adds the environment definition, local setup script, documentation, and notes needed to run it reproducibly.
- The tutorial logic still primarily lives in the upstream script rather than in a packaged Python application.

## Source

This repository is based on the PyCon US 2026 tutorial page:

- [Implementing RAG in Python: Build a Retrieval-Augmented Generation System](https://us.pycon.org/2026/schedule/presentation/56/)
- [Upstream notebook repository](https://github.com/ismichel/rag_tutorial)

Session details from that page:

- Event: PyCon US 2026 Tutorial
- Presenter: Isabel Michel
- Experience level: Some experience
- Time: Wednesday, May 13, 2026, 1:30 p.m.-5:00 p.m.
