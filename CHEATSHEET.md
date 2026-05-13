# RAG Tutorial Cheatsheet

This cheat sheet is intentionally limited to the scope of the PyCon US 2026 tutorial and this repo.

- Tutorial goal: build a small Python Retrieval-Augmented Generation (RAG) system.
- Tutorial scope from the conference page: ingest a document corpus, retrieve with cosine similarity, inject retrieved context into prompts, and produce grounded answers.
- Repo-specific assets: `world-history-text.pdf` and `.drive-setup/setup.py`.
- Repo-specific models from `setup.py`: `sentence-transformers/all-MiniLM-L6-v2` for embeddings and `Qwen/Qwen2.5-0.5B-Instruct` for generation.

The code examples below use a practical stack inferred from that scope:

- `PyMuPDF` for PDF text extraction
- `spaCy` for sentence splitting
- `sentence-transformers` for embeddings
- `numpy` for similarity and ranking
- `torch` for tensor search and top-k retrieval
- `transformers` for local text generation

## 1. The Mental Model

RAG is just this:

```text
document -> text -> chunks -> embeddings -> similarity search -> prompt -> answer
```

A normal LLM tries to answer from what it learned during training.

A RAG system answers from:

- the user question
- the retrieved document chunks
- the model's language ability

The retrieval step is the key difference.

## 2. Terms You Need To Know

### Corpus

The set of documents you search over.

In this repo, the starting corpus is one PDF:

- `world-history-text.pdf`

### Document ingestion

Turning source files into raw text your Python code can work with.

For this tutorial, ingestion mostly means:

- open the PDF
- extract text page by page
- combine it into one long string

### Chunk

A chunk is one small piece of document text.

Why chunk at all:

- whole documents are too large and too vague to retrieve well
- tiny fragments lose context
- chunks give retrieval something useful to rank

### Embedding

An embedding is a list of numbers representing meaning.

Two texts with similar meaning should have embeddings that are close together in vector space.

### Query

The user's question, turned into text and then into an embedding.

### Index

The searchable structure that holds your chunks and embeddings.

For this tutorial, an index does not need to be a database. A plain Python list of chunks plus a NumPy array of embeddings is enough.

### Cosine similarity

The score used to compare two vectors by angle rather than raw size.

Formula:

```text
cos(a, b) = dot(a, b) / (||a|| * ||b||)
```

Interpretation:

- closer to `1.0`: very similar
- around `0.0`: unrelated
- below `0.0`: opposite direction, usually not useful here

Tutorial note:

- the upstream notebook often uses dot product for ranking because normalized embeddings make dot product and cosine behave very similarly
- the underlying retrieval idea is still the same: compare vectors and rank the closest ones

### Top-k retrieval

Pick the `k` highest-scoring chunks.

Example:

- `top_k = 3` means use the 3 best chunks as context

### Prompt construction

Build the text you send to the LLM.

A good RAG prompt usually includes:

- the task
- the retrieved context
- the user question
- a rule to avoid unsupported guesses

### Grounded answer

An answer that is supported by retrieved text.

### Hallucination

A confident answer that is not supported by the retrieved context.

RAG reduces hallucinations, but does not remove them.

## 3. What You Are Probably Building

For this tutorial, the simplest useful workflow is:

1. Read the PDF.
2. Extract text.
3. Split the text into chunks.
4. Embed each chunk once.
5. Embed the user's question.
6. Rank chunks by cosine similarity.
7. Put the best chunks into the prompt.
8. Ask the LLM to answer only from that context.

That is enough for a real beginner RAG prototype.

## 4. Small Code Patterns

### Read text from the PDF

```python
import pymupdf as fitz

doc = fitz.open("world-history-text.pdf")
pages = [page.get_text("text", sort=True) for page in doc]
text = "\n".join(pages)

print(text[:500])
```

What to know:

- `page.get_text("text")` extracts plain text from each page.
- `sort=True` can improve reading order by sorting text top-left to bottom-right.
- PDF extraction is often messy. Broken spacing and line breaks are normal.
- `PyMuPDF` is the installed package; many tutorials still refer to it as `fitz`.

### Clean text a little

```python
import re

def normalize_text(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()

text = normalize_text(text)
```

What to know:

- do light cleanup first
- do not over-clean and accidentally remove meaning

### Split text into chunks

The upstream notebook uses sentence-based chunking with spaCy, then groups several sentences together.

```python
from spacy.lang.en import English

nlp = English()
nlp.add_pipe("sentencizer")

doc = nlp(text)
sentences = [str(sent).strip() for sent in doc.sents if str(sent).strip()]

def chunk_sentences(sentences: list[str], sentences_per_chunk: int = 10) -> list[str]:
    chunks = []

    for i in range(0, len(sentences), sentences_per_chunk):
        chunk = " ".join(sentences[i:i + sentences_per_chunk]).strip()
        if chunk:
            chunks.append(chunk)

    return chunks

chunks = chunk_sentences(sentences, sentences_per_chunk=10)
print(len(chunks))
print(chunks[0][:200])
```

What to know:

- sentence-based chunks are often easier to reason about than raw character windows
- too few sentences per chunk can lose context
- too many sentences per chunk can make retrieval broad and noisy

Reasonable beginner defaults:

- `5` to `10` sentences per chunk for a first pass
- experiment after you inspect retrieval quality

### Turn chunks into embeddings

```python
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
chunk_vectors = embedder.encode(chunks)
```

What to know:

- you embed the corpus once, then reuse those vectors
- embedding the whole corpus for every question is wasteful

### Embed the user's question

```python
question = "Why did decolonization accelerate after World War II?"
query_vector = embedder.encode(question)
```

What to know:

- query text is usually short
- chunk text is usually longer
- that short-question/long-passage setup is normal in RAG

### Rank chunks with cosine similarity

```python
import torch

chunk_tensor = torch.tensor(chunk_vectors, dtype=torch.float32)
query_tensor = torch.tensor(query_vector, dtype=torch.float32)

scores = torch.matmul(chunk_tensor, query_tensor)
top_k = 3
top_scores, top_ids = torch.topk(scores, k=top_k)

retrieved_chunks = [chunks[i] for i in top_ids.tolist()]

for score, chunk in zip(top_scores.tolist(), retrieved_chunks):
    print(score)
    print(chunk[:200], "\n")
```

What to know:

- retrieval is just ranking by score
- the upstream notebook uses `torch.topk` to get the best matches
- inspect the retrieved chunks before blaming the LLM
- bad retrieval usually causes bad answers

### Build a grounded prompt

```python
context = "\n\n".join(retrieved_chunks)

prompt = f"""You are answering questions about the provided history text.
Use only the context below.
If the answer is not in the context, say: "I don't know based on the provided context."

Context:
{context}

Question: {question}

Answer:
"""
```

What to know:

- the prompt should explicitly constrain the model
- "use only the context" is a simple and effective beginner rule
- telling the model how to fail is important

### Generate an answer with the local model

```python
from transformers import pipeline

generator = pipeline(
    task="text-generation",
    model="Qwen/Qwen2.5-0.5B-Instruct",
)

result = generator(prompt, max_new_tokens=150, do_sample=False)
full_text = result[0]["generated_text"]
answer = full_text[len(prompt):].strip()

print(answer)
```

What to know:

- `do_sample=False` makes tutorial runs more repeatable
- the returned text often includes the original prompt
- small instruct models can still produce unsupported claims, so inspect the evidence

### Minimal end-to-end function

```python
def answer_question(question: str, top_k: int = 3) -> str:
    query_vector = embedder.encode(question)
    scores = np.array([cosine_similarity(query_vector, vec) for vec in chunk_vectors])
    top_ids = scores.argsort()[-top_k:][::-1]

    context = "\n\n".join(chunks[i] for i in top_ids)
    prompt = f"""Use only the context below to answer the question.
If the answer is missing, say you don't know.

Context:
{context}

Question: {question}
Answer:
"""

    output = generator(prompt, max_new_tokens=150, do_sample=False)[0]["generated_text"]
    return output[len(prompt):].strip()
```

That function is already a complete toy RAG system.

## 5. The Main Tradeoffs

### Chunk size

Smaller chunks:

- improve precision
- risk losing context

Larger chunks:

- preserve context
- risk retrieving broad, noisy passages

### Overlap

Less overlap:

- simpler
- cheaper
- more likely to cut important text in half

More overlap:

- preserves continuity
- increases duplicate information

### Top-k

Lower `k`:

- cleaner prompts
- risk missing evidence

Higher `k`:

- more evidence
- more noise
- longer prompts

### Prompt strictness

A strict prompt:

- reduces hallucination
- may answer "I don't know" more often

A loose prompt:

- sounds fluent
- is more likely to invent

## 6. What Good Retrieval Looks Like

Good retrieval means:

- the returned chunks clearly mention the answer
- the chunks are semantically related to the question
- the answer can be explained by pointing back to those chunks

Bad retrieval often looks like:

- chunks that share words but miss the real topic
- chunks that are too broad to answer the question
- chunks from the wrong section because chunking broke context

## 7. How To Evaluate Your Results

Before tuning the model, ask these questions:

- Did the right chunks get retrieved?
- Did the answer stay inside the retrieved evidence?
- Did the answer combine multiple chunks correctly?
- Did the prompt tell the model what to do when context is missing?
- Did chunk size or top-k make the prompt too noisy?

A simple manual evaluation loop:

1. Ask a question whose answer you can find in the PDF.
2. Print the top retrieved chunks.
3. Read them yourself.
4. Compare them against the model answer.
5. Change one variable at a time.

## 8. Common Beginner Mistakes

### Mistake: treating RAG like magic

RAG is not a single model trick. It is a pipeline, and each step can fail.

### Mistake: skipping inspection of retrieved chunks

If retrieval is wrong, generation is built on bad evidence.

### Mistake: chunks that are too large

Huge chunks often reduce retrieval quality.

### Mistake: chunks with no overlap

Important context can get split across chunk boundaries.

### Mistake: no fallback rule in the prompt

If you do not tell the model how to behave when context is missing, it may guess.

### Mistake: re-embedding the corpus on every question

Corpus embeddings should usually be computed once and reused.

### Mistake: tuning the LLM before fixing retrieval

In beginner RAG systems, retrieval quality usually matters more than fancy generation settings.

## 9. Practical Defaults For This Tutorial

If you just need a working baseline, start here:

- one source document: `world-history-text.pdf`
- simple text extraction
- light whitespace normalization
- character chunks with overlap
- `all-MiniLM-L6-v2` embeddings
- cosine similarity
- `top_k = 3`
- a strict prompt
- deterministic generation: `do_sample=False`

This is small enough to understand and good enough to learn from.

## 10. What Is Out Of Scope

These topics are useful, but not required for this tutorial baseline:

- vector databases
- hybrid keyword plus semantic retrieval
- rerankers
- agent frameworks
- fine-tuning
- multi-document pipelines
- production deployment
- latency optimization
- GPU serving stacks

If you can build the small baseline first, those become much easier later.

## 11. Minimal Install Hint

This repo now includes a [`pyproject.toml`](./pyproject.toml) and `uv.lock`.

The shortest setup path is:

```powershell
uv sync --extra notebook
uv run python .drive-setup\setup.py
```

The setup script downloads the PDF and model assets used by this repo.

## 12. References

- PyCon tutorial page: <https://us.pycon.org/2026/schedule/presentation/56/>
- Repo bootstrap script: [`.drive-setup/setup.py`](./.drive-setup/setup.py)
- pypdf text extraction docs: <https://pypdf.readthedocs.io/en/4.2.0/user/extract-text.html>
- Sentence Transformers quickstart: <https://sbert.net/docs/quickstart.html>
- Sentence Transformers semantic search docs: <https://www.sbert.net/examples/sentence_transformer/applications/semantic-search/README.html>
- Hugging Face pipelines docs: <https://huggingface.co/docs/transformers/main/pipeline_tutorial>
- Hugging Face Hub download docs: <https://huggingface.co/docs/huggingface_hub/en/guides/download>
