what is RAG

1 - Retrieval - find info from a source that is relevant to a query
2 - Augmented - use the retrieved info to modify input to the LLM
3 - Generation - produce some output from that input (text in response to prompt)

why use RAG
Reduce hallucinations
work with custom data - private or domain specific

why build local RAG models

privacy
speed
cost




tqdm for pdf read progress
pymupdf/fitz for reading pdf

use splitting to find sentences first
then replace with spacy - english model - for sentencizing
get the token count - naively using len

build a metadata dict using sentence count in each page, page token count, page sentence count
use pandas dataframe to look at this metadata using describe columns.

chunking pages:
chunk
then, add addl metadata -- sentence chunks and num of chunks
