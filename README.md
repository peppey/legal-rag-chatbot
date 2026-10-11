# Legal RAG Chatbot

A retrieval-augmented research assistant for a (fictional) German law firm. It answers questions about case files, contracts and internal firm guidelines, cites the document excerpts it used and says so when the files do not contain an answer.

The project is built on Google Cloud: documents are embedded and retrieved with Vertex AI RAG Engine, answers are generated with Gemini. The corpus, the evaluation questions and the assistant's prompts are in German.

## How it works

```
JSONL documents ──► slim embedding file ──► Cloud Storage ──► Vertex AI RAG corpus
                                                                     │
question ──► retrieve top 20 chunks ──► rerank (semantic ranker) ──► top 5 ──► Gemini ──► answer with [n] citations
```

- **Ingestion**: [prepare.py](src/legal_rag_chatbot/ingestion/prepare.py) keeps only the fields that should be embedded (`document_id`, `case_id`, `document_type`, `document_title`, `created_at`, `content`). [upload.py](src/legal_rag_chatbot/ingestion/upload.py) uploads the file to a Cloud Storage bucket.
- **Corpus**: [corpus.py](src/legal_rag_chatbot/rag/corpus.py) creates a RAG corpus (embedding model `text-embedding-005`, chunks of 512 tokens with an overlap of 100) and imports the file from Cloud Storage.
- **Retrieval**: [query.py](src/legal_rag_chatbot/rag/query.py) retrieves chunks, optionally reranks them, and maps each chunk back to its `document_id`.
- **Answering**: [chat.py](src/legal_rag_chatbot/llm/chat.py) offers two modes:
  - `rerank` (default): retrieve 20 chunks, rerank with `semantic-ranker-512@latest`, pass the best 5 to `gemini-2.5-flash` and require `[n]` citations.
  - `grounded`: Gemini's built-in RAG grounding tool with `top_k=5`.
- **Answer format**: a short file memo (short answer, facts, deadlines and amounts, legal questions, assessment, open points). The assistant is instructed not to invent facts and to abstain when the documents do not support an answer.

## The dataset

The corpus is synthetic. Every person, company and case is fictional.

- 177 documents: 25 cases (`CASE-001` to `CASE-025`) with 6 to 8 documents each, plus 5 internal firm documents (`KANZLEI-01` to `KANZLEI-05`).
- Practice areas include tenancy, employment, sales, corporate, data protection, copyright, insurance, family and inheritance law.
- Document types: client intake notes, contracts, lawyer letters, opposing letters, internal file memos, client e-mails, court filings, expert opinions and firm guidelines.
- Lengths range from about 100 to about 2,900 words, so short letters and long memos and filings are both covered.

| Path | Content |
|---|---|
| `data/processed/synthetic_legal_counsel_documents.jsonl` | The full documents with metadata (source of truth) |
| `data/chunks/synthetic_legal_counsel_documents_slim.jsonl` | Derived file that is uploaded and embedded |
| `data/evaluation/retrieval_eval.json` | Evaluation questions with graded relevance labels and reference answers |
| `data/evaluation/results/` | JSON reports of past evaluation runs |
| `data/raw/pdf/` | PDF renderings of the documents (generated, git-ignored) |
| `data/preprocessed/` | OCR results of the PDFs, one JSON file per document (generated) |

## Setup

Requirements: [uv](https://docs.astral.sh/uv/) and Python 3.10 or newer (CI and Docker use 3.12). A Google Cloud project with the Vertex AI and Cloud Storage APIs enabled.

```bash
uv sync
gcloud auth application-default login
```

The project, region, bucket and corpus name are constants in the code, not environment variables. If you use your own Google Cloud project, adapt them:

| Constant | File |
|---|---|
| `PROJECT_ID`, `LOCATION` | [corpus.py](src/legal_rag_chatbot/rag/corpus.py), [query.py](src/legal_rag_chatbot/rag/query.py), [chat.py](src/legal_rag_chatbot/llm/chat.py) |
| `BUCKET_NAME`, `DESTINATION_PATH` | [upload_dataset.py](src/legal_rag_chatbot/scripts/upload_dataset.py) |
| `GCS_PATH` | [corpus.py](src/legal_rag_chatbot/rag/corpus.py) |
| `CORPUS_NAME` | [query.py](src/legal_rag_chatbot/rag/query.py), [chat.py](src/legal_rag_chatbot/llm/chat.py) |

## Usage

Build the corpus (once, and again whenever the documents change):

```bash
# Write the slim embedding file and upload it to Cloud Storage
uv run python -m legal_rag_chatbot.scripts.upload_dataset

# Create a RAG corpus and import the file, then copy the printed corpus name into CORPUS_NAME
uv run python -m legal_rag_chatbot.rag.corpus
```

Ask questions:

```bash
# Show the retrieved chunks for a question
uv run python -m legal_rag_chatbot.rag.query

# Answer a question with cited sources (edit the question at the bottom of the file)
uv run python -m legal_rag_chatbot.llm.chat
```

Or from Python:

```python
from legal_rag_chatbot.llm.chat import answer_with_sources

answer = answer_with_sources("Bis wann muss Frau Keller der Kündigung widersprechen?")
print(answer.text)
for source in answer.cited_sources():
    print(source.number, source.document_id)
```

Export the documents as PDFs to `data/raw/pdf/` (safe to run repeatedly):

```bash
uv run python -m legal_rag_chatbot.scripts.export_pdfs
```

The PDFs are not part of the RAG pipeline yet, which reads the JSONL file.

Run OCR on the PDFs with Google Cloud Document AI (Enterprise Document OCR, German language hint) and write one JSON file per PDF to `data/preprocessed/`:

```bash
gcloud services enable documentai.googleapis.com   # once per project
uv run python -m legal_rag_chatbot.scripts.run_ocr
```

The first run creates an `OCR_PROCESSOR` named `legal-rag-ocr` in the `eu` region. Existing results are skipped, use `--force` to redo them and `--limit N` for a trial run. Each JSON file contains `document_id`, the full `text`, the text and confidence per page, and OCR metadata. Online processing accepts up to 15 pages per PDF. The OCR output is not yet fed into the RAG pipeline either, which still reads the JSONL file.

## Evaluation

[retrieval_eval.json](data/evaluation/retrieval_eval.json) holds 77 questions in nine categories (for example legal assessment, calculations, deadlines, multi-case questions and questions that the corpus cannot answer). Each question lists the relevant documents with a relevance grade from 1 to 3 and a reference answer.

**Retrieval** (Recall, Precision, Hit rate, MRR, nDCG at several cutoffs; chunks are mapped back to documents):

```bash
uv run python -m legal_rag_chatbot.scripts.evaluate_retrieval --top-k 20
uv run python -m legal_rag_chatbot.scripts.evaluate_retrieval --top-k 20 \
    --ranker service --ranker-model semantic-ranker-512@latest
```

**Answers** (`gemini-2.5-pro` as judge: correctness, hallucination rate, correct abstention on unanswerable questions):

```bash
uv run python -m legal_rag_chatbot.scripts.evaluate_answers --mode rerank
uv run python -m legal_rag_chatbot.scripts.evaluate_answers --mode grounded --limit 10
```

Both scripts accept `--limit N` for a quick run, `--workers` and `--output` for the report path (default: `data/evaluation/results/`). The relevance labels belong to specific document contents. If you change the documents substantially, review the labels and re-import the corpus before comparing results with older reports.

## Development

```bash
uv run pytest
uv run ruff check .
uv run ruff format .
```

CI ([ci.yml](.github/workflows/ci.yml)) runs lint, format check and tests on every push and pull request. On `main` it also builds the Docker image and pushes it to Artifact Registry. The [Dockerfile](Dockerfile) installs the package with `uv`; its default command is still the placeholder entry point `legal-rag-chatbot`.

## Notes

- This is a research aid and does not replace legal review. Answers should be verified against the cited sources.
- All data is synthetic and must not be mixed with real client data.
