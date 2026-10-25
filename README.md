# Legal RAG Chatbot

A retrieval-augmented research assistant for a (fictional) German law firm. It answers questions about case files, contracts and internal firm guidelines, cites the document excerpts it used and says so when the files do not contain an answer.

The project is built on Google Cloud: documents are embedded and retrieved with Vertex AI RAG Engine, answers are generated with Gemini. The corpus, the evaluation questions and the assistant's prompts are in German.

## How it works

```
raw JSONL ──► PDFs ──► OCR (Document AI) ──► JSON per PDF ──► structure-aware chunks ──► Cloud Storage ──► Vertex AI RAG corpus
                                                                                                              │
question ──► retrieve top 20 chunks ──► rerank (semantic ranker) ──► top 5 ──► Gemini ──► answer with [n] citations
```

- **Ingestion**: [prepare.py](src/legal_rag_chatbot/ingestion/prepare.py) writes the chunk file, one JSONL line per chunk. The text comes from the OCR results in `data/preprocessed/`, the other fields (`document_id`, `case_id`, `document_type`, `document_title`, `created_at`) from the raw JSONL. [upload.py](src/legal_rag_chatbot/ingestion/upload.py) uploads the file to a Cloud Storage bucket.
- **Chunking**: [chunking.py](src/legal_rag_chatbot/ingestion/chunking.py) cuts the OCR text at the paragraph boundaries reported by Document AI and drops page footers. Every numbered point (`3. ...`) and contract clause (`§ 9 ...`) starts a new chunk, sub-points stay with their point, wrapped lines and paragraphs continued across a page break are joined. Chunks of at most 300 words are packed from these blocks, small ones are merged, longer ones are split at sentence boundaries. Each chunk carries the document metadata and a `section` such as `Punkt 3: Fristen` or `§ 9 Kündigung`, so it can be understood and attributed on its own.
- **Corpus**: [corpus.py](src/legal_rag_chatbot/rag/corpus.py) creates a RAG corpus (embedding model `text-embedding-005`) and imports the chunk file from Cloud Storage. The Vertex chunk size (1,024 tokens, no overlap) is only a ceiling above the size of one chunk record, the splitting itself happens in `chunking.py`.
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
- Lengths range from about 100 to about 3,000 words, so short letters and long memos and filings are both covered.

| Path | Content |
|---|---|
| `data/raw/synthetic_legal_counsel_documents.jsonl` | The original documents with metadata (source of truth, the only copy of this file) |
| `data/raw/pdf/` | PDF renderings of the documents (generated, git-ignored) |
| `data/raw/pdf_with_noise/{A,B,Geteilt}/` | The PDFs that are OCRed, one folder per access group (see Access control) |
| `data/preprocessed/` | OCR results of the PDFs, one JSON file per document with its `access_group` (generated) |
| `data/chunks/synthetic_legal_counsel_documents_chunks_{A,B,Geteilt}.jsonl` | Derived chunk files, one per group, that are uploaded and embedded |
| `data/evaluation/retrieval_eval.json` | Evaluation questions with graded relevance labels and reference answers |
| `data/evaluation/results/` | JSON reports of past evaluation runs |

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
| `BUCKET_NAME`, `DESTINATION_DIR` | [upload_dataset.py](src/legal_rag_chatbot/scripts/upload_dataset.py) |
| `GCS_URIS` | [corpus.py](src/legal_rag_chatbot/rag/corpus.py) |
| `CORPUS_NAME` | [query.py](src/legal_rag_chatbot/rag/query.py), [chat.py](src/legal_rag_chatbot/llm/chat.py) |

## Usage

Build the corpus (once, and again whenever the documents change). The steps depend on each other: PDFs first, then OCR, then the upload.

```bash
# 1. Render the documents as PDFs (data/raw/pdf)
uv run python -m legal_rag_chatbot.scripts.export_pdfs

# 2. OCR the PDFs in data/raw/pdf_with_noise/{A,B,Geteilt} with Document AI (data/preprocessed), see below for the one-time setup
uv run python -m legal_rag_chatbot.scripts.run_ocr

# 3. Chunk the OCR text and upload one chunk file per access group to Cloud Storage
uv run python -m legal_rag_chatbot.scripts.upload_dataset

# 4. Create a RAG corpus and import the three files, then copy the printed corpus name into CORPUS_NAME
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
from legal_rag_chatbot.access import SYSTEM_ACCESS
from legal_rag_chatbot.llm.chat import answer_with_sources

answer = answer_with_sources(
    "Bis wann muss Frau Keller der Kündigung widersprechen?", access=SYSTEM_ACCESS
)
print(answer.text)
for source in answer.cited_sources():
    print(source.number, source.document_id)
```

### Web frontend

The React frontend in `frontend/` (Vite, TypeScript) talks to the FastAPI backend in [api.py](src/legal_rag_chatbot/api.py). Vite proxies `/api` to `http://127.0.0.1:8000`. Requires Node.js 18 or newer. Start the backend and the frontend in two terminals:

```bash
# Terminal 1: backend on http://127.0.0.1:8000 (log in in the frontend; for curl without a token use AUTH_MODE=dev DEV_ROLE=admin)
uv run uvicorn legal_rag_chatbot.api:app --reload

# Terminal 2: frontend on http://localhost:5173
cd frontend
npm install   # first run only
npm run dev
```

`npm run build` creates a production build in `frontend/dist/`.

Documents can be added by drag and drop onto the page or with the "Dokumente hinzufügen" button (PDF, DOCX, TXT, up to 20 MB each, several at once). Uploads are chunked with the same logic as the corpus and embedded with `text-embedding-005` into an in-memory store ([session.py](src/legal_rag_chatbot/rag/session.py)). The chat searches the uploads that are listed in the sidebar in addition to the RAG corpus and marks their excerpts as uploaded documents. Uploads are never written to Cloud Storage or imported into the corpus, and they are lost when the backend restarts. Scanned PDFs without a text layer are sent to Document AI (same OCR processor as the corpus, region `eu`) in slices of 15 pages, up to 60 pages; this takes a few seconds per slice and costs per page. All other uploads stay inside the backend.

### Access control

Every document belongs to one group, given by the folder of its PDF in `data/raw/pdf_with_noise/`: `A`, `B` or `Geteilt`. All documents of a case must be in the same group. The OCR stores the group in `access_group`, `upload_dataset` writes one chunk file per group and `corpus` imports them as three RagFiles of the one corpus.

The roles in [access_policy.json](src/legal_rag_chatbot/access_policy.json) map to groups: `a` reads `A` and `Geteilt`, `b` reads `B` and `Geteilt`, `admin` reads everything. Retrieval passes only the RagFile ids of the allowed groups to Vertex ([query.py](src/legal_rag_chatbot/rag/query.py)); it never searches the whole corpus, so a user without a readable group gets no excerpts. The PDF endpoint answers 404 for documents of other groups, and uploads are visible only to the user who made them. Evaluations and scripts use `SYSTEM_ACCESS` (all groups).

The API expects the Identity Platform ID token as `Authorization: Bearer ...` ([auth.py](src/legal_rag_chatbot/auth.py)). It checks signature, expiry, issuer and audience against the Google signing keys and reads the role from the `role` claim. Requests without a valid token get 401, tokens without a known role 403. `AUTH_MODE=dev` skips the token and takes the role from `DEV_ROLE` or the `X-Dev-Role` header; use it for local work only. After re-importing the corpus, restart the backend (the RagFile ids are cached).

Users are created in Identity Platform (email/password, self-registration disabled). The role is a custom claim, set after the account exists:

```bash
# Admin REST API, TOKEN=$(gcloud auth print-access-token)
curl -X POST -H "Authorization: Bearer $TOKEN" -H "x-goog-user-project: legal-rag-chatbot" \
  -H "Content-Type: application/json" \
  -d '{"localId": "<uid>", "customAttributes": "{\"role\": \"a\"}"}' \
  https://identitytoolkit.googleapis.com/v1/projects/legal-rag-chatbot/accounts:update
```

The user has to sign in again to get the new claim. The frontend reads `VITE_FIREBASE_PROJECT_ID` and `VITE_FIREBASE_API_KEY` from [frontend/.env](frontend/.env) (a browser API key restricted to the Identity Toolkit API) and shows a login page unless `VITE_DEMO_MODE` is true.

### PDF export and OCR

`export_pdfs` renders every document of the raw JSONL as a PDF and can be run repeatedly. `run_ocr` sends the PDFs to Google Cloud Document AI (Enterprise Document OCR, German language hint) and writes one JSON file per PDF to `data/preprocessed/`. Enable the API once per project:

```bash
gcloud services enable documentai.googleapis.com
```

The first OCR run creates an `OCR_PROCESSOR` named `legal-rag-ocr` in the `eu` region. Existing results are skipped, use `--force` to redo them and `--limit N` for a trial run. Each JSON file contains `document_id`, the full `text`, `paragraph_starts` (offsets of the paragraph boundaries in `text`), the text and confidence per page, and OCR metadata. Online processing accepts up to 15 pages per PDF.

The RAG corpus is built from the OCR text, so the retrieval and answer evaluations score what the system reads after OCR. The raw JSONL only supplies the metadata. Rerun the upload and the corpus import whenever PDFs or OCR results change.

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

### Search routing

Not every message needs a search of the RAG corpus. The chat endpoints take an optional `search_mode` (default `auto`):

| Mode | Behaviour |
|---|---|
| `auto` | Greetings, thanks and questions about the assistant get a short answer without any search. Without uploads everything else searches the corpus. With uploads, one small model call decides between uploads, corpus or both (and rewrites follow-up questions in the same call). If that call fails, everything is searched. |
| `corpus` | Always search the corpus, ignore uploads. Used by the evaluations, so their results stay comparable. |
| `uploads` | Only the uploaded documents, falls back to the corpus if they yield nothing. |
| `none` | No search. |

The routing prompt (`route.md`) and the smalltalk prompt (`smalltalk.md`) are part of the versioned prompt set. The frontend does not send `search_mode` yet.

### Prompt versions

The prompts (system, answer format, citations, uploads, query rewriting and the judge) are Markdown files in [src/legal_rag_chatbot/prompts/](src/legal_rag_chatbot/prompts/), one directory per version (`v1`, `v2`, ...). To change a prompt, copy the latest version directory, edit the copy and select it for a run:

```bash
PROMPT_VERSION=v2 uv run python -m legal_rag_chatbot.scripts.evaluate_answers
PROMPT_VERSION=v2 uv run uvicorn legal_rag_chatbot.api:app
```

Without `PROMPT_VERSION` the default `v2` is used (`v1` has no `security.md`, so the chat code no longer runs with it). Never edit a released version, otherwise old reports can no longer be reproduced. Both evaluation reports store `prompt_version` and `prompt_hash` in their `config`, so a changed version directory shows up as a new hash.

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
