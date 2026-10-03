import agentplatform
from agentplatform import types
from google.genai import types as genai_types


PROJECT_ID = "legal-rag-chatbot"
LOCATION = "europe-west3"

CORPUS_DISPLAY_NAME = "legal-counsel-corpus"
GCS_PATH = "gs://legal-rag-chatbot-documents/raw/synthetic_legal_counsel_documents.jsonl"


def create_corpus() -> str:
    """Create a RAG corpus and import documents from Google Cloud Storage."""
    client = agentplatform.Client(
        project=PROJECT_ID,
        location=LOCATION,
    )

    embedding_model_config = types.RagEmbeddingModelConfig(
        vertex_prediction_endpoint=(
            types.RagEmbeddingModelConfigVertexPredictionEndpoint(
                endpoint="publishers/google/models/text-embedding-005",
            )
        )
    )

    corpus = client.rag.create_corpus(
        rag_corpus=types.RagCorpus(
            display_name=CORPUS_DISPLAY_NAME,
            rag_vector_db_config=types.RagVectorDbConfig(
                rag_embedding_model_config=embedding_model_config,
            ),
        )
    )

    client.rag.import_files(
        name=corpus.name,
        import_config=types.ImportRagFilesConfig(
            gcs_source=genai_types.GcsSource(
                uris=[GCS_PATH],
            ),
            rag_file_transformation_config=(
                types.RagFileTransformationConfig(
                    rag_file_chunking_config=(
                        types.RagFileChunkingConfig(
                            chunk_size=512,
                            chunk_overlap=100,
                        )
                    ),
                )
            ),
        ),
    )

    print(f"Created corpus: {corpus.name}")
    return corpus.name


if __name__ == "__main__":
    create_corpus()