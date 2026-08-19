from pydantic_settings import BaseSettings

class Settings(BaseSettings):

    qdrant_mode: str = "local"
    qdrant_host: str = "127.0.0.1"
    qdrant_port: int = 6333

    # embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2" # INGLES
    embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2" # ESPAÑOL

    mlflow_tracking_uri: str = "file:./mlflow/mlruns"
    mlflow_experiment_prefix: str = "rag-system-eval"

    corpus_dir: str = "shared/corpus"
    questions_path: str = "shared/eval/questions.jsonl"

    class Config:
        env_file = ".env"
        env_prefix = "RAG_"

settings = Settings()
