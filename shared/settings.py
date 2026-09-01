from pydantic_settings import BaseSettings

class Settings(BaseSettings):

    qdrant_mode: str = "server"
    qdrant_host: str = "127.0.0.1"
    qdrant_port: int = 6333

    embedder_mode: str = "server"
    embedder_url: str = "http://127.0.0.1:1010"

    # embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2" # INGLES
    embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2" # ESPAÑOL

    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    mlflow_experiment_prefix: str = "rag-system-eval"
    mlflow_tracking_uri: str = "http://127.0.0.1:5000/"
    
    ollama_host: str = "http://127.0.0.1:11434"

    corpus_dir: str = "shared/corpus"
    questions_path: str = "shared/eval/questions.jsonl"

    class Config:
        env_file = ".env"
        env_prefix = "RAG_"

settings = Settings()
