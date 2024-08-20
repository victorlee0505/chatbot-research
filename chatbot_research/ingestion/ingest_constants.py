from chromadb.config import Settings

# Define the folder for storing database
PERSIST_DIRECTORY_AZURE = './storage_azure'
PERSIST_DIRECTORY_HF = './storage_hf'
PERSIST_DIRECTORY_PARENT_HF = PERSIST_DIRECTORY_HF + '/parent'

# Define the Chroma settings ONLINE Azure ingest
CHROMA_SETTINGS_AZURE = Settings(
        # chroma_db_impl='duckdb+parquet',
        persist_directory=PERSIST_DIRECTORY_AZURE,
        anonymized_telemetry=False
)

# Define the Chroma settings for OFFLINE HuggingFace ingest
CHROMA_SETTINGS_HF = Settings(
        # chroma_db_impl='duckdb+parquet',
        persist_directory=PERSIST_DIRECTORY_HF,
        anonymized_telemetry=False
)

CHROMA_SETTINGS_PARENT_HF = Settings(
        # chroma_db_impl='duckdb+parquet',
        persist_directory=PERSIST_DIRECTORY_PARENT_HF,
        anonymized_telemetry=False
)

ALL_MINILM_L6_V2 = 'sentence-transformers/all-MiniLM-L6-v2'