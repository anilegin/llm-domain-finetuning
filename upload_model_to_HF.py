from huggingface_hub import HfApi

api = HfApi()

# Create the repo (skip if it already exists)
api.create_repo(repo_id="anilegin/Qwen2.5-3B-Instruct-FT-RAG-LoRA", repo_type="model", exist_ok=True)

# Upload the whole folder
api.upload_folder(
    folder_path="models/qwen-rag-lora-k3-seq4096-lr1e4",
    repo_id="anilegin/Qwen2.5-3B-Instruct-FT-RAG-LoRA",
    repo_type="model",
)

# Create the repo (skip if it already exists)
api.create_repo(repo_id="anilegin/SmolLM-3B-Instruct-FT-RAG-LoRA", repo_type="model", exist_ok=True)

# Upload the whole folder
api.upload_folder(
    folder_path="models/smollm3-rag-lora-k3-seq4096-lr1e4",
    repo_id="anilegin/SmolLM-3B-Instruct-FT-RAG-LoRA",
    repo_type="model",
)

# Create the repo (skip if it already exists)
api.create_repo(repo_id="anilegin/Qwen2.5-3B-Instruct-FT-RAG-LoRA-corrupt", repo_type="model", exist_ok=True)

# Upload the whole folder
api.upload_folder(
    folder_path="models/qwen-rag-lora-k3-seq4096-lr1e4-corrupt",
    repo_id="anilegin/Qwen2.5-3B-Instruct-FT-RAG-LoRA-corrupt",
    repo_type="model",
)