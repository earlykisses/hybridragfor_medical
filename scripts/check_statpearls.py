from huggingface_hub import list_repo_files

files = list_repo_files(
    repo_id="MedRAG/statpearls",
    repo_type="dataset"
)

for file in files[:100]:
    print(file)