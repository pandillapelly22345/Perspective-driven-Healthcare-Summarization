from huggingface_hub import upload_folder

upload_folder(
    repo_id="harshvardhini123/fine_tuned_bart",
    folder_path="fine_tuned_bart",
    repo_type="model"
)

upload_folder(
    repo_id="harshvardhini123/fine_tuned_model",
    folder_path="fine_tuned_model",
    repo_type="model"
)
