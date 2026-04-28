import os
from huggingface_hub import HfApi

def main():
    print("=== Auto-Uploading to Hugging Face ===")
    token = input("Paste your Hugging Face Token: ").strip()
    if not token:
        print("Error: Token cannot be empty.")
        return

    try:
        # Authenticate using the token directly
        api = HfApi(token=token)
        
        # Automatically fetch your username using the token
        user_info = api.whoami()
        username = user_info["name"]
        
        # Automatically set the space name
        repo_id = f"{username}/Apex-LLM-Engine"
        
        print(f"\nDetected user: {username}")
        print(f"Targeting Space: {repo_id}")

        # Ensure the Space exists (creates it if it doesn't)
        try:
            api.repo_info(repo_id=repo_id, repo_type="space")
        except Exception:
            print("Creating the Space automatically...")
            api.create_repo(repo_id=repo_id, repo_type="space", space_sdk="docker", exist_ok=True)

        upload_dir = os.path.dirname(os.path.abspath(__file__))
        print("\nUploading files... (Please wait a few minutes for the 600MB weights file)")

        # Upload everything silently without asking for repo names
        api.upload_folder(
            folder_path=upload_dir,
            repo_id=repo_id,
            repo_type="space",
            ignore_patterns=["upload_to_hf.py", "app/*", "app", "__pycache__/*", "venv/*", ".git/*", "*.log"],
            commit_message="Auto-Deploying Apex LLM Engine"
        )
        
        print("\n✅ Upload Complete! Everything is online.")
        print(f"Your live app is building here: https://huggingface.co/spaces/{repo_id}")

    except Exception as e:
        print(f"\n❌ Error:\n{str(e)}")

if __name__ == "__main__":
    main()
