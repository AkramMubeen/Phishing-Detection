import os
from pathlib import Path

package_name = "Phishing"

list_of_files = [
    "github/workflows/.gitkeep",
    f"src/{package_name}/__init__.py",
    f"src/{package_name}/components/__init__.py",
    f"src/{package_name}/components/data_ingestion.py",
    f"src/{package_name}/components/data_transformation.py",
    f"src/{package_name}/components/model_trainer.py",
    f"src/{package_name}/components/model_evaluation.py"
    f"src/{package_name}/pipelines/__init__.py",
    f"src/{package_name}/pipelines/training_pipeline.py",
    f"src/{package_name}/pipelines/prediction_pipeline.py",
    f"src/{package_name}/logger.py",
    f"src/{package_name}/exception.py",
    f"src/{package_name}/utils/__init__.py",
    "notebooks/research.ipynb",
    "notebooks/data/.gitkeep",
    "requirements.txt",
    "setup.py",
    "init_setup.sh",
]

# Create directories and empty files
for file_path in list_of_files:
    file_path = Path(file_path)
    file_dir = file_path.parent

    file_dir.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist

    if not file_path.exists() or file_path.stat().st_size == 0:
        with open(file_path, "w") as f:
            pass
    else:
        print(f"File already exists: {file_path}")
