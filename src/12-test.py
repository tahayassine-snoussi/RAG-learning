from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
persistent_directory = BASE_DIR / "db" / "chroma_db"

print(persistent_directory)