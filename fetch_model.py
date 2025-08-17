# fetch_model.py
import os, sys, requests

MODEL_URL = os.getenv("MODEL_URL")
MODEL_PATH = os.getenv("MODEL_PATH", "stress.pth")

if not MODEL_URL:
    print("MODEL_URL no está definido. Saliendo.")
    sys.exit(0)

if os.path.exists(MODEL_PATH):
    print(f"{MODEL_PATH} ya existe. Omitiendo descarga.")
    sys.exit(0)

print(f"Descargando modelo desde {MODEL_URL} -> {MODEL_PATH}")
with requests.get(MODEL_URL, stream=True, timeout=300) as r:
    r.raise_for_status()
    with open(MODEL_PATH, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

print("Descarga completada.")
