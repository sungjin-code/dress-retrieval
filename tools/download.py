"""Download the models of the Street2Shop paper into models/ (SigLIP2, Tianmu-MERE, OWLv2, marqo). Token read from .env (HF_TOKEN)."""

from dotenv import load_dotenv
from huggingface_hub import snapshot_download

load_dotenv()  # HF_TOKEN

# ponytail: full repo; SigLIP2 vision tower is inside the single checkpoint, load only vision_model at eval time
for m in [
    "google/siglip2-base-patch16-224",
    "TianmuLab/Tianmu-MERE",
    "google/owlv2-base-patch16-ensemble",
    "Marqo/marqo-ecommerce-embeddings-B",
]:
    snapshot_download(m, local_dir=f"models/{m.split('/')[1]}")
