import sys
import numpy as np


def test_sbert_logic():
    print("--- Starting Sentence-Transformers Functional Verification ---")

    # Optional: only enforce this in repos that actually depend on SBERT
    try:
        print("--> Loading SentenceTransformer model...")
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("NOTE: sentence_transformers not installed; skipping SBERT drift test.")
        return  # treat as pass for repos that don't use SBERT

    try:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        print("    [✓] Model loaded.")

        print("--> Generating embeddings...")
        sentences = [
            "AURA is a modernization agent.",
            "ASE 2026 is a top-tier conference.",
        ]
        embeddings = model.encode(sentences)

        if embeddings.shape == (2, 384):
            print(f"    [✓] Embeddings valid. Shape: {embeddings.shape}")
        else:
            raise ValueError(f"Unexpected embedding shape: {embeddings.shape}")

        norm = np.linalg.norm(embeddings[0])
        print(f"    [✓] L2 Norm: {norm:.4f}")
        if not (0.9 < norm < 1.1):
            raise ValueError("Embedding normalization drift detected.")

        print("--- SMOKE TEST PASSED ---")

    except Exception as e:
        print(f"CRITICAL VALIDATION FAILURE: {e}")
        sys.exit(1)


if __name__ == "__main__":
    test_sbert_logic()
