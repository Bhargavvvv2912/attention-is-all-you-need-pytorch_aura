import sys
import numpy as np

def test_sbert_logic():
    print("--- Starting Sentence-Transformers Functional Verification ---")
    
    try:
        # 1. Model Initialization
        # We use a small model to keep the GitHub Runner fast.
        print("--> Loading SentenceTransformer model...")
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        print("    [✓] Model loaded.")

        # 2. Embedding Generation (The Numpy/Torch Handshake)
        # Upgrading numpy to 2.0+ often breaks the .cpu().numpy() calls 
        # inside older sentence-transformers versions.
        print("--> Generating embeddings...")
        sentences = ["AURA is a modernization agent.", "ASE 2026 is a top-tier conference."]
        embeddings = model.encode(sentences)
        
        if embeddings.shape == (2, 384):
            print(f"    [✓] Embeddings valid. Shape: {embeddings.shape}")
        else:
            raise ValueError(f"Unexpected embedding shape: {embeddings.shape}")

        # 3. Precision Check
        # Ensures that upgrading scipy/numpy hasn't introduced floating point drift.
        norm = np.linalg.norm(embeddings[0])
        print(f"    [✓] L2 Norm: {norm:.4f}")
        
        if not (0.9 < norm < 1.1):
            raise ValueError("Embedding normalization drift detected.")

        print("--- SMOKE TEST PASSED ---")

    except Exception as e:
        print(f"CRITICAL VALIDATION FAILURE: {str(e)}")
        # Likely failures: 
        # - UserWarning: Specified group 'I' is not known (Ruff/Linter conflict)
        # - ImportError: cannot import name 'triu' from 'scipy.linalg' (SciPy drift)
        sys.exit(1)

if __name__ == "__main__":
    test_sbert_logic()