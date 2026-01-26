import sys
import torch

def test_transformer_dependencies():
    print("--- Starting Annotated Transformer Verification ---")
    try:
        # 1. Test Torchtext Legacy API
        # This is the 'Trap': Versions < 0.12 had 'data.Field'
        # Versions 0.12+ moved it to 'legacy' or deleted it entirely.
        print("--> Verifying torchtext data structures...")
        import torchtext
        
        try:
            # Check for legacy location (0.9.0 - 0.11.0)
            from torchtext.legacy import data as legacy_data
            field = legacy_data.Field(lower=True)
            print("    [✓] Found via torchtext.legacy")
        except ImportError:
            # Check for original location (< 0.9.0)
            try:
                from torchtext import data as original_data
                field = original_data.Field(lower=True)
                print("    [✓] Found via torchtext.data")
            except (ImportError, AttributeError):
                print("    [!] ERROR: torchtext.data.Field not found. API is broken.")
                raise ImportError("Incompatible torchtext version: Field API missing.")

        # 2. Test Spacy Integration
        print("--> Verifying Spacy NLP integration...")
        import spacy
        nlp = spacy.blank("en")
        doc = nlp("AURA validation test.")
        
        # 3. Test Tensor Ops (NumPy/Torch bridge)
        print("--> Verifying tensor operations...")
        x = torch.randn(1, 10, 512)
        attn = torch.matmul(x, x.transpose(-1, -2))
        
        print(f"--> Attention matrix shape: {attn.shape}")
        print("--- SMOKE TEST PASSED ---")

    except Exception as e:
        print(f"CRITICAL VALIDATION FAILURE: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    test_transformer_dependencies()