import sys
import torch

def test_transformer_dependencies():
    print("--- Starting Annotated Transformer Verification ---")
    try:
        print("--> Checking torchtext version...")
        import torchtext
        print(f"    [Version]: {torchtext.__version__}")

        # The 'Legacy API' Trap
        print("--> Verifying Field API availability...")
        try:
            # First try the 0.9.0 - 0.11.0 location
            from torchtext.legacy import data as legacy_data
            field = legacy_data.Field(lower=True)
            print("    [✓] Field found in torchtext.legacy")
        except ImportError:
            try:
                # Then try the < 0.9.0 location
                from torchtext import data as original_data
                field = original_data.Field(lower=True)
                print("    [✓] Field found in torchtext.data")
            except (ImportError, AttributeError):
                print("    [!] ERROR: Field API is physically missing from this version.")
                raise ImportError("Incompatible torchtext version.")

        # 3. Test Tensor Ops
        print("--> Verifying tensor operations...")
        x = torch.randn(1, 10, 512)
        attn = torch.matmul(x, x.transpose(-1, -2))
        print(f"    [✓] Attention matrix verified: {attn.shape}")
        
        print("--- SMOKE TEST PASSED ---")

    except Exception as e:
        print(f"CRITICAL VALIDATION FAILURE: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    test_transformer_dependencies()