import sys
import torch
import torchtext

def test_transformer_310():
    print("--- Starting Annotated Transformer Verification (Py3.10) ---")
    try:
        print(f"--> Torchtext Version: {torchtext.__version__}")
        
        # Test Data Structures (Post-Legacy migration)
        print("--> Verifying torchtext vocab structures...")
        from torchtext.vocab import vocab
        from collections import Counter, OrderedDict
        
        # Simple functional test of the 0.12+ API
        c = Counter(['hello', 'world'])
        sorted_by_freq_tuples = sorted(c.items(), key=lambda x: x[1], reverse=True)
        ordered_dict = OrderedDict(sorted_by_freq_tuples)
        v = vocab(ordered_dict)
        print(f"    [✓] Vocab initialized with {len(v)} tokens.")

        # Test Tensor Operations (The core math)
        print("--> Verifying attention matrix math...")
        q = torch.randn(1, 8, 10, 64)
        k = torch.randn(1, 8, 10, 64)
        attn = torch.matmul(q, k.transpose(-2, -1))
        print(f"    [✓] Tensor attention verified: {attn.shape}")
        
        print("--- SMOKE TEST PASSED ---")

    except Exception as e:
        print(f"CRITICAL VALIDATION FAILURE: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    test_transformer_310()