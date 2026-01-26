import sys
import torch
import torchtext
from collections import Counter, OrderedDict

def test_transformer_310():
    print("--- Starting Annotated Transformer Verification (Py3.10) ---")
    try:
        print(f"--> Torchtext Version: {torchtext.__version__}")
        
        # Test Data Structures (0.13.0 API)
        print("--> Verifying torchtext vocab structures...")
        from torchtext.vocab import vocab
        
        # Simple functional test
        c = Counter(['attention', 'is', 'all', 'you', 'need'])
        ordered_dict = OrderedDict(sorted(c.items(), key=lambda x: x[1], reverse=True))
        v = vocab(ordered_dict)
        print(f"    [✓] Vocab initialized with {len(v)} tokens.")

        # Test Tensor Operations (The core math)
        print("--> Verifying attention matrix math...")
        q = torch.randn(1, 8, 10, 64)
        k = torch.randn(1, 8, 10, 64)
        attn = torch.matmul(q, k.transpose(-2, -1))
        
        # Check for NaN/Inf (Numerical stability)
        if not torch.isfinite(attn).all():
            raise ValueError("Attention weights are not finite!")
            
        print(f"    [✓] Tensor attention verified: {attn.shape}")
        print("--- SMOKE TEST PASSED ---")

    except Exception as e:
        print(f"CRITICAL VALIDATION FAILURE: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    test_transformer_310()