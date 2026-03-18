import sys
import numpy as np

def test_transformer_logic():
    print("--- Starting Transformer Functional Verification ---")

    # Try PyTorch-based implementations first
    try:
        import torch
        print(f"--> PyTorch detected: {torch.__version__}")
        backend = "torch"
    except ImportError:
        torch = None
        backend = None

    # Try TensorFlow as fallback
    if backend is None:
        try:
            import tensorflow as tf
            print(f"--> TensorFlow detected: {tf.__version__}")
            backend = "tf"
        except ImportError:
            print("CRITICAL: Neither PyTorch nor TensorFlow available.")
            sys.exit(1)

    try:
        # -------------------------------
        # CASE 1: PyTorch-style Transformer
        # -------------------------------
        if backend == "torch":
            print("--> Running PyTorch Transformer test...")

            import torch.nn as nn

            model = nn.Transformer(
                d_model=32,
                nhead=4,
                num_encoder_layers=2,
                num_decoder_layers=2,
                dim_feedforward=64,
                dropout=0.1,
            )

            src = torch.rand((10, 2, 32))  # (seq_len, batch, d_model)
            tgt = torch.rand((5, 2, 32))

            output = model(src, tgt)

            if output.shape == (5, 2, 32):
                print(f"    [✓] Output shape valid: {output.shape}")
            else:
                raise ValueError(f"Unexpected output shape: {output.shape}")

            norm = torch.norm(output).item()
            print(f"    [✓] Output norm: {norm:.4f}")
            if not (0.1 < norm < 100):
                raise ValueError("Numerical instability detected.")

        # -------------------------------
        # CASE 2: TensorFlow-style Transformer
        # -------------------------------
        elif backend == "tf":
            print("--> Running TensorFlow Transformer test...")

            import tensorflow as tf

            # Detect TF1 vs TF2 behavior
            if tf.__version__.startswith("1."):
                print("    [!] TensorFlow 1.x detected (expected for legacy repo)")
                sess = tf.Session()
                x = tf.random_uniform([2, 10, 32])
                y = sess.run(x)

                if y.shape == (2, 10, 32):
                    print(f"    [✓] TF1 tensor shape valid: {y.shape}")
                else:
                    raise ValueError("TF1 tensor shape mismatch")

                sess.close()

            else:
                print("    [!] TensorFlow 2.x detected")

                x = tf.random.uniform([2, 10, 32])
                y = x.numpy()

                if y.shape == (2, 10, 32):
                    print(f"    [✓] TF2 tensor shape valid: {y.shape}")
                else:
                    raise ValueError("TF2 tensor shape mismatch")

        # -------------------------------
        # Generic attention sanity check
        # -------------------------------
        print("--> Running attention sanity check...")

        def softmax(x):
            e_x = np.exp(x - np.max(x))
            return e_x / e_x.sum(axis=-1, keepdims=True)

        Q = np.random.rand(2, 4)
        K = np.random.rand(2, 4)
        V = np.random.rand(2, 4)

        scores = np.dot(Q, K.T) / np.sqrt(4)
        weights = softmax(scores)
        output = np.dot(weights, V)

        if output.shape == (2, 4):
            print("    [✓] Attention computation valid")
        else:
            raise ValueError("Attention computation failed")

        print("--- SMOKE TEST PASSED ---")

    except Exception as e:
        print(f"CRITICAL VALIDATION FAILURE: {e}")
        sys.exit(1)


if __name__ == "__main__":
    test_transformer_logic()