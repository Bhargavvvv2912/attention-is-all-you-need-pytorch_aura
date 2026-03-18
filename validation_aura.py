import sys

def test_tf1_transformer_compat():
    print("--- Starting TF1 Compatibility Test (Attention Is All You Need) ---")

    try:
        import tensorflow as tf
        print(f"--> Detected TensorFlow version: {tf.__version__}")
    except ImportError:
        print("CRITICAL: TensorFlow not installed.")
        sys.exit(1)

    # -------------------------------
    # Enforce TF1 requirement
    # -------------------------------
    if not tf.__version__.startswith("1."):
        print("CRITICAL: TensorFlow 2.x detected, but this repo requires TF 1.x.")
        sys.exit(1)

    try:
        print("--> Running TF1-style graph execution test...")

        # TF1-style graph construction
        x = tf.placeholder(tf.float32, shape=(None, 10))
        W = tf.Variable(tf.random_normal([10, 5]))
        b = tf.Variable(tf.zeros([5]))
        y = tf.matmul(x, W) + b

        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)
            import numpy as np
            dummy_input = np.random.rand(2, 10)
            output = sess.run(y, feed_dict={x: dummy_input})

        if output.shape == (2, 5):
            print(f"    [✓] TF1 graph execution valid: {output.shape}")
        else:
            raise ValueError(f"Unexpected output shape: {output.shape}")

        print("--> Checking for deprecated modules (tf.contrib)...")

        if not hasattr(tf, "contrib"):
            raise RuntimeError("tf.contrib missing — incompatible TF version")

        print("    [✓] tf.contrib available")

        print("--- TF1 COMPATIBILITY TEST PASSED ---")

    except Exception as e:
        print(f"CRITICAL VALIDATION FAILURE: {e}")
        sys.exit(1)


if __name__ == "__main__":
    test_tf1_transformer_compat()