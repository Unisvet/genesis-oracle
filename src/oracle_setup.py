import os

# Set Keras backend to JAX before importing Keras
os.environ["KERAS_BACKEND"] = "jax"

import keras
import jax

def main():
    """
    Initializes the Genesis Oracle environment with JAX backend.
    """
    print(f"Successfully initialized Keras with {keras.backend.backend()} backend.")
    print(f"JAX version: {jax.__version__}")

if __name__ == "__main__":
    main()
