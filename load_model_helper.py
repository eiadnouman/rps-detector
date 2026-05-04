"""
load_model_helper.py
Loads rps_best.keras (saved with Keras 3 Sequential) by:
  1. Rebuilding the exact same architecture using Keras 3
  2. Loading weights directly from the .weights.h5 inside the zip
"""
import os, zipfile, tempfile
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


def _build_model():
    """Exact same architecture used during training."""
    import keras
    from keras.applications import EfficientNetB0
    from keras.layers import GlobalAveragePooling2D, BatchNormalization, Dense, Dropout
    from keras import Sequential

    base_model = EfficientNetB0(
        weights=None,         # weights loaded separately
        include_top=False,
        input_shape=(224, 224, 3)
    )
    base_model.trainable = False

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        BatchNormalization(),
        Dense(256, activation='relu'),
        Dropout(0.4),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(3, activation='softmax')
    ])
    return model


def load_rps_model(path='rps_best.keras'):
    """
    Build the model and load weights from a .keras zip file.
    Returns the fully-loaded model ready for inference.
    """
    model = _build_model()

    # Force-build the model so weights can be assigned
    import numpy as np
    model(np.zeros((1, 224, 224, 3), dtype='float32'), training=False)

    # Extract model.weights.h5 from the zip
    with zipfile.ZipFile(path) as zf:
        with tempfile.NamedTemporaryFile(suffix='.weights.h5', delete=False) as tmp:
            tmp.write(zf.read('model.weights.h5'))
            tmp_path = tmp.name

    try:
        model.load_weights(tmp_path)
        print(f"Model weights loaded from {path}")
    except Exception as e:
        print(f"load_weights failed ({e}), trying by_name...")
        model.load_weights(tmp_path, skip_mismatch=True)
    finally:
        os.unlink(tmp_path)

    return model


# Quick test when run directly
if __name__ == '__main__':
    import numpy as np
    m = load_rps_model('rps_best.keras')
    dummy = np.zeros((1, 224, 224, 3), dtype='float32')
    pred = m.predict(dummy, verbose=0)
    print("Prediction shape:", pred.shape)
    print("Prediction:", pred)
