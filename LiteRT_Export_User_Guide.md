# Keras LiteRT Export - User Guide

**Version:** 2.0  
**Last Updated:** 2025  
**For:** Keras 3.x and Keras-Hub users

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Quick Start](#2-quick-start)
3. [Installation](#3-installation)
4. [Basic Usage](#4-basic-usage)
5. [Advanced Usage](#5-advanced-usage)
6. [Model-Specific Guides](#6-model-specific-guides)
7. [Optimization & Quantization](#7-optimization--quantization)
8. [Mobile Deployment](#8-mobile-deployment)
9. [Troubleshooting](#9-troubleshooting)
10. [FAQ](#10-faq)
11. [Developer Guide](#11-developer-guide-custom-models-and-advanced-usage)

---

## 1. Introduction

### What is LiteRT Export?

LiteRT (formerly TensorFlow Lite) export enables you to deploy Keras models on mobile, embedded, and edge devices. This guide covers the **one-line export API** in Keras 3.x that makes mobile deployment as simple as:

```python
model.export("model.tflite", format="litert")
```

### Key Benefits

- ✅ **One-line export** - No manual TFLite Converter configuration
- ✅ **Multi-backend support** - Train with JAX/PyTorch, export to LiteRT
- ✅ **Automatic dict handling** - Keras-Hub models work out-of-the-box
- ✅ **Dynamic shapes** - Resize inputs at runtime for flexibility
- ✅ **Built-in optimization** - Quantization support via `litert_kwargs`

### What's Supported

| Feature | Status |
|---------|--------|
| **Model Types** | Functional, Sequential, Subclassed |
| **Backends** | TensorFlow ✅, JAX ✅, PyTorch ✅ |
| **Input Types** | List, Dict, Nested structures |
| **Keras-Hub** | All task types (text, vision, multimodal) |
| **Quantization** | Dynamic range, Full integer, Float16 |

---

## 2. Quick Start

### 30-Second Example

```python
import keras
import keras_hub

# Example 1: Keras Core Sequential Model
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation='softmax')
])

# Export to TFLite (one line!)
model.export("mnist.tflite", format="litert")
print("✓ Exported mnist.tflite")

# Example 2: Keras-Hub Pretrained Model
# Sequence length is configured at model construction via the preprocessor
model = keras_hub.models.GemmaCausalLM.from_preset("gemma_2b_en")

# Export to LiteRT
model.export("gemma.tflite", format="litert")
print("✓ Exported gemma.tflite")
```

**Output:**
```
✓ Exported mnist.tflite (156 KB, ~5 seconds)
✓ Exported gemma.tflite (4.2 GB, ~45 seconds)
```

### What Happens Behind the Scenes

When you call `model.export("model.tflite", format="litert")`, the Keras Core 
export system performs these steps:

**1. Input Signature Inference**
```python
# Keras Core automatically infers the input signature:
from keras.src.export.export_utils import get_input_signature
input_signature = get_input_signature(model)
# Result: e.g., {"token_ids": TensorSpec(None, 128), ...}

# For standard Keras models:
# Uses model.inputs or infers from build configuration
```

**2. Nested Input Handling** (automatic)
```python
# If model uses nested inputs (dicts, lists):
# Keras Core's LiteRTExporter creates an adapter model automatically
# that converts flat list inputs → nested structure → original model
```

**3. TFLite Conversion**
```python
# Keras handles conversion via keras.src.export.litert
converter = tf.lite.TFLiteConverter.from_keras_model(model_to_export)
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS  # Fallback for complex ops
]
tflite_model = converter.convert()
```

**4. File Output**
```python
# Write flatbuffer to disk
with open("model.tflite", "wb") as f:
    f.write(tflite_model)
```

**Visual Flow:**

```
┌─────────────────┐
│  Your Model     │ (Keras Sequential, Functional, Subclassed, or Keras-Hub)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 1. Signature    │ Auto-detect input shapes/types
│    Inference    │ (Keras Core: get_input_signature)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 2. Nested Input │ Create adapter if model uses dict/nested inputs
│    Adaptation   │ (handled automatically by Keras Core)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 3. TFLite       │ Convert to .tflite format
│    Conversion   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 4. File Output  │ Save .tflite file
└─────────────────┘
```

**Performance Metrics:**

| Model Type | Export Time | File Size | Notes |
|------------|-------------|-----------|-------|
| Small Sequential (MNIST) | 3-8s | 50-200 KB | Fast, lightweight |
| ResNet50 (ImageNet) | 10-25s | 90-100 MB | Standard CNN |
| Gemma 2B (text gen) | 30-60s | 4-5 GB | Large transformer |
| BERT Base | 15-30s | 400-450 MB | Encoder transformer |
| MobileNetV3 | 5-15s | 10-20 MB | Mobile-optimized |

---

## 3. Installation

### Requirements

```bash
# Base requirements
pip install keras>=3.0.0 tensorflow>=2.16.0

# For Keras-Hub models
pip install keras-hub>=0.17.0
```

### Backend Configuration

LiteRT export **requires TensorFlow backend** for conversion, even if you train with JAX or PyTorch.

**Training with JAX/PyTorch, Exporting with TensorFlow:**

```python
# train.py - Train with JAX
import os
os.environ["KERAS_BACKEND"] = "jax"

import keras
import numpy as np

# Create model
model = keras.Sequential([...])

# Train with JAX
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
model.fit(X_train, y_train, epochs=10)

# Save weights (backend-agnostic)
model.save_weights("model_weights.weights.h5")
print("✓ Training complete with JAX backend")
```

```python
# export.py - Export with TensorFlow
import os
os.environ["KERAS_BACKEND"] = "tensorflow"  # Switch to TensorFlow

import keras

# Recreate model architecture (same code as training)
model = keras.Sequential([...])

# Load trained weights (works across backends!)
model.load_weights("model_weights.weights.h5")

# Export to LiteRT
model.export("model.tflite", format="litert")
print("✓ Export complete with TensorFlow backend")
```

**Why TensorFlow is Required:**

TFLite Converter is a TensorFlow-specific tool. The export process:
1. Converts Keras model → TensorFlow SavedModel format
2. TFLite Converter reads SavedModel → converts to .tflite
3. Output is a backend-agnostic .tflite file

**Verify Backend:**

```python
import keras
print(f"Current backend: {keras.backend.backend()}")

# Expected output for export:
# Current backend: tensorflow
```

**Docker Setup (Recommended for Multiple Backends):**

```dockerfile
# Dockerfile for export environment
FROM python:3.10-slim

# Install TensorFlow-only environment for export
RUN pip install keras>=3.0.0 tensorflow>=2.16.0 keras-hub>=0.17.0

# Set default backend
ENV KERAS_BACKEND=tensorflow

WORKDIR /app
CMD ["python", "export.py"]
```

```bash
# Build and run
docker build -t keras-export .
docker run -v $(pwd):/app keras-export
```

---

## 4. Basic Usage

### 4.1 Keras Core Models

#### Sequential Models

```python
import keras

# Create and train model
model = keras.Sequential([
    keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
    keras.layers.MaxPooling2D(),
    keras.layers.Flatten(),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy')
# ... training code ...

# Export to LiteRT
model.export("model.tflite", format="litert")
```

#### Functional Models

```python
import keras

inputs = keras.Input(shape=(224, 224, 3))
x = keras.layers.Conv2D(64, 3, activation='relu')(inputs)
x = keras.layers.GlobalAveragePooling2D()(x)
outputs = keras.layers.Dense(1000, activation='softmax')(x)

model = keras.Model(inputs=inputs, outputs=outputs)
model.export("classifier.tflite", format="litert")
```

#### Subclassed Models

```python
import keras
import numpy as np
import tensorflow as tf

class CustomModel(keras.Model):
    """Custom model with complex architecture."""
    
    def __init__(self):
        super().__init__()
        self.dense1 = keras.layers.Dense(128, activation='relu')
        self.dropout = keras.layers.Dropout(0.2)
        self.dense2 = keras.layers.Dense(64, activation='relu')
        self.output_layer = keras.layers.Dense(10, activation='softmax')
    
    def call(self, inputs, training=False):
        """Forward pass with training-aware dropout."""
        x = self.dense1(inputs)
        x = self.dropout(x, training=training)
        x = self.dense2(x)
        return self.output_layer(x)

# Create model
model = CustomModel()

# CRITICAL: Call the model with sample data first to establish input shapes
sample_input = np.zeros((1, 784), dtype=np.float32)
_ = model(sample_input, training=False)

# Export to LiteRT
model.export("custom.tflite", format="litert")
```

**Important for Subclassed Models:**

1. **Call Model with Sample Data First**
   ```python
   # Subclassed models need to be called once to establish input shapes
   sample_input = np.zeros((1, 784), dtype=np.float32)
   _ = model(sample_input, training=False)
   ```

2. **Handle Training Flag**
   ```python
   # If your model has training-dependent behavior (dropout, batch norm):
   class MyModel(keras.Model):
       def call(self, inputs, training=False):
           # Use training=False for inference
           x = self.dropout(x, training=training)
           return x
   
   # Export uses training=False automatically
   model.export("model.tflite", format="litert")
   ```

3. **Trace with Representative Data**
   ```python
   # For models with dynamic behavior, trace with real data
   # to ensure all code paths are captured
   
   sample_data = load_representative_samples()  # Real data, not zeros
   
   # Trace multiple times with different inputs
   for sample in sample_data[:10]:
       _ = model(sample, training=False)
   
   # Now export
   model.export("traced_model.tflite", format="litert")
   ```

4. **Avoid Python Control Flow**
   ```python
   # ❌ BAD: Python if/else won't be traced
   class BadModel(keras.Model):
       def call(self, inputs):
           if inputs.shape[1] > 100:  # Python condition
               return self.large_input_path(inputs)
           else:
               return self.small_input_path(inputs)
   
   # ✅ GOOD: Use TensorFlow ops
   class GoodModel(keras.Model):
       def call(self, inputs):
           condition = tf.shape(inputs)[1] > 100  # TF condition
           return tf.cond(
               condition,
               lambda: self.large_input_path(inputs),
               lambda: self.small_input_path(inputs)
           )
   ```

5. **Provide Explicit Input Signature** (for complex cases)
   ```python
   # If automatic signature inference fails:
   from keras.src.utils import InputSpec
   
   model.export(
       "model.tflite",
       format="litert",
       input_signature=[
           InputSpec(shape=(None, 784), dtype="float32")
       ]
   )
   ```

**Common Pitfalls:**

| Issue | Symptom | Solution |
|-------|---------|----------|
| Model not traced | Export fails or wrong shapes | Call model with sample data before export |
| Dynamic shapes not captured | Wrong output shapes | Use `InputSpec` with dynamic dimensions |
| Python control flow | Incorrect graph | Use TensorFlow ops (`tf.cond`, `tf.while_loop`) |
| Training flag | Different outputs | Ensure `training=False` during export |
| Custom layers not traceable | Conversion error | Implement `get_config()` and use TF ops only |

#### Multi-Input Models

```python
import keras

input1 = keras.Input(shape=(32,), name='input1')
input2 = keras.Input(shape=(64,), name='input2')

x1 = keras.layers.Dense(64)(input1)
x2 = keras.layers.Dense(64)(input2)

concatenated = keras.layers.Concatenate()([x1, x2])
outputs = keras.layers.Dense(10, activation='softmax')(concatenated)

model = keras.Model(inputs=[input1, input2], outputs=outputs)
model.export("multi_input.tflite", format="litert")
```

### 4.2 Keras-Hub Models

Keras-Hub models use the standard Keras `model.export()` API. The sequence length 
and image size are determined at model construction time (via the preprocessor), 
not during export.

#### Text Models (CausalLM)

```python
import keras_hub

# Load pretrained model with a specific sequence length
# The sequence_length is set in the preprocessor
preprocessor = keras_hub.models.GemmaCausalLMPreprocessor.from_preset(
    "gemma_2b_en",
    sequence_length=128  # Set desired sequence length here
)
model = keras_hub.models.GemmaCausalLM.from_preset(
    "gemma_2b_en",
    preprocessor=preprocessor
)

# Export to LiteRT (sequence length is already set)
model.export("gemma.tflite", format="litert")
```

**Alternative: Export with custom litert_kwargs:**

```python
import keras_hub
import tensorflow as tf

model = keras_hub.models.GemmaCausalLM.from_preset("gemma_2b_en")

# Export with quantization options via litert_kwargs
model.export(
    "gemma_quantized.tflite",
    format="litert",
    litert_kwargs={
        "optimizations": [tf.lite.Optimize.DEFAULT]
    }
)
```

**Supported Text Models:**
- `GemmaCausalLM` - Gemma 2B, 7B
- `GPTNeoXCausalLM` - GPT-NeoX models
- `LlamaCausalLM` - LLaMA models
- `BertTextClassifier` - BERT-based classifiers
- `DistilBertTextClassifier` - DistilBERT models

#### Vision Models (Image Classification)

```python
import keras_hub

# Load vision model - image size is determined by the preset/preprocessor
model = keras_hub.models.ImageClassifier.from_preset(
    "efficientnetv2_b0_imagenet"
)

# Export to LiteRT (image size is already set by the preset)
model.export("efficientnet.tflite", format="litert")
```

**Note:** Image size is determined at model construction time via the 
preprocessor/preset. To use a different image size, configure the 
ImageConverter during model creation.

**Supported Vision Models:**
- `ImageClassifier` - EfficientNet, ResNet, ViT, etc.
- `ObjectDetector` - YOLO, RetinaNet
- `ImageSegmenter` - DeepLab, U-Net, SAM
- `DepthEstimator` - DPT models

#### Multimodal Models

```python
import keras_hub

# PaliGemma (vision + language)
# Sequence length is set at model construction via preprocessor
model = keras_hub.models.PaliGemmaCausalLM.from_preset("paligemma_3b_224")
model.export("paligemma.tflite", format="litert")

# Gemma3 with vision encoder
model = keras_hub.models.Gemma3CausalLM.from_preset("gemma3_2b_en")
model.export("gemma3.tflite", format="litert")
```

#### Seq2Seq Models

```python
import keras_hub

# T5 or BART models - sequence length determined by preset
model = keras_hub.models.Seq2SeqLM.from_preset("t5_small")
model.export("t5.tflite", format="litert")
```

---

## 5. Advanced Usage

### 5.1 Dynamic Shape Export

Export models with flexible input dimensions that can be resized at runtime:

```python
import keras_hub

# Export model - shapes are determined by the model's build configuration
model = keras_hub.models.GemmaCausalLM.from_preset("gemma_2b_en")
model.export("gemma.tflite", format="litert")

# In your mobile app (Python example):
# Use ai_edge_litert.interpreter (preferred) or tf.lite.Interpreter (fallback)
try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    from tensorflow.lite import Interpreter

interpreter = Interpreter(model_path="gemma.tflite")
input_details = interpreter.get_input_details()

# Resize inputs if the model was exported with dynamic shapes
interpreter.resize_tensor_input(input_details[0]['index'], [1, 256])
interpreter.resize_tensor_input(input_details[1]['index'], [1, 256])

interpreter.allocate_tensors()
# Now run inference
```

**When to Use Dynamic Shapes:**
- Variable-length text inputs
- Multiple image sizes
- Batch size variations
- Memory-constrained devices

**Trade-offs:**
- ✅ More flexible at runtime
- ❌ Slightly slower first inference (shape propagation)
- ❌ May not work with some hardware accelerators

### 5.2 Custom Input Signatures

Override automatic signature inference for complex models:

```python
import keras
import tensorflow as tf

# Define custom signature using TensorSpec
input_signature = {
    "token_ids": tf.TensorSpec(dtype=tf.int32, shape=(None, 512)),
    "padding_mask": tf.TensorSpec(dtype=tf.int32, shape=(None, 512)),
}

# Export with custom signature via litert_kwargs
model.export(
    "custom.tflite",
    format="litert",
    input_signature=input_signature
)
```

### 5.3 Backend-Specific Export

#### Training with JAX, Exporting to LiteRT

```python
import os

# Train with JAX
os.environ["KERAS_BACKEND"] = "jax"
import keras

model = keras.Sequential([...])
model.compile(...)
model.fit(x_train, y_train)

# Save trained model
model.save("trained_model.keras")

# Switch to TensorFlow for export
os.environ["KERAS_BACKEND"] = "tensorflow"
import keras as keras_tf

# Load and export
model = keras_tf.models.load_model("trained_model.keras")
model.export("model.tflite", format="litert")
```

#### Training with PyTorch, Exporting to LiteRT

```python
import os

# Train with PyTorch
os.environ["KERAS_BACKEND"] = "torch"
import keras

model = keras.Sequential([...])
model.compile(...)
model.fit(x_train, y_train)

# Save trained model
model.save("trained_model.keras")

# Switch to TensorFlow for export
os.environ["KERAS_BACKEND"] = "tensorflow"
import keras as keras_tf

# Load and export
model = keras_tf.models.load_model("trained_model.keras")
model.export("model.tflite", format="litert")
```

---

## 6. Model-Specific Guides

### 6.1 Text Generation Models (GPT, Gemma, LLaMA)

```python
import keras_hub

# Load model - sequence length is configured via the preprocessor
# Default sequence length comes from the preset
model = keras_hub.models.GemmaCausalLM.from_preset("gemma_2b_en")

# Or specify a custom sequence length at construction time:
# preprocessor = keras_hub.models.GemmaCausalLMPreprocessor.from_preset(
#     "gemma_2b_en", sequence_length=128
# )
# model = keras_hub.models.GemmaCausalLM.from_preset(
#     "gemma_2b_en", preprocessor=preprocessor
# )

# Export for mobile inference
model.export("gemma_mobile.tflite", format="litert")

# Test inference - use ai_edge_litert (preferred) or tf.lite (fallback)
try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    from tensorflow.lite import Interpreter

interpreter = Interpreter(model_path="gemma_mobile.tflite")
interpreter.allocate_tensors()

# Get input details
input_details = interpreter.get_input_details()
print("Inputs:", [(d['name'], d['shape']) for d in input_details])

# Prepare inputs (list order, not dict!)
import numpy as np
seq_len = input_details[0]['shape'][1]  # Get sequence length from model
token_ids = np.array([[1, 2, 3] + [0] * (seq_len - 3)], dtype=np.int32)
padding_mask = np.array([[1, 1, 1] + [0] * (seq_len - 3)], dtype=np.int32)

# Set inputs (order matches signature - check input_details for names)
interpreter.set_tensor(input_details[0]['index'], token_ids)
interpreter.set_tensor(input_details[1]['index'], padding_mask)

# Run inference
interpreter.invoke()

# Get output
output_details = interpreter.get_output_details()
logits = interpreter.get_tensor(output_details[0]['index'])
```

**Important Notes:**
- Exported .tflite uses **list inputs**, not dict
- Sequence length is determined at model construction time
- Check `input_details` for the correct input order and shapes

### 6.2 Image Classification Models

```python
import keras_hub

# Load and export - image size is determined by the preset
model = keras_hub.models.ImageClassifier.from_preset(
    "efficientnetv2_b0_imagenet"
)
model.export("efficientnet.tflite", format="litert")

# Mobile inference
import tensorflow as tf
import numpy as np
from PIL import Image

# Use ai_edge_litert (preferred) or tf.lite (fallback)
try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    from tensorflow.lite import Interpreter

interpreter = Interpreter(model_path="efficientnet.tflite")
interpreter.allocate_tensors()

# Load and preprocess image
img = Image.open("cat.jpg").resize((224, 224))
img_array = np.array(img, dtype=np.float32) / 255.0
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

# Run inference
input_details = interpreter.get_input_details()
interpreter.set_tensor(input_details[0]['index'], img_array)
interpreter.invoke()

# Get predictions
output_details = interpreter.get_output_details()
predictions = interpreter.get_tensor(output_details[0]['index'])
```

### 6.3 Object Detection Models

```python
import keras_hub

# Load detector - image size is determined by the preset
model = keras_hub.models.ObjectDetector.from_preset("retinanet_resnet50")

# Export to LiteRT
model.export("detector.tflite", format="litert")
```

### 6.4 Multimodal Models (Vision + Language)

```python
import keras_hub

# PaliGemma for vision-language tasks
# Image size (224) and sequence length are determined by the preset
model = keras_hub.models.PaliGemmaCausalLM.from_preset("paligemma_3b_224")

# Export
model.export("paligemma.tflite", format="litert")

# Mobile inference requires both image and text inputs
# Check input_details for the correct input order and shapes
```

---

## 7. Optimization & Quantization

Quantization reduces model size and improves inference speed by using lower-precision data types (int8, float16) instead of float32.

### 7.1 Dynamic Range Quantization

**What it does:** Converts weights from float32 to int8, keeps activations as float32.

**Best for:** Quick optimization with minimal setup, good balance of size/accuracy.

```python
import tensorflow as tf
import keras

model = keras.Sequential([...])

# Export with dynamic range quantization using litert_kwargs
model.export(
    "model_quantized.tflite",
    format="litert",
    litert_kwargs={
        "optimizations": [tf.lite.Optimize.DEFAULT]
    }
)
```

**Results:**

| Metric | Before | After | Reduction |
|--------|--------|-------|-----------|
| File Size | 100 MB | 25 MB | **75%** |
| Inference Time | 100ms | 35-50ms | **50-65%** |
| Accuracy | 95.2% | 94.8% | -0.4% |
| RAM Usage | 120 MB | 40 MB | **67%** |

**When to Use:**
- ✅ Quick optimization without extra work
- ✅ Acceptable ~0.5-1% accuracy drop
- ✅ No dataset available for calibration
- ✅ Targeting CPU inference

**Limitations:**
- Activations still float32 (less speedup on some hardware)
- Hardware accelerators may not fully utilize

### 7.2 Full Integer Quantization (INT8)

**What it does:** Converts weights AND activations to int8 using representative dataset for calibration.

**Best for:** Maximum performance on mobile/edge devices with int8 acceleration.

```python
import tensorflow as tf
import keras
import numpy as np

model = keras.Sequential([...])

# Step 1: Prepare representative dataset (100-500 samples typical)
def representative_dataset():
    """Generator that yields calibration data."""
    for i in range(100):
        # Real data from validation set (NOT random!)
        sample = np.random.random((1, 224, 224, 3)).astype(np.float32)
        yield [sample]

# Step 2: For advanced quantization, use TFLite converter directly
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Enable optimizations
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Provide representative dataset
converter.representative_dataset = representative_dataset

# Enforce INT8 for weights and activations
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

# Optional: Integer-only inference (no float32 ops)
converter.inference_input_type = tf.int8  # Input is int8
converter.inference_output_type = tf.int8  # Output is int8

# Convert
tflite_model = converter.convert()

# Save
with open("model_int8.tflite", "wb") as f:
    f.write(tflite_model)
```

**Alternative: Use Keras Export API with litert_kwargs**

For simpler quantization, you can pass options via `litert_kwargs`:

```python
import keras
import tensorflow as tf

model = keras.Sequential([...])

# Prepare calibration data
def representative_dataset():
    for _ in range(100):
        yield [np.random.random((1, 224, 224, 3)).astype(np.float32)]

# Export with quantization via litert_kwargs
model.export(
    "model_quantized.tflite",
    format="litert",
    litert_kwargs={
        "optimizations": [tf.lite.Optimize.DEFAULT],
        "representative_dataset": representative_dataset
    }
)
```

**Results:**

| Metric | Float32 | Dynamic Range | Full INT8 |
|--------|---------|---------------|-----------|
| File Size | 100 MB | 25 MB | 25 MB |
| Inference (CPU) | 100ms | 40ms | 30ms |
| Inference (NPU) | 80ms | 60ms | **15ms** |
| Accuracy | 95.2% | 94.8% | 94.5% |
| Power Consumption | 100% | 70% | **40%** |

**When to Use:**
- ✅ Maximum performance needed
- ✅ Hardware with int8 acceleration (Edge TPU, NPU, DSP)
- ✅ Can tolerate ~1% accuracy drop
- ✅ Have representative dataset available

**Important Notes:**

1. **Use Real Data for Calibration**
   ```python
   # ❌ BAD: Random data doesn't represent distribution
   def bad_dataset():
       for _ in range(100):
           yield [np.random.random((1, 224, 224, 3))]
   
   # ✅ GOOD: Use validation set samples
   def good_dataset():
       validation_data = load_validation_set()
       for sample in validation_data[:200]:
           yield [sample]
   ```

2. **Calibration Dataset Size**
   - **Too small (<50):** Poor quantization, accuracy loss
   - **Sweet spot (100-500):** Good balance
   - **Too large (>1000):** Diminishing returns, slow export

3. **Check Accuracy After Quantization**
   ```python
   import tensorflow as tf
   
   # Export quantized model
   model.export(
       "quantized.tflite", 
       format="litert",
       litert_kwargs={
           "optimizations": [tf.lite.Optimize.DEFAULT],
           "representative_dataset": representative_dataset
       }
   )
   
   # Evaluate on validation set
   try:
       from ai_edge_litert.interpreter import Interpreter
   except ImportError:
       from tensorflow.lite import Interpreter
   
   interpreter = Interpreter("quantized.tflite")
   interpreter.allocate_tensors()
   
   accuracy = evaluate_tflite_model(interpreter, validation_data)
   print(f"Quantized model accuracy: {accuracy:.2%}")
   
   # If accuracy drop > 2%, consider:
   # - More calibration samples
   # - Per-channel quantization
   # - Hybrid quantization (keep sensitive layers as float32)
   ```

### 7.3 Float16 Quantization

**What it does:** Converts weights to float16, keeps high precision with good size reduction.

**Best for:** GPU inference, balance between size and accuracy.

```python
import tensorflow as tf
import keras

model = keras.Sequential([...])

# For float16 quantization, use TFLite converter directly
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]

tflite_model = converter.convert()

with open("model_float16.tflite", "wb") as f:
    f.write(tflite_model)
```

**Results:**

| Metric | Float32 | Float16 |
|--------|---------|---------|
| File Size | 100 MB | 50 MB |
| Inference (GPU) | 50ms | 25ms |
| Accuracy | 95.2% | 95.1% |

**When to Use:**
- ✅ GPU inference (mobile GPUs support float16)
- ✅ Minimal accuracy loss required (<0.1%)
- ✅ 50% size reduction sufficient

### 7.4 Hybrid Quantization

**What it does:** INT8 for most layers, float32 for sensitive layers (attention, softmax).

```python
import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Allow both INT8 and float32 ops
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
    tf.lite.OpsSet.TFLITE_BUILTINS
]

converter.representative_dataset = representative_dataset

tflite_model = converter.convert()
```

**When to Use:**
- ✅ Full INT8 causes >2% accuracy drop
- ✅ Have sensitive layers (transformers, attention)
- ✅ Need balance between performance and accuracy

### 7.5 Quantization Comparison

```python
# Benchmark script
import time
import numpy as np

def benchmark_model(model_path, test_data, num_runs=100):
    """Benchmark TFLite model."""
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        from tensorflow.lite import Interpreter
    
    interpreter = Interpreter(model_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    times = []
    for _ in range(num_runs):
        start = time.time()
        interpreter.set_tensor(input_details[0]['index'], test_data)
        interpreter.invoke()
        _ = interpreter.get_tensor(output_details[0]['index'])
        times.append(time.time() - start)
    
    return {
        'mean_ms': np.mean(times) * 1000,
        'std_ms': np.std(times) * 1000,
        'min_ms': np.min(times) * 1000,
        'max_ms': np.max(times) * 1000
    }

# Test all quantization modes
models = {
    'Float32': 'model.tflite',
    'Dynamic Range': 'model_dynamic.tflite',
    'INT8': 'model_int8.tflite',
    'Float16': 'model_float16.tflite'
}

test_data = np.random.random((1, 224, 224, 3)).astype(np.float32)

print("Quantization Benchmark Results:")
print("-" * 60)
for name, path in models.items():
    results = benchmark_model(path, test_data)
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"{name:15} | Size: {size_mb:6.2f} MB | "
          f"Time: {results['mean_ms']:6.2f}ms ± {results['std_ms']:.2f}ms")
```

```python
import tensorflow as tf
import numpy as np

# Create representative dataset generator
def representative_dataset():
    for i in range(100):
        # Yield sample data matching your model's input
        sample = np.random.random((1, 224, 224, 3)).astype(np.float32)
        yield [sample]

# Export with full integer quantization using litert_kwargs
model.export(
    "model_int8.tflite",
    format="litert",
    litert_kwargs={
        "optimizations": [tf.lite.Optimize.DEFAULT],
        "representative_dataset": representative_dataset
    }
)
```

**Benefits:**
- Maximum speed on integer-only hardware
- Up to 4x faster than float32
- Enables hardware acceleration (Edge TPU, DSP)
- Requires calibration dataset

### 7.6 Float16 Quantization (Simple)

Balance between size and accuracy using litert_kwargs:

```python
import tensorflow as tf

model.export(
    "model_quantized.tflite",
    format="litert",
    litert_kwargs={
        "optimizations": [tf.lite.Optimize.DEFAULT],
        "target_spec": {
            "supported_types": [tf.float16]
        }
    }
)
```

**Benefits:**
- Smaller than float32
- Better accuracy than int8
- GPU-friendly

### 7.7 Quantization for Text Models

```python
import tensorflow as tf
import keras_hub
import numpy as np

model = keras_hub.models.GemmaCausalLM.from_preset("gemma_2b_en")

# Representative dataset for text models
def text_representative_dataset():
    # Sample prompts
    prompts = [
        "Hello, how are you?",
        "What is machine learning?",
        "Tell me a story.",
        # Add more diverse prompts
    ]
    
    for prompt in prompts:
        # Tokenize (you'll need the preprocessor)
        preprocessor = model.preprocessor
        tokens = preprocessor.tokenizer(prompt)
        token_ids = tokens['token_ids'][:128]  # Match sequence length
        padding_mask = tokens['padding_mask'][:128]
        
        # Pad to fixed length
        token_ids = np.pad(token_ids, (0, 128 - len(token_ids)))
        padding_mask = np.pad(padding_mask, (0, 128 - len(padding_mask)))
        
        yield [
            np.array([token_ids], dtype=np.int32),
            np.array([padding_mask], dtype=np.int32)
        ]

# Export with quantization using litert_kwargs
model.export(
    "gemma_quantized.tflite",
    format="litert",
    litert_kwargs={
        "optimizations": [tf.lite.Optimize.DEFAULT],
        "representative_dataset": text_representative_dataset
    }
)
```

### 7.8 Quantization Comparison

| Method | Size Reduction | Speed Increase | Accuracy | Dataset Required |
|--------|---------------|----------------|----------|------------------|
| None | 0% | 1x | 100% | No |
| Dynamic Range | 75% | 2-3x | 99% | No |
| Float16 | 50% | 1.5-2x | 99.5% | No |
| Full Integer | 75% | 3-4x | 97-99% | Yes |


---

## 9. Troubleshooting

### 9.1 Common Errors

#### "The LiteRT export requires the filepath to end with '.tflite'"

**Solution:**
```python
# Wrong
model.export("model", format="litert")

# Correct
model.export("model.tflite", format="litert")
```

#### "Backend must be TensorFlow for LiteRT export"

**Solution:**
```python
import os
os.environ["KERAS_BACKEND"] = "tensorflow"

# Then import keras
import keras
```

#### "Unable to infer input signature"

**Solution:**
```python
# For Subclassed models, call with sample data first
import numpy as np
sample_input = np.zeros((1, 28, 28, 1))
_ = model(sample_input, training=False)

# Then export
model.export("model.tflite", format="litert")
```

#### "Out of memory during conversion"

Large models (e.g., gemma_7b) may require significant RAM during conversion.

**Solutions:**
1. Use a machine with more RAM (cloud instance)
2. Export with quantization to reduce memory:
```python
model.export(
    "model.tflite",
    format="litert",
    litert_kwargs={
        "optimizations": [tf.lite.Optimize.DEFAULT]
    }
)
```

#### "Some ops are not supported by TFLite"

**Solution:**
Enable TensorFlow Lite's flexible ops using litert_kwargs:
```python
import tensorflow as tf

model.export(
    "model.tflite",
    format="litert",
    litert_kwargs={
        "target_spec": {
            "supported_ops": [
                tf.lite.OpsSet.TFLITE_BUILTINS,
                tf.lite.OpsSet.SELECT_TF_OPS
            ]
        }
    }
)
```

### 9.2 Validation After Export

Always verify the exported model before deploying:

```python
import numpy as np

# Use ai_edge_litert (preferred) or tf.lite (fallback)
try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    from tensorflow.lite import Interpreter

# 1. Load the TFLite model
interpreter = Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# 2. Get input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("=== Model Validation ===")
print(f"Inputs: {len(input_details)}")
for i, detail in enumerate(input_details):
    print(f"  Input {i}: shape={detail['shape']}, dtype={detail['dtype']}")

print(f"Outputs: {len(output_details)}")
for i, detail in enumerate(output_details):
    print(f"  Output {i}: shape={detail['shape']}, dtype={detail['dtype']}")

# 3. Test with sample data
sample_input = np.random.random(input_details[0]['shape']).astype(np.float32)

interpreter.set_tensor(input_details[0]['index'], sample_input)
interpreter.invoke()
output = interpreter.get_tensor(output_details[0]['index'])

print(f"\n✓ Model runs successfully!")
print(f"Output shape: {output.shape}")
print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")

# 4. Compare with original Keras model (sanity check)
keras_output = model.predict(sample_input)
tflite_output = output

# Calculate difference
diff = np.abs(keras_output - tflite_output).mean()
print(f"\nKeras vs TFLite difference: {diff:.6f}")

if diff < 1e-4:
    print("✓ Outputs match (excellent)")
elif diff < 1e-2:
    print("⚠ Small difference (acceptable for quantized models)")
else:
    print("❌ Large difference (investigate!)")
```

### 9.3 Debugging Tips

**1. Enable Verbose Logging**

```python
import logging
logging.getLogger("tensorflow").setLevel(logging.DEBUG)

model.export("model.tflite", format="litert")
# Will print detailed conversion logs
```

**2. Inspect Model Graph**

```python
# Visualize what TFLite converter sees
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Get concrete functions
concrete_func = model.save_spec()
print("Model signature:")
print(concrete_func.structured_input_signature)
print(concrete_func.structured_outputs)
```

**3. Test Conversion Strategies**

```python
# If default export fails, try explicit strategies

# Strategy 1: Direct (faster)
from keras.src.export.litert import export_litert
try:
    export_litert(model, "model_direct.tflite", use_wrapper=False)
    print("✓ Direct conversion succeeded")
except Exception as e:
    print(f"✗ Direct conversion failed: {e}")

# Strategy 2: Wrapper-based (more compatible)
try:
    export_litert(model, "model_wrapper.tflite", use_wrapper=True)
    print("✓ Wrapper-based conversion succeeded")
except Exception as e:
    print(f"✗ Wrapper-based conversion failed: {e}")
```

**4. Check Model Size**

```python
import os

def analyze_model_size(filepath):
    """Check if model size is reasonable."""
    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    
    if size_mb < 0.1:
        print(f"⚠ Warning: Model is very small ({size_mb:.2f} MB)")
        print("  This might indicate export issues (weights not saved)")
    elif size_mb > 5000:
        print(f"⚠ Warning: Model is very large ({size_mb:.2f} MB)")
        print("  Consider quantization to reduce size")
    else:
        print(f"✓ Model size: {size_mb:.2f} MB (reasonable)")
    
    return size_mb

analyze_model_size("model.tflite")
```

**5. Verify Weights Were Saved**

```python
def check_weights_saved(model_path):
    """Verify model contains weights (not just structure)."""
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        from tensorflow.lite import Interpreter
    
    interpreter = Interpreter(model_path)
    
    # Count non-zero weights
    total_weights = 0
    non_zero_weights = 0
    
    for detail in interpreter.get_tensor_details():
        if detail['name'].endswith('weights') or 'kernel' in detail['name']:
            tensor = interpreter.get_tensor(detail['index'])
            total_weights += tensor.size
            non_zero_weights += np.count_nonzero(tensor)
    
    if total_weights == 0:
        print("❌ No weights found in model!")
        return False
    
    sparsity = 1 - (non_zero_weights / total_weights)
    print(f"✓ Found {total_weights} weight values")
    print(f"  Sparsity: {sparsity*100:.2f}%")
    
    if non_zero_weights < total_weights * 0.01:
        print("⚠ Warning: Most weights are zero (model might not be trained)")
    
    return True

check_weights_saved("model.tflite")
```

**6. Profile Conversion Time**

```python
import time

start = time.time()
model.export("model.tflite", format="litert")
duration = time.time() - start

print(f"Export took {duration:.2f} seconds")

# Typical times:
# - Small models (<100MB): 5-30s
# - Medium models (100-500MB): 30-120s
# - Large models (>500MB): 2-10min

if duration > 300:  # 5 minutes
    print("⚠ Conversion is very slow. Consider:")
    print("  1. Using quantization")
    print("  2. Running on more powerful machine")
    print("  3. Checking for inefficient custom layers")
```

**7. Debug Dict Input Adapter**

```python
# If dict input adapter is causing issues
from keras.src.export.litert import LiteRTExporter

exporter = LiteRTExporter(model)

# Check if dict adapter will be used
if exporter._has_dict_inputs():
    print("Model uses dict inputs - adapter will be created")
    
    # Get dict signature
    input_sig = model.inputs if model.built else None
    print(f"Input signature: {input_sig}")
    
    # Create adapter manually to inspect
    adapter = exporter._create_dict_adapter(input_sig)
    print(f"Adapter inputs: {adapter.inputs}")
    print(f"Adapter outputs: {adapter.outputs}")
else:
    print("Model uses list inputs - no adapter needed")
```

**8. Compare Inference Performance**

```python
import time
import numpy as np

def benchmark_inference(model_path, input_shape, num_runs=100):
    """Benchmark TFLite model inference speed."""
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        from tensorflow.lite import Interpreter
    
    interpreter = Interpreter(model_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Warmup
    sample = np.random.random(input_shape).astype(np.float32)
    for _ in range(10):
        interpreter.set_tensor(input_details[0]['index'], sample)
        interpreter.invoke()
    
    # Benchmark
    times = []
    for _ in range(num_runs):
        start = time.time()
        interpreter.set_tensor(input_details[0]['index'], sample)
        interpreter.invoke()
        _ = interpreter.get_tensor(output_details[0]['index'])
        times.append(time.time() - start)
    
    mean_ms = np.mean(times) * 1000
    std_ms = np.std(times) * 1000
    
    print(f"Inference time: {mean_ms:.2f}ms ± {std_ms:.2f}ms")
    print(f"Throughput: {1000/mean_ms:.1f} inferences/sec")
    
    return mean_ms

# Compare float32 vs quantized
print("Float32 model:")
benchmark_inference("model.tflite", (1, 224, 224, 3))

print("\nQuantized model:")
benchmark_inference("model_quantized.tflite", (1, 224, 224, 3))
```

### 9.4 Common Issues and Solutions

| Issue | Symptom | Solution |
|-------|---------|----------|
| **Unable to infer signature** | Export fails for subclassed model | Call model with sample data before export |
| **Wrong backend** | `RuntimeError: Backend must be TensorFlow` | Set `KERAS_BACKEND=tensorflow` before importing |
| **Out of memory** | `ResourceExhaustedError` | Use quantization, cloud machine, or export in steps |
| **Unsupported ops** | `ConverterError: Op X not supported` | Enable `SELECT_TF_OPS`, simplify custom layers |
| **Wrong file extension** | `ValueError: filepath must end with .tflite` | Use `.tflite` extension |
| **Empty model** | Very small file (<1MB for large model) | Check training, verify weights saved |
| **Slow inference** | >100ms for small model | Enable GPU delegate, optimize model |
| **Accuracy drop** | >5% difference from Keras | Use representative dataset for quantization |
| **Dict input error** | `TypeError: dict not supported` | Update Keras-Hub, use explicit signature |
| **Dynamic shapes** | Wrong output shapes | Specify `input_signature` with `None` dimensions |

#### Verbose Mode

```python
model.export("model.tflite", format="litert", verbose=True)
```

#### Inspect .tflite File

```python
# Use ai_edge_litert (preferred) or tf.lite (fallback)
try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    from tensorflow.lite import Interpreter

interpreter = Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Check inputs
input_details = interpreter.get_input_details()
for detail in input_details:
    print(f"Input: {detail['name']}")
    print(f"  Shape: {detail['shape']}")
    print(f"  Type: {detail['dtype']}")

# Check outputs
output_details = interpreter.get_output_details()
for detail in output_details:
    print(f"Output: {detail['name']}")
    print(f"  Shape: {detail['shape']}")
    print(f"  Type: {detail['dtype']}")
```

#### Test Inference Before Deployment

```python
import numpy as np

# Use ai_edge_litert (preferred) or tf.lite (fallback)
try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    from tensorflow.lite import Interpreter

# Load model
interpreter = Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Create dummy input
input_details = interpreter.get_input_details()
dummy_input = np.random.random(input_details[0]['shape']).astype(np.float32)

# Run test inference
interpreter.set_tensor(input_details[0]['index'], dummy_input)
interpreter.invoke()

# Get output
output_details = interpreter.get_output_details()
output = interpreter.get_tensor(output_details[0]['index'])
print("Output shape:", output.shape)
print("Output sample:", output[0, :5])
```

---

## 10. FAQ

### Q: Can I export models trained with JAX or PyTorch?

**A:** Yes! Train with any backend, save as `.keras`, then load with TensorFlow backend for export:

```python
# Train with JAX
os.environ["KERAS_BACKEND"] = "jax"
model.fit(...)
model.save("model.keras")

# Export with TensorFlow
os.environ["KERAS_BACKEND"] = "tensorflow"
model = keras.models.load_model("model.keras")
model.export("model.tflite", format="litert")
```

### Q: What's the difference between dynamic and fixed shapes?

**Dynamic shapes:**
- Flexible input dimensions
- Resize at runtime with `interpreter.resize_tensor_input()`
- Slightly slower first inference

**Fixed shapes:**
- Faster inference
- Better hardware acceleration
- Must preprocess inputs to exact size

### Q: How do I choose quantization method?

| Use Case | Recommendation |
|----------|----------------|
| Minimal accuracy loss | Float16 or Dynamic Range |
| Maximum speed | Full Integer + representative dataset |
| No dataset available | Dynamic Range |
| GPU deployment | Float16 |
| Edge TPU/DSP | Full Integer |

### Q: Why is my exported model so large?

Check if you're using quantization:

```python
import os

# Without quantization
model.export("model.tflite", format="litert")
print(f"Size: {os.path.getsize('model.tflite') / 1024 / 1024:.2f} MB")

# With quantization (~75% smaller)
model.export("model_quant.tflite", format="litert",
             litert_kwargs={"optimizations": [tf.lite.Optimize.DEFAULT]})
print(f"Size: {os.path.getsize('model_quant.tflite') / 1024 / 1024:.2f} MB")
```

### Q: Can I export custom layers?

Yes, but they must be TensorFlow-compatible. If you get conversion errors:

1. Ensure custom layer uses TensorFlow ops
2. Override `get_config()` and `from_config()`
3. Consider implementing as a Functional model instead

### Q: How do I handle preprocessing on mobile?

**Option 1:** Include preprocessing in the model
```python
# Add preprocessing layers
inputs = keras.Input(shape=(None, None, 3))
x = keras.layers.Rescaling(1./255)(inputs)
x = keras.layers.Resizing(224, 224)(x)
# ... rest of model
```

**Option 2:** Preprocess in mobile app (more flexible)
```kotlin
// Android example
val bitmap = BitmapFactory.decodeFile(imagePath)
val resized = Bitmap.createScaledBitmap(bitmap, 224, 224, true)
val normalized = normalizePixels(resized)  // /255.0
```

### Q: What's the maximum model size for mobile?

**Practical limits:**
- **Small models:** <50 MB (ideal for most apps)
- **Medium models:** 50-200 MB (acceptable)
- **Large models:** 200+ MB (consider on-demand download)
- **Very large models:** 1+ GB (requires special handling)

Use quantization to reduce size!

### Q: Can I update the model without app update?

Yes! Download `.tflite` files dynamically:

```kotlin
// Android example
val modelUrl = "https://yourserver.com/model.tflite"
val modelFile = File(context.filesDir, "model.tflite")

// Download model
downloadFile(modelUrl, modelFile)

// Load interpreter
val interpreter = Interpreter(modelFile)
```

### Q: My model works in Keras but fails on mobile. Why?

**Common reasons:**

1. **Input format mismatch**
   ```python
   # Keras (during training): dict inputs
   model({"token_ids": tensor1, "padding_mask": tensor2})
   
   # TFLite (exported): list inputs
   interpreter.run([tensor1, tensor2])  # Order matters!
   ```

2. **Data type mismatch**
   ```kotlin
   // Check expected dtype
   val inputDetails = interpreter.getInputDetails()
   println(inputDetails[0].dtype)  // Should match your input
   ```

3. **Shape mismatch**
   ```python
   # Verify exported shapes
   try:
       from ai_edge_litert.interpreter import Interpreter
   except ImportError:
       from tensorflow.lite import Interpreter
   
   interpreter = Interpreter("model.tflite")
   print(interpreter.get_input_details()[0]['shape'])
   ```

### Q: Can I export models trained with PyTorch backend?

Yes! Save weights with PyTorch, load with TensorFlow:

```python
# Train with PyTorch
import os
os.environ["KERAS_BACKEND"] = "torch"
import keras

model = keras.Sequential([...])
model.fit(X, y)
model.save_weights("weights.weights.h5")

# Export with TensorFlow
os.environ["KERAS_BACKEND"] = "tensorflow"
import keras as keras_tf

model = keras_tf.Sequential([...])  # Same architecture
model.load_weights("weights.weights.h5")
model.export("model.tflite", format="litert")
```

### Q: How do I handle batch inference on mobile?

**Option 1: Export with dynamic batch**
```python
import tensorflow as tf

model.export("model.tflite", format="litert",
             input_signature=[tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32)])
# First dimension is batch size (dynamic)
```

**Option 2: Export with fixed batch**
```python
# Batch size = 8
model.export("model_batch8.tflite", format="litert",
             input_signature=[tf.TensorSpec(shape=(8, 224, 224, 3), dtype=tf.float32)])
```

**Mobile inference:**
```kotlin
// Process multiple images at once
val batch = Array(8) { FloatArray(224 * 224 * 3) }
// Fill batch...

interpreter.run(batch, outputArray)
```

### Q: What's the difference between LiteRT and TFLite?

**LiteRT** is the new name for **TensorFlow Lite**. They are the same runtime:
- Old name: TensorFlow Lite (TFLite)
- New name: LiteRT
- File format: `.tflite` (unchanged)
- APIs: Compatible

Use `format="litert"` in Keras 3.x.

### Q: Can I use trained LoRA adapters with exported models?

Not directly. You need to merge LoRA weights before export:

```python
import keras_hub

# Load base model
model = keras_hub.models.GemmaCausalLM.from_preset("gemma_2b_en")

# Apply LoRA (training)
# ... fine-tuning code ...

# Merge LoRA weights into base model
model.merge_lora_weights()

# Now export
model.export("gemma_finetuned.tflite", format="litert")
```

### Q: How do I deploy multiple models in one app?

Load multiple interpreters:

```kotlin
class MultiModelInference(context: Context) {
    private val detectorInterpreter: Interpreter
    private val classifierInterpreter: Interpreter
    
    init {
        // Load detection model
        val detectorModel = loadModelFile(context, "detector.tflite")
        detectorInterpreter = Interpreter(detectorModel)
        
        // Load classification model
        val classifierModel = loadModelFile(context, "classifier.tflite")
        classifierInterpreter = Interpreter(classifierModel)
    }
    
    fun pipeline(image: ByteBuffer): Classification {
        // 1. Detect objects
        val boxes = detectorInterpreter.run(image)
        
        // 2. Classify detected regions
        val classifications = boxes.map { box ->
            val region = cropImage(image, box)
            classifierInterpreter.run(region)
        }
        
        return classifications
    }
    
    fun close() {
        detectorInterpreter.close()
        classifierInterpreter.close()
    }
}
```

### Q: Why is inference slow on mobile?

**Check these factors:**

1. **Not using hardware acceleration**
   ```kotlin
   // Enable GPU
   val options = Interpreter.Options()
   options.addDelegate(GpuDelegate())
   ```

2. **Model not quantized**
   ```python
   # Quantize to reduce computation
   model.export("model.tflite", format="litert",
                litert_kwargs={"optimizations": [tf.lite.Optimize.DEFAULT]})
   ```

3. **Wrong thread count**
   ```kotlin
   options.setNumThreads(4)  // Try different values
   ```

4. **Inefficient preprocessing**
   ```kotlin
   // Bad: Creating new bitmap each time
   Bitmap.createScaledBitmap(input, 224, 224, true)
   
   // Good: Reuse buffers
   val inputBuffer = ByteBuffer.allocateDirect(224 * 224 * 3 * 4)
   inputBuffer.rewind()
   // Fill buffer...
   ```

5. **Cold start overhead**
   ```kotlin
   // Warmup on app launch
   val dummyInput = createDummyInput()
   interpreter.run(dummyInput, dummyOutput)  // First run is slow
   ```

### Q: Can I convert TensorFlow SavedModel to LiteRT via Keras?

Yes:

```python
import keras

# Load SavedModel
model = keras.models.load_model("saved_model/")

# Export to LiteRT
model.export("model.tflite", format="litert")
```

### Q: How do I profile model performance?

**On Desktop:**
```python
import tensorflow as tf

# Create profiling tool
profiler = tf.lite.experimental.Profiler("model.tflite")

# Run with profiling
try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    from tensorflow.lite import Interpreter

interpreter = Interpreter("model.tflite")
interpreter.allocate_tensors()

# Enable profiling
profiler.profile()

# Run inference
sample_input = np.random.random((1, 224, 224, 3)).astype(np.float32)
interpreter.set_tensor(0, sample_input)
interpreter.invoke()

# Get profiling results
profiler.get_summary()
```

**On Android:**
```kotlin
// Add benchmark library
implementation 'org.tensorflow:tensorflow-lite-task-vision:0.4.4'

// Benchmark
val benchmark = BenchmarkHelper(modelFile)
val results = benchmark.run(numRuns = 100)
println("Average: ${results.averageTimeMs}ms")
println("Min: ${results.minTimeMs}ms")
println("Max: ${results.maxTimeMs}ms")
```

### Q: Can I use dropout during mobile inference?

No, and you shouldn't. Dropout is for training only:

```python
class MyModel(keras.Model):
    def call(self, x, training=False):
        x = self.dense1(x)
        x = self.dropout(x, training=training)  # Only active during training
        return self.dense2(x)

# Export automatically uses training=False
model.export("model.tflite", format="litert")
```

Mobile inference always runs with `training=False`.

### Q: How do I handle variable-length inputs (text)?

**Option 1: Set sequence length at model construction (recommended)**
```python
# Configure sequence length when creating the model
preprocessor = keras_hub.models.GemmaCausalLMPreprocessor.from_preset(
    "gemma_2b_en",
    sequence_length=512  # Set desired length here
)
model = keras_hub.models.GemmaCausalLM.from_preset(
    "gemma_2b_en",
    preprocessor=preprocessor
)
model.export("model.tflite", format="litert")

# Pad shorter inputs to 512 on mobile
```

**Option 2: Export with custom input_signature**
```python
import tensorflow as tf

model.export(
    "model.tflite",
    format="litert",
    input_signature=[tf.TensorSpec(shape=(None, None), dtype=tf.int32)]
)

# Mobile can use any length (slower)
```

**Recommendation:** Use fixed length (Option 1) for better performance.

### Q: Can I export Hugging Face models?

Convert to Keras first:

```python
from transformers import TFBertModel
import keras

# Load Hugging Face model (TensorFlow version)
hf_model = TFBertModel.from_pretrained("bert-base-uncased")

# Wrap in Keras model
inputs = keras.Input(shape=(128,), dtype="int32")
outputs = hf_model(inputs)[0]
keras_model = keras.Model(inputs=inputs, outputs=outputs)

# Export to LiteRT
keras_model.export("bert.tflite", format="litert")
```

Or use Keras-Hub's native implementations (recommended):
```python
import keras_hub

model = keras_hub.models.BertClassifier.from_preset("bert_base_en")
model.export("bert.tflite", format="litert")
```

---

## 11. Developer Guide: Custom Models and Advanced Usage

This section is for developers who want to export custom Keras models or understand
the export process in detail.

### 11.1 Overview

Keras-Hub models use the standard Keras `model.export()` API directly. The export 
functionality is implemented in Keras Core (`keras.src.export.litert`), and 
Keras-Hub models inherit this capability without any custom export handling.

**Key Points:**
- Export is handled by Keras Core, not by Keras-Hub
- Use `model.export(path, format="litert", litert_kwargs={...})`
- Input signatures are automatically inferred from the model
- Custom kwargs are passed via `litert_kwargs` dict

### 11.2 The Export API

#### Basic Export

```python
import keras

# Any Keras model (Sequential, Functional, or Subclassed)
model = keras.Sequential([...])

# Basic export
model.export("model.tflite", format="litert")

# Export with additional options
model.export(
    "model.tflite",
    format="litert",
    input_signature=custom_signature,  # Optional: override auto-inference
    litert_kwargs={
        "optimizations": [tf.lite.Optimize.DEFAULT],
        "representative_dataset": my_dataset
    }
)
```

#### What Keras Export Handles

The Keras LiteRT exporter (`keras.src.export.litert.LiteRTExporter`) automatically:
1. **Infers input signature** from the model's build configuration
2. **Handles nested inputs** (dicts, lists) by creating adapter models
3. **Converts to TFLite format** using TensorFlow's converter

### 11.3 Custom Subclassed Models

For subclassed models (inheriting from `keras.Model`), you must ensure the model
is built before export:

```python
import keras
import numpy as np

class CustomTextClassifier(keras.Model):
    """Custom text classification model."""
    
    def __init__(self, num_classes=2, **kwargs):
        super().__init__(**kwargs)
        self.embedding = keras.layers.Embedding(10000, 128)
        self.pooling = keras.layers.GlobalAveragePooling1D()
        self.classifier = keras.layers.Dense(num_classes, activation="softmax")
    
    def call(self, inputs, training=False):
        """Forward pass."""
        x = self.embedding(inputs)
        x = self.pooling(x)
        return self.classifier(x)

# Create model
model = CustomTextClassifier(num_classes=10)

# CRITICAL: Call model with sample data to build it
sample_input = np.zeros((1, 128), dtype=np.int32)
_ = model(sample_input)

# Now export
model.export("classifier.tflite", format="litert")
```

### 11.4 Custom Input Signatures

If automatic signature inference fails, provide an explicit input signature:

```python
import keras
import tensorflow as tf

model = CustomTextClassifier(num_classes=10)

# Define explicit input signature
input_signature = [
    tf.TensorSpec(shape=(None, 128), dtype=tf.int32, name="input")
]

# Export with custom signature
model.export(
    "classifier.tflite",
    format="litert",
    input_signature=input_signature
)
```

### 11.5 Common Pitfalls and Solutions

#### Pitfall 1: Not Building Subclassed Models

❌ **BAD:**
```python
model = CustomModel()
model.export("model.tflite", format="litert")  # ❌ May fail - model not built!
```

✅ **GOOD:**
```python
model = CustomModel()
_ = model(sample_input)  # Build the model first
model.export("model.tflite", format="litert")
```

#### Pitfall 2: Using Python Control Flow

❌ **BAD:**
```python
class BadModel(keras.Model):
    def call(self, inputs):
        if inputs.shape[1] > 100:  # ❌ Python condition won't trace
            return self.large_path(inputs)
        return self.small_path(inputs)
```

✅ **GOOD:**
```python
import tensorflow as tf

class GoodModel(keras.Model):
    def call(self, inputs):
        # Use TF ops for conditional logic
        condition = tf.shape(inputs)[1] > 100
        return tf.cond(
            condition,
            lambda: self.large_path(inputs),
            lambda: self.small_path(inputs)
        )
```

#### Pitfall 3: Not Handling litert_kwargs Properly

❌ **BAD:**
```python
# Passing optimizations directly (wrong)
model.export("model.tflite", format="litert", optimizations=[...])
```

✅ **GOOD:**
```python
# Use litert_kwargs dict
model.export(
    "model.tflite",
    format="litert",
    litert_kwargs={"optimizations": [...]}
)
```

### 11.6 Testing Export

```python
import pytest
import tempfile
import os

def test_litert_export(model, input_data):
    """Test LiteRT export for a model."""
    with tempfile.TemporaryDirectory() as temp_dir:
        filepath = os.path.join(temp_dir, "model.tflite")
        
        # Export
        model.export(filepath, format="litert")
        
        # Verify file exists
        assert os.path.exists(filepath)
        assert os.path.getsize(filepath) > 0
        
        # Load and test inference
        try:
            from ai_edge_litert.interpreter import Interpreter
        except ImportError:
            from tensorflow.lite import Interpreter
        
        interpreter = Interpreter(model_path=filepath)
        interpreter.allocate_tensors()
        
        # Run inference
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Set inputs and invoke
        for i, (name, data) in enumerate(input_data.items()):
            interpreter.set_tensor(input_details[i]['index'], data)
        interpreter.invoke()
        
        # Verify output
        output = interpreter.get_tensor(output_details[0]['index'])
        assert output is not None
```

### 11.7 Common Issues and Solutions

| Issue | Symptom | Solution |
|-------|---------|----------|
| **Model not built** | `Model inputs are not defined` | Call model with sample data before export |
| **Wrong backend** | `Backend must be TensorFlow` | Set `KERAS_BACKEND=tensorflow` before importing |
| **Invalid kwargs** | `Unknown converter attribute` | Use `litert_kwargs` dict for converter options |
| **Python control flow** | Incorrect graph | Use TensorFlow ops (`tf.cond`, `tf.while_loop`) |
| **Dynamic shapes** | Wrong output shapes | Specify `input_signature` explicitly |

---

## Next Steps

- **Examples:** Check `keras-hub/examples/` for complete examples
- **API Reference:** See [keras.io](https://keras.io) for full API documentation
- **Community:** Join [Keras Discord](https://discord.gg/keras) for support

---

**Document Version:** 2.0  
**Last Updated:** 2025  
**Feedback:** [GitHub Issues](https://github.com/keras-team/keras/issues)
