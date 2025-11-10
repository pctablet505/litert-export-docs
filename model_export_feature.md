# LiteRT Model Export: Unified Design for Keras and Keras-Hub

**Self link**: [Keras PR #21674](https://github.com/keras-team/keras/pull/21674) | [Keras-Hub PR #2405](https://github.com/keras-team/keras-hub/pull/2405)  
**Visibility**: Public (Open Source)  
**Status**: Stage 1 Complete ✅ | Stage 2 Planned 🚧  
**Authors**: Keras Export Team (@fchollet, @mattdangerw, @divyashreepathihalli)  
**Contributors**: @sampathweb, @hertschuh  
**Team**: Keras Core Team  
**Tracking Issue**: [Keras #21674](https://github.com/keras-team/keras/pull/21674), [Keras-Hub #2405](https://github.com/keras-team/keras-hub/pull/2405)  
**Last major revision**: November 10, 2025

---

# Context

## Objective

**Stage 1 (This Document)**: Enable seamless one-line export of Keras and Keras-Hub models to LiteRT (TensorFlow Lite) format for on-device inference, making mobile and edge deployment accessible to all Keras users without requiring manual TFLite converter knowledge.

**Stage 2 (Future Work)**: Develop comprehensive Android/iOS runtime libraries with preprocessing and postprocessing APIs to simplify on-device inference. This will include:
- C++/Java/Kotlin inference wrappers for mobile platforms
- Task-specific preprocessing (tokenization for text models, image normalization for vision models)
- Postprocessing utilities (argmax, NMS, detokenization, etc.)
- Unified API across different model types
- See [Android Runtime Library Design](https://github.com/keras-team/keras-hub/issues/XXXXX) for details *(placeholder)*

## Background

**The Problem**: 
- Converting Keras models to TensorFlow Lite for mobile/edge deployment is complex and error-prone
- Different model architectures (Sequential, Functional, Subclassed) require different conversion approaches
- Keras-Hub task models with dictionary inputs are incompatible with TFLite's tensor-based interface
- Users must manually handle input signatures, variable tracking, and stateful layers
- No unified API exists across Keras Core and Keras-Hub

**Why It's Important**:
- On-device ML is critical for privacy, low latency, offline functionality, and cost reduction
- LiteRT is a framework-agnostic runtime for neural network inference on mobile/edge devices
- Keras has 1M+ users who need simplified export workflows
- Current manual conversion requires 100+ lines of boilerplate per model type

**Existing Solutions**:
- TensorFlow's `tf.lite.TFLiteConverter` - low-level, requires expertise
- PyTorch models: Can convert directly via [PyTorch to LiteRT](https://ai.google.dev/edge/litert/models/convert_pytorch)
- JAX models: Can convert via [JAX to LiteRT](https://ai.google.dev/edge/litert/models/convert_jax) (similar to Keras JAX backend approach)
- ONNX export: Different format, not optimized for mobile

---

# Design

## Overview

This document describes the **end-to-end LiteRT (TFLite) export system** spanning both Keras and Keras-Hub. The implementation is split across two repositories with clear separation of concerns:

### Two-Tier Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Keras-Hub Export Layer                       │
│  - Task-specific configurations (CausalLM, ImageClassifier)     │
│  - Input adaptation (dict → list conversion)                    │
│  - Model-type registry                                          │
│  - Extends model.export() for task models                       │
└────────────────────────┬────────────────────────────────────────┘
                         │ Delegates to
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│                    Keras Core Export Layer                      │
│  - Base model.export() API                                      │
│  - TFLite converter integration                                 │
│  - Multi-strategy conversion (direct/wrapper)                   │
│  - Stateful layer support                                       │
│  - AOT compilation                                              │
└─────────────────────────────────────────────────────────────────┘
```

### What Was Implemented Where and Why

| Feature | Repository | Rationale |
|---------|-----------|-----------|
| **`model.export()` base API** | Keras Core | Universal interface for all Keras models |
| **TFLite converter logic** | Keras Core | Core conversion, all Keras models benefit |
| **Wrapper-based conversion** | Keras Core | Handles subclassed models universally |
| **Sequential/Functional direct conversion** | Keras Core | Fast path for simple models |
| **AOT compilation support** | Keras Core | Hardware optimization, model-agnostic |
| **Task model configurations** | Keras-Hub | Specific to NLP/CV task models |
| **Input structure adaptation** | Keras-Hub | Handles dict inputs unique to task models |
| **Model type registry** | Keras-Hub | Maps task models to export configs |
| **Preset model support** | Keras-Hub | Integration with model zoo |

### Why Split Across Two Repositories?

**Keras Core** provides the foundation:
- ✅ General-purpose export for **any** Keras model
- ✅ No dependencies on task-specific logic
- ✅ Optimized conversion strategies
- ✅ Clean, minimal API surface

**Keras-Hub** adds specialization:
- ✅ Task-specific export configurations
- ✅ Input structure handling (dictionary → list)
- ✅ Model-type-aware export parameters
- ✅ Preset model compatibility
- ❌ Would bloat Keras Core with NLP/CV-specific code
- ❌ Requires knowledge of task model architectures

### Quick Start Examples

> **Note**: For detailed usage examples and best practices, see the separate [LiteRT Export User Guide](https://github.com/keras-team/keras-hub/blob/main/docs/guides/litert_export.md) *(placeholder link)*.

#### Keras Core: Standard Models

```python
import keras

# Sequential model - one-line export
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(100,)),
    keras.layers.Dense(10, activation='softmax')
])
model.export('model.tflite', format='litert')
```

#### Keras-Hub: Task Models

```python
import keras_hub

# Text classifier
classifier = keras_hub.models.BertClassifier.from_preset("bert_base_en", num_classes=2)
keras_hub.export.export_litert(classifier, 'bert.tflite', max_sequence_length=512)

# Image classifier (size auto-inferred from preprocessor)
img_model = keras_hub.models.ImageClassifier.from_preset("resnet_50_imagenet")
keras_hub.export.export_litert(img_model, 'resnet50.tflite')
```

### Key Features Across Both Systems

#### Keras Core Features (PR #21674)
- ✅ **Unified `model.export()` API** for all Keras models
- ✅ **Multi-strategy conversion** (direct Sequential/Functional, wrapper for Subclassed)
- ✅ **Stateful layer support** (LSTM, GRU, BatchNormalization)
- ✅ **Variable tracking** for complex model graphs
- ✅ **AOT compilation** for hardware-specific optimization
- ✅ **Automatic input signature inference**
- ✅ **Resource variable handling**
- ✅ **TensorFlow backend dependency** (clean separation)

#### Keras-Hub Features (PR #2405)
- ✅ **Task model configurations** (CausalLM, TextClassifier, Seq2SeqLM, ImageClassifier, ObjectDetector, ImageSegmenter)
- ✅ **Input adaptation layer** (dict ↔ list conversion for TFLite)
- ✅ **Model adapter pattern** (TextModelAdapter, ImageModelAdapter, BaseModelAdapter)
- ✅ **Registry pattern** following HuggingFace Optimum design
- ✅ **Per-model-type export parameters** (sequence_length for text, image_size for vision)
- ✅ **Preset model support** (100+ models from model zoo)
- ✅ **Clear error messages** for unsupported model types
- ✅ **Extensible architecture** for future task types

### Supported Model Types

| Model Type | Library | Input Format | Example Models | Status |
|------------|---------|--------------|----------------|---------|
| Sequential/Functional | Keras Core | Tensor/list | Any custom model | ✅ |
| Subclassed | Keras Core | Tensor/list | Custom nn.Module style | ✅ |
| **CausalLM** | **Keras-Hub** | **Dict: `{token_ids, padding_mask}`** | **GPT, Gemma, LLaMA** | **✅** |
| **TextClassifier** | **Keras-Hub** | **Dict: `{token_ids, padding_mask}`** | **BERT, RoBERTa, DistilBERT** | **✅** |
| **Seq2SeqLM** | **Keras-Hub** | **Dict: `{encoder_*, decoder_*}`** | **T5, BART, Encoder-Decoder** | **✅** |
| **ImageClassifier** | **Keras-Hub** | **Tensor or Dict: `{images}`** | **ResNet, EfficientNet, ViT** | **✅** |
| **ObjectDetector** | **Keras-Hub** | **Tensor or Dict: `{images}`** | **YOLO, RetinaNet, DETR** | **✅** |
| **ImageSegmenter** | **Keras-Hub** | **Tensor or Dict: `{images}`** | **DeepLab, Segment Anything** | **✅** |
| Audio Models | Keras-Hub | - | Wav2Vec2, Whisper | 🚧 Future |
| Multimodal | Keras-Hub | - | CLIP, LLaVA | 🚧 Future |

---

## 1. Background and Motivation

### 1.1 The Challenge: Exporting ML Models to Edge Devices

**On-Device ML** is critical for:
- Privacy (no data leaves device)
- Low latency (no network roundtrip)
- Offline functionality
- Cost reduction (no server costs)

**TensorFlow Lite (LiteRT)** is the industry standard for mobile/edge deployment, but conversion is complex:
- Different model architectures need different strategies
- Input signature inference is error-prone
- Stateful layers (LSTM, BatchNorm) require special handling
- Task models (NLP, Vision) have unique input structures

### 1.2 State Before These PRs

**For Keras Users (Before PR #21674)**: Users had to manually use `tf.lite.TFLiteConverter` with 3 different approaches (SavedModel, direct conversion, or wrapper pattern), each with different failure modes. No unified API, unclear error messages, and no support for stateful layers.

**For Keras-Hub Users (Before PR #2405)**: Keras-Hub models couldn't be exported at all due to dictionary inputs being incompatible with TFLite's tensor-based interface. Users needed 100+ lines of boilerplate per model type to manually create wrappers, handle signatures, and track variables.

### 1.3 What These PRs Solve

#### Keras Core (PR #21674) Solved
✅ **Unified API**: `model.export('model.tflite', format='litert')`  
✅ **Multi-strategy conversion**: Automatic fallback for different model types  
✅ **Stateful layer support**: LSTM, GRU, BatchNormalization work correctly  
✅ **Clear errors**: Informative messages guide users  
✅ **AOT compilation**: Hardware-specific optimization  
✅ **Input signature inference**: Automatic for most cases  

#### Keras-Hub (PR #2405) Solved
✅ **Task model export**: CausalLM, TextClassifier, ImageClassifier, etc.  
✅ **Input adaptation**: Dict → List conversion automatic  
✅ **Per-model-type configs**: Sequence length for text, image size for vision  
✅ **Preset model support**: Works with `from_preset()` models  
✅ **Registry pattern**: Extensible for future task types  
✅ **Clear validation**: Explicit errors for unsupported models  

### 1.4 Design Goals

**Primary Goals**:
1. **Ease of Use**: One-line export for 99% of use cases
2. **Correctness**: Numerical accuracy guaranteed
3. **Robustness**: Handle edge cases gracefully
4. **Extensibility**: Easy to add new model types
5. **Performance**: Fast conversion, optimized models
6. **Clarity**: Clear error messages and documentation

**Non-Goals** (Explicitly Out of Scope):
- ❌ Quantization via Keras API (use TFLite converter's quantization features)
- ❌ Custom op registration (use TFLite's mechanisms)
- ❌ Model compression/pruning (separate concern)
- ❌ Real-time conversion (focus on correctness)

---

## 2. Design Philosophy

### 2.1 Separation of Concerns

The implementation follows a **clear layered architecture**:

```
┌──────────────────────────────────────────────────────────────────┐
│ User Code: model.export('model.tflite', format='litert')        │
└─────────────────────────┬────────────────────────────────────────┘
                          │
        ┌─────────────────┴──────────────────┐
        │ Is this a Keras-Hub task model?    │
        └─────────────────┬──────────────────┘
                │                     │
          YES   │                     │ NO
                │                     │
                ↓                     ↓
    ┌───────────────────┐   ┌────────────────────┐
    │  Keras-Hub Layer  │   │   Keras Core       │
    │  (PR #2405)       │   │   (PR #21674)      │
    ├───────────────────┤   ├────────────────────┤
    │ 1. Get config     │   │ 1. Infer signature │
    │    for model type │   │ 2. Choose strategy │
    │ 2. Create adapter │   │ 3. Convert to      │
    │    (dict→list)    │   │    TFLite          │
    │ 3. Delegate to    │   └────────────────────┘
    │    Keras exporter │             │
    └─────────┬─────────┘             │
              │                       │
              └───────────┬───────────┘
                          ↓
              ┌───────────────────────┐
              │  TFLite Converter     │
              │  (TensorFlow)         │
              └───────────────────────┘
                          │
                          ↓
                  model.tflite file
```

### 2.2 Key Principles

1. **Delegate, Don't Duplicate**: Keras-Hub uses Keras Core's conversion, doesn't reimplement it
2. **Fail Fast**: Validate early, provide clear errors
3. **Convention Over Configuration**: Sensible defaults, explicit only when needed
4. **Extensibility**: Easy to add new model types via registry
5. **Testability**: Mock models for fast tests, real models for E2E validation

### 2.3 Why This Split?

| Aspect | Keras Core | Keras-Hub | Reason |
|--------|-----------|-----------|---------|
| **Model Types** | Any Keras model | Task models only | Core stays general-purpose |
| **Dependencies** | TensorFlow only | Keras Core + task models | Keras-Hub imports Keras, not vice versa |
| **Input Handling** | Tensors, lists | Dicts, tensors, lists | Task models have unique requirements |
| **Configuration** | Generic | Per-task-type | Different tasks need different params |
| **Complexity** | Converter logic | Adapter logic | Each layer has clear responsibility |
| **Release Cycle** | Stable | Fast-moving | Can iterate Hub without Keras changes |

---

## 3. System Architecture

### 3.1 Overall System Diagram

The export system has a three-layer architecture:

1. **User Code** → Calls `model.export('file.tflite', format='litert')`
2. **Keras-Hub Layer** (for task models) → Detects model type from registry, creates input adapters (dict→list conversion), delegates to Keras Core
3. **Keras Core Layer** → Ensures model is built, infers input signatures, chooses conversion strategy, handles variable tracking
4. **TensorFlow LiteRT Converter** → Performs actual TFLite conversion with optimizations

### 3.2 Component Responsibilities

#### Keras Core Components

| Component | File | Responsibility |
|-----------|------|----------------|
| **`model.export()`** | `keras/src/models/model.py` | Entry point, format routing |
| **`export_litert()`** | `keras/src/export/litert.py` | Main export orchestration |
| **`LiteRTExporter`** | `keras/src/export/litert.py` | Conversion logic, strategy selection |
| **`KerasModelWrapper`** | `keras/src/export/litert.py` | Variable tracking for wrapper strategy |
| **Input signature utils** | `keras/src/export/export_utils.py` | Signature inference/validation |

#### Keras-Hub Components

| Component | File | Responsibility |
|-----------|------|----------------|
| **`ExporterRegistry`** | `keras_hub/src/export/registry.py` | Maps models to configs |
| **`export_model()`** | `keras_hub/src/export/registry.py` | Task model export entry |
| **`Task.export()`** | `keras_hub/src/export/registry.py` | Extension of Task class |
| **`KerasHubExporterConfig`** | `keras_hub/src/export/base.py` | Base config class |
| **`CausalLMExporterConfig`** | `keras_hub/src/export/configs.py` | Text generation config |
| **`ImageClassifierExporterConfig`** | `keras_hub/src/export/configs.py` | Image classification config |
| **... (4 more configs)** | `keras_hub/src/export/configs.py` | Other task configs |
| **`LiteRTExporter`** | `keras_hub/src/export/litert.py` | Keras-Hub LiteRT exporter |
| **`TextModelAdapter`** | `keras_hub/src/export/litert.py` | Dict→List for text models |
| **`ImageModelAdapter`** | `keras_hub/src/export/litert.py` | Flexible input for vision |
| **`BaseModelAdapter`** | `keras_hub/src/export/litert.py` | Fallback adapter |

### 3.3 Data Flow

**Keras-Hub Task Model Flow** (e.g., CausalLM, ImageClassifier):
```
User: model.export('model.tflite', format='litert')
  │
  ├─→ [Keras-Hub Task.export()] → Detects model type, gets config from registry
  ├─→ [LiteRTExporter (Keras-Hub)] → Creates adapter wrapper (dict→list conversion)
  ├─→ [LiteRTExporter (Keras Core)] → Infers signature, chooses strategy, handles variables
  ├─→ [TFLite Converter] → Converts to TFLite format with optimizations
  └─→ Output: model.tflite
```

**Direct Keras Core Flow** (Sequential/Functional models): Skips Keras-Hub layer, goes directly to Keras Core export → TFLite converter.

### 3.4 Adapter Pattern Deep Dive

The **Model Adapter** is the key abstraction enabling Keras-Hub to work with 
Keras Core's converter:

```python
# Problem: Keras-Hub models accept dicts
def call(self, inputs):  # inputs = {"token_ids": ..., "padding_mask": ...}
    token_ids = inputs["token_ids"]
    padding_mask = inputs["padding_mask"]
    # ...

# Solution: Adapter wraps model to accept lists
class TextModelAdapter(keras.Model):
    def __init__(self, inner_model):
        super().__init__()
        self.inner_model = inner_model
    
    def call(self, inputs):  # inputs = [token_ids, padding_mask]
        # Convert list → dict
        model_inputs = {
            "token_ids": inputs[0],
            "padding_mask": inputs[1],
        }
        return self.inner_model(model_inputs)

# Now Keras Core can convert it:
wrapped = TextModelAdapter(task_model)
wrapped.export('model.tflite', format='litert')  # ✓ Works!
```

**Why this works:**
- TFLite converter prefers positional (list) inputs
- Keras-Hub models use named (dict) inputs
- Adapter bridges the gap without modifying either layer

---

## 4. Integration Between Keras and Keras-Hub

### 4.1 How Keras-Hub Extends Keras Core

Keras-Hub **imports and uses** Keras Core's export functionality:

```python
# In keras_hub/src/export/litert.py
from keras.src.export.litert import LiteRTExporter as KerasLiteRTExporter
from keras.src.export.litert import export_litert as keras_export_litert

class LiteRTExporter(KerasHubExporter):
    """Keras-Hub's LiteRT exporter that delegates to Keras Core."""
    
    def export(self, model, filepath, **kwargs):
        # 1. Create adapter wrapper
        adapter = self._create_adapter(model)
        
        # 2. Delegate to Keras Core
        return keras_export_litert(
            model=adapter,
            filepath=filepath,
            input_signature=self._get_signature(model),
            **kwargs
        )
```

**Key Insight**: Keras-Hub doesn't reimplement conversion logic. It only:
1. Determines the correct adapter type
2. Creates the adapter wrapper
3. Calls Keras Core's `export_litert()` with the wrapped model

### 4.2 Dependency Graph

```
┌────────────────────────────────────────────────┐
│  User Application                              │
│  - Imports keras_hub                           │
│  - Uses task models                            │
└───────────────────┬────────────────────────────┘
                    │ depends on
                    ↓
┌────────────────────────────────────────────────┐
│  Keras-Hub (PR #2405)                          │
│  ┌──────────────────────────────────────────┐ │
│  │ Task Models (CausalLM, Classifier, etc.) │ │
│  └────────────────┬─────────────────────────┘ │
│                   │                            │
│  ┌────────────────┴─────────────────────────┐ │
│  │ Export System                            │ │
│  │ - Registry                               │ │
│  │ - Configs                                │ │
│  │ - LiteRTExporter (Hub)                   │ │
│  │ - Model Adapters                         │ │
│  └────────────────┬─────────────────────────┘ │
└───────────────────┼──────────────────────────┘
                    │ imports
                    ↓
┌────────────────────────────────────────────────┐
│  Keras Core (PR #21674)                        │
│  ┌──────────────────────────────────────────┐ │
│  │ Base Model Class                         │ │
│  │ - model.export() method                  │ │
│  └────────────────┬─────────────────────────┘ │
│                   │                            │
│  ┌────────────────┴─────────────────────────┐ │
│  │ Export Infrastructure                    │ │
│  │ - export_litert()                        │ │
│  │ - LiteRTExporter (Core)                  │ │
│  │ - Input signature utils                  │ │
│  │ - KerasModelWrapper                      │ │
│  └────────────────┬─────────────────────────┘ │
└───────────────────┼──────────────────────────┘
                    │ uses
                    ↓
┌────────────────────────────────────────────────┐
│  TensorFlow                                    │
│  - tf.lite.TFLiteConverter                     │
│  - tf.function / ConcreteFunction              │
│  - SavedModel utilities                        │
└────────────────────────────────────────────────┘
```

### 4.3 API Compatibility

Both layers expose the **same interface** to users:

```python
# Keras Core API
model = keras.Sequential([...])
model.export('model.tflite', format='litert')

# Keras-Hub API (identical signature!)
model = keras_hub.models.BertClassifier.from_preset('bert_base_en')
model.export('model.tflite', format='litert')

# Both support same kwargs:
model.export('model.tflite', format='litert', 
             litert_kwargs={
                 'aot_compile_targets': ['qualcomm'],
                 'verbose': True
             })
```

**Design Benefit**: Users don't need to learn different APIs. The experience is 
consistent whether using Keras Core or Keras-Hub.

### 4.4 Error Handling Coordination

Errors can originate from either layer:

```python
# Keras-Hub errors (early validation)
try:
    audio_model.export('audio.tflite')  # AudioClassifier not supported yet
except ValueError as e:
    # "Unsupported model type 'audio_classifier'. 
    #  Supported types: text_classifier, causal_lm, ..."

# Keras Core errors (conversion failures)
try:
    model.export('model.tflite', format='litert',
                 litert_kwargs={'input_signature': invalid_signature})
except ValueError as e:
    # "Input signature mismatch: expected 2 inputs, got 1"
```

**Principle**: Each layer validates what it's responsible for:
- **Keras-Hub**: Model type support, adapter creation, config validation
- **Keras Core**: Signature inference, conversion strategy, TFLite conversion

---

# PART II: KERAS CORE IMPLEMENTATION (PR #21674)

## 5. Keras Core: Base Export Infrastructure

### 5.1 Entry Point: `model.export()`

**Location**: `keras/src/models/model.py`

```python
class Model:
    def export(self, filepath, format=None, **kwargs):
        """Export model to various formats.
        
        Args:
            filepath: Destination path (e.g., 'model.tflite')
            format: Export format ('litert', 'saved_model', 'onnx', etc.)
            **kwargs: Format-specific arguments
        
        Raises:
            ValueError: If format is unsupported or args are invalid
        """
        if format == "litert":
            from keras.src.export.litert import export_litert
            return export_litert(self, filepath, **kwargs.get('litert_kwargs', {}))
        elif format == "saved_model":
            # ... other formats
        else:
            raise ValueError(f"Unsupported format: {format}")
```

**Key Design Decisions**:
- **Format parameter**: Explicit format selection (not inferred from extension)
- **Nested kwargs**: `litert_kwargs` allows format-specific params without cluttering API
- **Lazy imports**: Only import converters when needed (faster startup)

### 5.2 Core Exporter: `LiteRTExporter`

**Location**: `keras/src/export/litert.py`

```python
class LiteRTExporter:
    """Main class responsible for converting Keras models to LiteRT."""
    
    def export(
        self,
        model: keras.Model,
        filepath: str,
        input_signature: Optional[List[tf.TensorSpec]] = None,
        aot_compile_targets: Optional[List[str]] = None,
        verbose: bool = False,
    ):
        """Convert Keras model to TFLite format.
        
        Args:
            model: Keras model to export
            filepath: Destination .tflite file path
            input_signature: Optional input shape/dtype specs
            aot_compile_targets: Optional list of hardware targets
                                  for AOT compilation
            verbose: Print conversion details
        
        Process:
            1. Ensure model is built
            2. Infer or validate input signature
            3. Choose conversion strategy
            4. Convert to TFLite
            5. Write to file
        """
        # Implementation details in next sections
```

### 5.3 Conversion Strategies

Keras Core uses **two different strategies** depending on model architecture:

#### Strategy 1: Direct Conversion (Preferred)

**When Used**: Sequential and Functional API models

**How It Works**:
```python
# Model is already a computational graph
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]
tflite_model = converter.convert()
```

**Advantages**:
- ✅ Faster conversion
- ✅ Better optimization potential
- ✅ Smaller model size
- ✅ No wrapper overhead

**Limitations**:
- ❌ Only works for Sequential/Functional models
- ❌ Fails for subclassed models with dynamic control flow

#### Strategy 2: Wrapper-Based Conversion

**When Used**: Subclassed models, models with complex control flow

**How It Works**:
```python
class KerasModelWrapper(tf.Module):
    """Wraps a Keras model for TFLite conversion via tf.function."""
    
    def __init__(self, keras_model):
        super().__init__()
        self.keras_model = keras_model
        # Track all variables
        self._track_variables(keras_model)
    
    @tf.function
    def __call__(self, *args):
        return self.keras_model(*args)
    
    def _track_variables(self, model):
        """Ensure all variables are tracked for export."""
        for layer in model.layers:
            for weight in layer.weights:
                # Track each variable
                self._track_weight(weight)

# Convert via ConcreteFunction
wrapper = KerasModelWrapper(model)
concrete_func = wrapper.__call__.get_concrete_function(input_signature)
converter = tf.lite.TFLiteConverter.from_concrete_functions(
    [concrete_func], wrapper
)
tflite_model = converter.convert()
```

**Advantages**:
- ✅ Works for all model types (subclassed, complex control flow)
- ✅ Handles stateful layers (LSTM, GRU, BatchNorm)
- ✅ Supports dynamic shapes

**Limitations**:
- ❌ Slower conversion
- ❌ Potential size overhead from wrapper

### 5.4 Input Signature Inference

**Location**: `keras/src/export/export_utils.py`

```python
def infer_input_signature(model: keras.Model) -> List[tf.TensorSpec]:
    """Automatically infer input signature from model's input layers.
    
    Args:
        model: Keras model (must be built)
    
    Returns:
        List of TensorSpec describing input shapes and dtypes
    
    Raises:
        ValueError: If model is not built or inputs are ambiguous
    """
    if not model.built:
        raise ValueError("Model must be built before export. "
                         "Call model.build(input_shape) or model(sample_input).")
    
    # Single input
    if hasattr(model, 'input') and not isinstance(model.input, list):
        return [tf.TensorSpec(
            shape=model.input.shape,
            dtype=model.input.dtype,
            name=model.input.name
        )]
    
    # Multiple inputs
    elif hasattr(model, 'inputs'):
        return [
            tf.TensorSpec(
                shape=inp.shape,
                dtype=inp.dtype,
                name=inp.name
            )
            for inp in model.inputs
        ]
    
    else:
        raise ValueError("Cannot infer input signature from model. "
                         "Please provide 'input_signature' explicitly.")
```

**Design Choice**: Automatic inference reduces boilerplate, but users can 
always override with explicit `input_signature`.

### 5.5 Stateful Layer Support

Keras Core handles **stateful layers** (LSTM, GRU, BatchNormalization) 
via ResourceVariable tracking:

```python
class KerasModelWrapper(tf.Module):
    def _track_variables(self, model):
        """Track all trainable and non-trainable variables."""
        # Trainable variables (weights, kernels)
        for var in model.trainable_variables:
            self._track_variable(var)
        
        # Non-trainable variables (batch norm moving mean/variance, 
        # LSTM states)
        for var in model.non_trainable_variables:
            self._track_variable(var)
    
    def _track_variable(self, var):
        """Register variable with TF Module tracking system."""
        # TensorFlow will serialize this in the SavedModel/TFLite
        setattr(self, f'var_{id(var)}', var)
```

**Why This Matters**:
- BatchNormalization has `moving_mean` and `moving_variance` (non-trainable)
- LSTM has hidden states
- Without tracking, these variables would be lost in conversion
- TFLite model would give incorrect results

### 5.6 AOT Compilation Support

**Ahead-of-Time (AOT) compilation** optimizes models for specific hardware:

```python
def export_litert(
    model,
    filepath,
    aot_compile_targets=None,  # e.g., ['qualcomm', 'mediatek']
    **kwargs
):
    # Standard conversion
    tflite_model = converter.convert()
    
    # If AOT targets specified, invoke compiler
    if aot_compile_targets:
        for target in aot_compile_targets:
            aot_model = compile_for_target(tflite_model, target)
            aot_filepath = filepath.replace('.tflite', f'.{target}.tflite')
            with open(aot_filepath, 'wb') as f:
                f.write(aot_model)
    
    # Write base model
    with open(filepath, 'wb') as f:
        f.write(tflite_model)
```

**Supported Targets**:
- `qualcomm`: Qualcomm Hexagon DSP
- `mediatek`: MediaTek APU
- `samsung`: Samsung NPU
- (More added as TFLite team expands support)

---

## 6. Keras Core: API Design

### 6.1 Public API

```python
# Method 1: Via model.export() (recommended)
model.export('model.tflite', format='litert')

# Method 2: Direct function call (advanced)
from keras.src.export.litert import export_litert
export_litert(model, 'model.tflite', input_signature=[...])

# Method 3: Via exporter class (rare, for customization)
from keras.src.export.litert import LiteRTExporter
exporter = LiteRTExporter()
exporter.export(model, 'model.tflite')
```

### 6.2 Configuration Options

All LiteRT-specific options are passed via `litert_kwargs`:

```python
model.export(
    'model.tflite',
    format='litert',
    litert_kwargs={
        # Input specification (optional, auto-inferred if omitted)
        'input_signature': [
            tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32)
        ],
        
        # AOT compilation targets (optional)
        'aot_compile_targets': ['qualcomm', 'mediatek'],
        
        # Verbose output (optional, default=False)
        'verbose': True,
        
        # Custom TFLite converter options (advanced)
        'converter_options': {
            'optimizations': [tf.lite.Optimize.DEFAULT],
            'target_spec': {...},
        }
    }
)
```

### 6.3 Error Messages

Keras Core provides **clear, actionable error messages**:

```python
# Error 1: Model not built
# → Model must be built before export. Call model.build(input_shape).

# Error 2: Invalid input signature
# → Input signature mismatch: model expects 2 inputs but signature 
#   has 1. Please provide a signature with 2 TensorSpec objects.

# Error 3: Unsupported layer
# → Layer 'custom_layer' is not supported for LiteRT export. 
#   Supported layer types: Dense, Conv2D, LSTM, ...

# Error 4: Conversion failure
# → TFLite conversion failed. This may be due to:
#   1. Unsupported operations in model
#   2. Dynamic control flow (if/while loops)
#   3. Ragged tensors
#   Try using 'verbose=True' for detailed logs.
```

---

## 7. Keras Core: Testing Strategy

### 7.1 Test Structure

**Location**: `keras/src/export/litert_test.py`

```python
class LiteRTExportTest(tf.test.TestCase):
    """Tests for Keras Core LiteRT export."""
    
    def test_sequential_model_export(self):
        """Test exporting a Sequential model."""
    
    def test_functional_model_export(self):
        """Test exporting a Functional API model."""
    
    def test_subclassed_model_export(self):
        """Test exporting a subclassed model (wrapper strategy)."""
    
    def test_multi_input_model(self):
        """Test model with multiple inputs."""
    
    def test_multi_output_model(self):
        """Test model with multiple outputs."""
    
    def test_stateful_layers(self):
        """Test LSTM, GRU, BatchNormalization."""
    
    def test_custom_input_signature(self):
        """Test providing explicit input signature."""
    
    def test_aot_compilation(self):
        """Test AOT compilation for hardware targets."""
    
    def test_error_handling(self):
        """Test error messages for common failure modes."""
```

### 7.2 Testing Philosophy

1. **Correctness**: Verify exported model produces same outputs as original
2. **Robustness**: Test edge cases (empty models, single-layer, very deep)
3. **Error Quality**: Ensure errors are clear and actionable
4. **Performance**: Track export time for regression detection

### 7.3 Example Test

```python
def test_sequential_model_export(self):
    # Create simple model
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(32,)),
        keras.layers.Dense(10, activation='softmax')
    ])
    
    # Export to TFLite
    filepath = self.get_temp_dir() + '/model.tflite'
    model.export(filepath, format='litert')
    
    # Load exported model
    interpreter = tf.lite.Interpreter(model_path=filepath)
    interpreter.allocate_tensors()
    
    # Test numerical equivalence
    test_input = np.random.random((1, 32)).astype(np.float32)
    
    # Original Keras model output
    keras_output = model(test_input).numpy()
    
    # TFLite model output
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], test_input)
    interpreter.invoke()
    tflite_output = interpreter.get_tensor(output_details[0]['index'])
    
    # Assert outputs match within tolerance
    self.assertAllClose(keras_output, tflite_output, atol=1e-5)
```

---

## 8. Keras Core: Implementation Details

### 8.1 File Structure

```
keras/src/export/
├── __init__.py              # Public API exports
├── export_utils.py          # Shared utilities (signature inference)
├── litert.py                # LiteRT exporter implementation
├── litert_test.py           # Unit tests
└── saved_model.py           # SavedModel exporter (not LiteRT-related)
```

### 8.2 Key Functions

#### `export_litert()`

Main entry point function:

```python
def export_litert(
    model: keras.Model,
    filepath: str,
    input_signature: Optional[List[tf.TensorSpec]] = None,
    **kwargs
) -> None:
    """Export Keras model to LiteRT format.
    
    This is the main function called by model.export(format='litert').
    """
    exporter = LiteRTExporter()
    exporter.export(model, filepath, input_signature=input_signature, **kwargs)
```

#### `_choose_conversion_strategy()`

Determines which strategy to use:

```python
def _choose_conversion_strategy(model: keras.Model) -> str:
    """Choose conversion strategy based on model architecture.
    
    Returns:
        'direct': Use from_keras_model() (Sequential/Functional)
        'wrapper': Use from_concrete_functions() (Subclassed)
    """
    if isinstance(model, keras.Sequential):
        return 'direct'
    elif isinstance(model, keras.Functional):
        return 'direct'
    else:
        # Subclassed or complex model
        return 'wrapper'
```

#### `_validate_model()`

Pre-export validation:

```python
def _validate_model(model: keras.Model) -> None:
    """Validate model is ready for export.
    
    Raises:
        ValueError: If model is not built or has unsupported components
    """
    if not model.built:
        raise ValueError("Model must be built...")
    
    # Check for known unsupported layers
    for layer in model.layers:
        if layer.__class__.__name__ in UNSUPPORTED_LAYERS:
            raise ValueError(f"Layer '{layer.name}' of type "
                             f"'{layer.__class__.__name__}' is not supported...")
```

### 8.3 Optimization Techniques

1. **Lazy Imports**: TensorFlow modules only loaded when export is called
2. **Caching**: Reuse ConcreteFunction if exporting same model multiple times
3. **Memory Management**: Clear intermediate representations after conversion
4. **Parallel AOT**: Compile for multiple targets in parallel (if supported)

---

# PART III: KERAS-HUB IMPLEMENTATION (PR #2405)

## 9. Keras-Hub: Task Model Export Extension

### 9.1 Overview

Keras-Hub extends Keras Core's export to handle **task-specific models** with 
unique requirements:

| Challenge | Keras-Hub Solution |
|-----------|-------------------|
| **Input Format** | Task models use dict inputs (`{"token_ids": ..., "padding_mask": ...}`) <br> → TextModelAdapter converts dict → list |
| **Model-Specific Params** | CausalLM needs `sequence_length`, ImageClassifier needs `image_size` <br> → Per-model-type config classes |
| **Type Detection** | Need to determine if model is text vs image vs other <br> → Registry pattern maps classes to configs |
| **Extensibility** | Easy to add new model types <br> → Simple registration API |

### 9.2 Entry Point: `Task.export()`

**Location**: `keras_hub/src/export/registry.py`

Keras-Hub **extends** the base `Task` class with export support:

```python
from keras_hub.src.models.task import Task
from keras_hub.src.export.registry import export_model

# Monkey-patch Task class to add export method
def export(self, filepath, format=None, **kwargs):
    """Export task model to various formats.
    
    This delegates to the registry-based export_model() function.
    """
    if format is None:
        # Infer format from filepath extension
        if filepath.endswith('.tflite'):
            format = 'litert'
        else:
            raise ValueError("Cannot infer format from filepath. "
                             "Please specify 'format' explicitly.")
    
    return export_model(self, filepath, format=format, **kwargs)

# Attach to Task class
Task.export = export
```

**Design Note**: Using monkey-patching allows adding export to all task models 
without modifying the base `Task` class definition.

### 9.3 Export Registry

**Location**: `keras_hub/src/export/registry.py`

The registry maps model classes to their exporter configs:

```python
class ExporterRegistry:
    """Central registry mapping model types to exporter configs."""
    
    _registry = {}  # {model_class: exporter_config_class}
    
    @classmethod
    def register(cls, model_class, exporter_config_class):
        """Register a model class with its exporter config.
        
        Args:
            model_class: Task model class (e.g., CausalLM)
            exporter_config_class: Config class (e.g., CausalLMExporterConfig)
        """
        cls._registry[model_class] = exporter_config_class
    
    @classmethod
    def get_config(cls, model):
        """Get exporter config for a model instance.
        
        Args:
            model: Task model instance
        
        Returns:
            Exporter config instance
        
        Raises:
            ValueError: If model class is not registered
        """
        for model_class, config_class in cls._registry.items():
            if isinstance(model, model_class):
                return config_class()
        
        raise ValueError(
            f"No exporter config found for model type {type(model).__name__}. "
            f"Supported types: {list(cls._registry.keys())}"
        )

# Registration happens at module import time
from keras_hub.src.models.causal_lm import CausalLM
from keras_hub.src.models.text_classifier import TextClassifier
from keras_hub.src.export.configs import (
    CausalLMExporterConfig,
    TextClassifierExporterConfig,
)

ExporterRegistry.register(CausalLM, CausalLMExporterConfig)
ExporterRegistry.register(TextClassifier, TextClassifierExporterConfig)
# ... register other model types
```

### 9.4 Export Model Function

**Location**: `keras_hub/src/export/registry.py`

```python
def export_model(model, filepath, format='litert', **kwargs):
    """Export a Keras-Hub task model.
    
    Args:
        model: Keras-Hub task model instance
        filepath: Destination file path
        format: Export format ('litert' only for now)
        **kwargs: Format-specific kwargs
    
    Raises:
        ValueError: If format is unsupported or model type is unsupported
    """
    if format != 'litert':
        raise ValueError(f"Unsupported format '{format}'. "
                         f"Currently only 'litert' is supported.")
    
    # Get config for this model type
    config = ExporterRegistry.get_config(model)
    
    # Get LiteRT exporter
    from keras_hub.src.export.litert import LiteRTExporter
    exporter = LiteRTExporter(config)
    
    # Perform export
    return exporter.export(model, filepath, **kwargs.get('litert_kwargs', {}))
```

---

## 10. Keras-Hub: Exporter Configurations

### 10.1 Base Config Class

**Location**: `keras_hub/src/export/base.py`

```python
class KerasHubExporterConfig(ABC):
    """Abstract base class for task-specific exporter configs.
    
    Each task type (CausalLM, TextClassifier, etc.) has a config subclass
    that defines:
    1. Which parameters are required for export
    2. Default values for those parameters
    3. Validation logic
    """
    
    @abstractmethod
    def get_export_param(self, model, **kwargs):
        """Get task-specific export parameters.
        
        Args:
            model: Task model instance
            **kwargs: User-provided overrides
        
        Returns:
            Dict of export parameters (e.g., {"sequence_length": 128})
        """
        pass
    
    @abstractmethod
    def validate(self, model):
        """Validate that model is compatible with this config.
        
        Args:
            model: Task model instance
        
        Raises:
            ValueError: If model is incompatible
        """
        pass
```

### 10.2 Text Model Configs

#### CausalLMExporterConfig

```python
class CausalLMExporterConfig(KerasHubExporterConfig):
    """Configuration for exporting CausalLM (text generation) models.
    
    Required params:
        - sequence_length: Max input/output sequence length
    """
    
    def get_export_param(self, model, **kwargs):
        """Get sequence_length for causal LM export.
        
        Priority:
            1. User-provided kwargs['sequence_length']
            2. Model's preprocessor.sequence_length
            3. Default: 128
        """
        if 'sequence_length' in kwargs:
            return {'sequence_length': kwargs['sequence_length']}
        
        if hasattr(model, 'preprocessor') and hasattr(model.preprocessor, 'sequence_length'):
            return {'sequence_length': model.preprocessor.sequence_length}
        
        return {'sequence_length': 128}  # Default
    
    def validate(self, model):
        """Validate model is a CausalLM."""
        from keras_hub.src.models.causal_lm import CausalLM
        if not isinstance(model, CausalLM):
            raise ValueError(f"Expected CausalLM, got {type(model).__name__}")
```

#### TextClassifierExporterConfig

```python
class TextClassifierExporterConfig(KerasHubExporterConfig):
    """Configuration for exporting text classifier models.
    
    Required params:
        - sequence_length: Max input sequence length
    """
    
    def get_export_param(self, model, **kwargs):
        """Get sequence_length for text classifier export."""
        if 'sequence_length' in kwargs:
            return {'sequence_length': kwargs['sequence_length']}
        
        if hasattr(model, 'preprocessor') and hasattr(model.preprocessor, 'sequence_length'):
            return {'sequence_length': model.preprocessor.sequence_length}
        
        return {'sequence_length': 128}
    
    def validate(self, model):
        """Validate model is a TextClassifier."""
        from keras_hub.src.models.text_classifier import TextClassifier
        if not isinstance(model, TextClassifier):
            raise ValueError(f"Expected TextClassifier, got {type(model).__name__}")
```

#### Seq2SeqLMExporterConfig

```python
class Seq2SeqLMExporterConfig(KerasHubExporterConfig):
    """Configuration for sequence-to-sequence models.
    
    Required params:
        - encoder_sequence_length: Max encoder input length
        - decoder_sequence_length: Max decoder input length
    """
    
    def get_export_param(self, model, **kwargs):
        """Get encoder/decoder sequence lengths."""
        params = {}
        
        # Encoder length
        if 'encoder_sequence_length' in kwargs:
            params['encoder_sequence_length'] = kwargs['encoder_sequence_length']
        elif hasattr(model, 'preprocessor') and hasattr(model.preprocessor, 'encoder_sequence_length'):
            params['encoder_sequence_length'] = model.preprocessor.encoder_sequence_length
        else:
            params['encoder_sequence_length'] = 128
        
        # Decoder length
        if 'decoder_sequence_length' in kwargs:
            params['decoder_sequence_length'] = kwargs['decoder_sequence_length']
        elif hasattr(model, 'preprocessor') and hasattr(model.preprocessor, 'decoder_sequence_length'):
            params['decoder_sequence_length'] = model.preprocessor.decoder_sequence_length
        else:
            params['decoder_sequence_length'] = 128
        
        return params
```

### 10.3 Image Model Configs

#### ImageClassifierExporterConfig

```python
class ImageClassifierExporterConfig(KerasHubExporterConfig):
    """Configuration for image classifier models.
    
    Required params:
        - image_size: Input image shape (H, W, C)
    """
    
    def get_export_param(self, model, **kwargs):
        """Get image_size for image classifier export.
        
        Priority:
            1. User-provided kwargs['image_size']
            2. Model's preprocessor.image_size
            3. Model's backbone input shape
            4. Default: (224, 224, 3)
        """
        if 'image_size' in kwargs:
            return {'image_size': kwargs['image_size']}
        
        if hasattr(model, 'preprocessor') and hasattr(model.preprocessor, 'image_size'):
            size = model.preprocessor.image_size
            if isinstance(size, (list, tuple)) and len(size) == 2:
                return {'image_size': (*size, 3)}  # Add channel dim
            return {'image_size': size}
        
        if hasattr(model, 'backbone') and hasattr(model.backbone, 'input_shape'):
            return {'image_size': model.backbone.input_shape[1:]}
        
        return {'image_size': (224, 224, 3)}
    
    def validate(self, model):
        """Validate model is an ImageClassifier."""
        from keras_hub.src.models.image_classifier import ImageClassifier
        if not isinstance(model, ImageClassifier):
            raise ValueError(f"Expected ImageClassifier, got {type(model).__name__}")
```

#### ObjectDetectorExporterConfig

```python
class ObjectDetectorExporterConfig(KerasHubExporterConfig):
    """Configuration for object detection models."""
    
    def get_export_param(self, model, **kwargs):
        """Get image_size for object detector export."""
        # Similar to ImageClassifierExporterConfig
        if 'image_size' in kwargs:
            return {'image_size': kwargs['image_size']}
        # ... (same priority order)
        return {'image_size': (512, 512, 3)}  # Larger default for detection
```

#### ImageSegmenterExporterConfig

```python
class ImageSegmenterExporterConfig(KerasHubExporterConfig):
    """Configuration for image segmentation models."""
    
    def get_export_param(self, model, **kwargs):
        """Get image_size for segmentation export."""
        if 'image_size' in kwargs:
            return {'image_size': kwargs['image_size']}
        # ... (same priority order)
        return {'image_size': (256, 256, 3)}  # Default for segmentation
```

---

## 11. Keras-Hub: Model Adapters

### 11.1 Adapter Purpose

**Problem**: TFLite converter works best with **positional (list) inputs**, 
but Keras-Hub task models use **named (dict) inputs**.

```python
# Keras-Hub model signature
def call(self, inputs):
    # inputs = {"token_ids": [B, L], "padding_mask": [B, L]}
    token_ids = inputs["token_ids"]
    padding_mask = inputs["padding_mask"]
    # ...

# TFLite converter prefers
def call(self, inputs):
    # inputs = [[B, L], [B, L]]  (list of tensors)
    token_ids = inputs[0]
    padding_mask = inputs[1]
    # ...
```

**Solution**: Wrap model in an adapter that transforms input format.

### 11.2 TextModelAdapter

**Location**: `keras_hub/src/export/litert.py`

```python
class TextModelAdapter(keras.Model):
    """Adapter for text models: converts list inputs → dict inputs.
    
    Text models expect dict with keys like:
        - "token_ids"
        - "padding_mask"
        - "segment_ids" (for some models)
    
    This adapter accepts a list and converts to dict.
    """
    
    def __init__(self, inner_model, sequence_length, **kwargs):
        super().__init__(**kwargs)
        self.inner_model = inner_model
        self.sequence_length = sequence_length
    
    def call(self, inputs):
        """Convert list → dict and call inner model.
        
        Args:
            inputs: List of tensors [token_ids, padding_mask, ...]
        
        Returns:
            Model output (logits, etc.)
        """
        # Build dict from list
        model_inputs = {
            "token_ids": inputs[0],
            "padding_mask": inputs[1] if len(inputs) > 1 else None,
        }
        
        # Remove None values
        model_inputs = {k: v for k, v in model_inputs.items() if v is not None}
        
        # Call inner model
        return self.inner_model(model_inputs)
    
    def get_config(self):
        return {
            "sequence_length": self.sequence_length,
        }
```

### 11.3 ImageModelAdapter

**Location**: `keras_hub/src/export/litert.py`

```python
class ImageModelAdapter(keras.Model):
    """Adapter for image models: handles both dict and tensor inputs.
    
    Image models can accept:
        - Dict: {"images": [B, H, W, C]}
        - Tensor: [B, H, W, C]
    
    This adapter normalizes to dict format.
    """
    
    def __init__(self, inner_model, image_size, **kwargs):
        super().__init__(**kwargs)
        self.inner_model = inner_model
        self.image_size = image_size
    
    def call(self, inputs):
        """Convert tensor/list → dict and call inner model.
        
        Args:
            inputs: Tensor [B, H, W, C] or list [[B, H, W, C]]
        
        Returns:
            Model output (class logits, bboxes, masks, etc.)
        """
        # If list with single element, extract tensor
        if isinstance(inputs, list) and len(inputs) == 1:
            inputs = inputs[0]
        
        # If already a dict, pass through
        if isinstance(inputs, dict):
            return self.inner_model(inputs)
        
        # Convert tensor to dict
        model_inputs = {"images": inputs}
        return self.inner_model(model_inputs)
    
    def get_config(self):
        return {
            "image_size": self.image_size,
        }
```

### 11.4 BaseModelAdapter (Fallback)

**Note**: As of the latest implementation, **BaseModelAdapter is removed**. 
The system now raises an explicit error for unsupported model types instead 
of returning a generic fallback.

```python
# OLD (removed):
class BaseModelAdapter(keras.Model):
    # Generic pass-through adapter
    pass

# NEW (current):
def _get_model_adapter_class(model):
    """Determine adapter class for model.
    
    Raises:
        ValueError: If model type is not supported
    """
    if isinstance(model, (CausalLM, TextClassifier, Seq2SeqLM)):
        return "text"
    elif isinstance(model, (ImageClassifier, ObjectDetector, ImageSegmenter)):
        return "image"
    else:
        supported_types = [
            "text_classifier", "causal_lm", "seq2seq_lm",
            "image_classifier", "object_detector", "image_segmenter"
        ]
        raise ValueError(
            f"Unsupported model type '{type(model).__name__}'. "
            f"Supported types: {', '.join(supported_types)}"
        )
```

**Rationale**: Failing explicitly is better than silently using a generic 
adapter that may produce incorrect results.

---

## 12. Keras-Hub: LiteRT Exporter

### 12.1 Exporter Class

**Location**: `keras_hub/src/export/litert.py`

```python
class LiteRTExporter(KerasHubExporter):
    """Keras-Hub LiteRT exporter that delegates to Keras Core.
    
    Responsibilities:
        1. Determine model adapter type (text/image)
        2. Extract task-specific export params
        3. Create adapter wrapper
        4. Delegate to Keras Core's LiteRTExporter
    """
    
    def __init__(self, config: KerasHubExporterConfig):
        self.config = config
    
    def export(self, model, filepath, **kwargs):
        """Export task model to LiteRT format.
        
        Args:
            model: Keras-Hub task model
            filepath: Destination .tflite path
            **kwargs: User overrides for export params
        
        Process:
            1. Validate model
            2. Get export params (sequence_length, image_size, etc.)
            3. Determine adapter type
            4. Create adapter wrapper
            5. Call Keras Core exporter
        """
        # Step 1: Validate
        self.config.validate(model)
        
        # Step 2: Get params
        export_params = self.config.get_export_param(model, **kwargs)
        
        # Step 3: Determine adapter type
        adapter_type = self._get_model_adapter_class(model)
        
        # Step 4: Create adapter
        if adapter_type == "text":
            wrapped_model = TextModelAdapter(
                model,
                sequence_length=export_params['sequence_length']
            )
        elif adapter_type == "image":
            wrapped_model = ImageModelAdapter(
                model,
                image_size=export_params['image_size']
            )
        else:
            raise ValueError(f"Unknown adapter type: {adapter_type}")
        
        # Step 5: Delegate to Keras Core
        from keras.src.export.litert import export_litert as keras_export_litert
        return keras_export_litert(
            model=wrapped_model,
            filepath=filepath,
            **kwargs  # Pass through any additional Keras Core params
        )
```

### 12.2 Type Detection

```python
def _get_model_adapter_class(self, model):
    """Determine which adapter type to use.
    
    Returns:
        "text": For CausalLM, TextClassifier, Seq2SeqLM
        "image": For ImageClassifier, ObjectDetector, ImageSegmenter
    
    Raises:
        ValueError: For unsupported model types (audio, multimodal, custom)
    """
    from keras_hub.src.models.causal_lm import CausalLM
    from keras_hub.src.models.text_classifier import TextClassifier
    from keras_hub.src.models.seq2seq_lm import Seq2SeqLM
    from keras_hub.src.models.image_classifier import ImageClassifier
    from keras_hub.src.models.object_detector import ObjectDetector
    from keras_hub.src.models.image_segmenter import ImageSegmenter
    
    # Text models
    if isinstance(model, (CausalLM, TextClassifier, Seq2SeqLM)):
        return "text"
    
    # Image models
    elif isinstance(model, (ImageClassifier, ObjectDetector, ImageSegmenter)):
        return "image"
    
    # Unsupported (audio, multimodal, custom models not yet implemented)
    else:
        supported = [
            "text_classifier", "causal_lm", "seq2seq_lm",
            "image_classifier", "object_detector", "image_segmenter"
        ]
        raise ValueError(
            f"Unsupported model type '{type(model).__name__}'. "
            f"Supported types: {', '.join(supported)}.\n"
            f"Audio and multimodal models will be supported in future releases."
        )
```

### 12.3 Parameter Extraction

```python
def _get_export_param(self, model, **kwargs):
    """Get task-specific export parameters.
    
    Delegates to config class's get_export_param() method.
    
    Returns:
        Dict with params like:
            - {"sequence_length": 128} for text models
            - {"image_size": (224, 224, 3)} for image models
            - {"encoder_sequence_length": 512, "decoder_sequence_length": 128} 
              for seq2seq
    """
    return self.config.get_export_param(model, **kwargs)
```

---

## 13. Keras-Hub: Testing

### 13.1 Test Structure

Keras-Hub has **5 test files** covering different model types:

```
keras_hub/src/export/
├── causal_lm_export_test.py       # CausalLM export tests
├── image_classifier_export_test.py # ImageClassifier export tests
├── object_detector_export_test.py  # ObjectDetector export tests
├── seq2seq_lm_export_test.py      # Seq2SeqLM export tests
└── text_classifier_export_test.py  # TextClassifier export tests
```

### 13.2 Test Philosophy

Each test file follows the **same pattern**:

1. **Mock Model Tests**: Fast, test export logic without large models
2. **Production Model Tests**: Slower, test real preset models (BERT, GPT, ResNet, etc.)
3. **Error Handling Tests**: Verify proper error messages

### 13.3 Example: CausalLM Tests

**Location**: `keras_hub/src/export/causal_lm_export_test.py`

```python
class CausalLMExportTest(tf.test.TestCase):
    """Tests for CausalLM export."""
    
    def test_export_with_mock_model(self):
        """Test export using a mock CausalLM model."""
        # Create lightweight mock model
        mock_model = self._create_mock_causal_lm()
        
        # Export
        filepath = self.get_temp_dir() + '/model.tflite'
        mock_model.export(filepath, format='litert')
        
        # Verify file exists
        self.assertTrue(os.path.exists(filepath))
        
        # Verify TFLite model loads
        interpreter = tf.lite.Interpreter(model_path=filepath)
        interpreter.allocate_tensors()
        
        # Verify input/output shapes
        input_details = interpreter.get_input_details()
        self.assertEqual(len(input_details), 2)  # token_ids, padding_mask
    
    def test_export_with_custom_sequence_length(self):
        """Test export with user-provided sequence_length."""
        mock_model = self._create_mock_causal_lm()
        
        filepath = self.get_temp_dir() + '/model.tflite'
        mock_model.export(
            filepath,
            format='litert',
            litert_kwargs={'sequence_length': 256}
        )
        
        # Verify exported model has correct input shape
        interpreter = tf.lite.Interpreter(model_path=filepath)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        
        # First input should be [batch, 256]
        self.assertEqual(input_details[0]['shape'][1], 256)
    
    @pytest.mark.large
    def test_export_gpt2(self):
        """Test export with real GPT-2 model (slow)."""
        model = keras_hub.models.GPT2CausalLM.from_preset("gpt2_base_en")
        
        filepath = self.get_temp_dir() + '/gpt2.tflite'
        model.export(filepath, format='litert')
        
        # Test numerical equivalence
        test_input = {"token_ids": [[1, 2, 3, 4, 5]], "padding_mask": [[1, 1, 1, 1, 1]]}
        keras_output = model(test_input)
        
        # Load TFLite and run inference
        interpreter = tf.lite.Interpreter(model_path=filepath)
        interpreter.allocate_tensors()
        # ... (TFLite inference)
        
        # Assert outputs match
        self.assertAllClose(keras_output, tflite_output, atol=1e-4)
```

### 13.4 Mock Model Creation

To keep tests fast, each test file has a `_create_mock_*()` helper:

```python
def _create_mock_causal_lm(self):
    """Create a minimal CausalLM for testing."""
    # Create simple backbone
    backbone = keras.Sequential([
        keras.layers.Embedding(1000, 64),  # Small vocab, small dim
        keras.layers.Dense(64),
    ])
    
    # Create mock preprocessor
    preprocessor = MockTokenizer(sequence_length=128)
    
    # Create CausalLM
    model = CausalLM(
        backbone=backbone,
        preprocessor=preprocessor,
    )
    
    # Build model
    model.build({"token_ids": (None, 128), "padding_mask": (None, 128)})
    
    return model
```

### 13.5 Test Results

**Current status** (as of last run):
- ✅ 42 tests passed
- ⏭️ 27 tests skipped (production model tests, require large downloads)
- ❌ 0 tests failed

**Skipped tests** are legitimate (production models require network access 
and large disk space). They can be run with:
```bash
pytest keras_hub/src/export/ --run-large
```

---

# PART IV: IMPLEMENTATION DETAILS

## 14. Complete Export Pipeline

### 14.1 End-to-End Flow

The complete export pipeline integrates both Keras Core and Keras-Hub layers:

```
User Code: model.export('model.tflite', format='litert')
    │
    ├─→ Is model a Keras-Hub Task model?
    │       │
    │       ├─ YES: Keras-Hub Flow
    │       │   │
    │       │   ├─→ [ExporterRegistry.get_config(model)]
    │       │   │     Returns: CausalLMExporterConfig / ImageClassifierExporterConfig / etc.
    │       │   │
    │       │   ├─→ [Config.get_export_param(model, **kwargs)]
    │       │   │     Extracts: sequence_length / image_size / etc.
    │       │   │
    │       │   ├─→ [LiteRTExporter._get_model_adapter_class(model)]
    │       │   │     Determines: "text" or "image"
    │       │   │
    │       │   ├─→ [Create Adapter: TextModelAdapter / ImageModelAdapter]
    │       │   │     Wraps model to convert dict inputs → list inputs
    │       │   │
    │       │   └─→ [Call Keras Core export_litert(wrapped_model)]
    │       │
    │       └─ NO: Keras Core Flow (Direct)
    │           │
    │           └─→ [Keras Core export_litert(model)]
    │
    ├─→ [Keras Core: Ensure model is built]
    │
    ├─→ [Keras Core: Infer or validate input signature]
    │
    ├─→ [Keras Core: Choose conversion strategy]
    │     ├─ Direct: from_keras_model() for Sequential/Functional
    │     └─ Wrapper: tf.Module + from_concrete_functions() for Subclassed
    │
    ├─→ [TFLite Converter: Convert to .tflite format]
    │
    ├─→ [Optional: AOT compilation for hardware targets]
    │
    └─→ [Write .tflite file to disk]
```

### 14.2 Backend Support

**Current Implementation**: Models must use TensorFlow backend during export

While Keras 3 supports multiple backends (TensorFlow, JAX, PyTorch), the LiteRT export currently requires the TensorFlow backend for conversion. This is because:
- The export process uses `tf.lite.TFLiteConverter` under the hood
- Tensor conversion is handled transparently by the exporter
- Models trained on other backends can be exported by reloading with TensorFlow backend

**Alternative Paths for Other Backends**:
- **JAX models**: Can export via [JAX to LiteRT conversion](https://ai.google.dev/edge/litert/models/convert_jax) using similar approach to Keras JAX backend export
- **PyTorch models**: Can export directly via [PyTorch to LiteRT](https://ai.google.dev/edge/litert/models/convert_pytorch)

```python
def export_litert(
    model,
    filepath,
    verbose=True,
    input_signature=None,
    aot_compile_targets=None,
    **kwargs,
):
    """Export the model as a LiteRT artifact for inference.

    Note: Requires TensorFlow backend. If your model uses JAX or PyTorch backend,
    consider using the direct conversion paths mentioned above, or reload the model
    with TensorFlow backend before export.
    
    Args:
        model: The Keras model to export.
        filepath: The path to save the exported artifact (.tflite file).
        verbose: `bool`. Whether to print progress messages during export.
        input_signature: Optional input signature specification. If `None`,
            it will be automatically inferred from the model.
        aot_compile_targets: Optional list of LiteRT targets for AOT
            compilation (e.g., ["qualcomm", "mediatek"]).
        **kwargs: Additional keyword arguments passed to TFLiteConverter.
    
    Returns:
        None. The model is saved to the specified filepath.
    
    Raises:
        RuntimeError: If TensorFlow is not available.
        ValueError: If model cannot be converted or built.
    
    Example:
    ```python
    from keras.src.export import export_litert
    
    # Direct usage (normally called via model.export)
    export_litert(model, "model.tflite", verbose=True)
    ```
    """
```
```

### LiteRTExporter Class

The core conversion logic is implemented in the `LiteRTExporter` class:

```python
class LiteRTExporter:
    """Exporter for the LiteRT (TFLite) format.

    This class handles the conversion of Keras models for LiteRT runtime and
    generates a `.tflite` model file. It supports:
    - Sequential, Functional, and Subclassed models
    - Multi-input and multi-output models
    - Stateful layers (LSTM, BatchNormalization)
    - Optional AOT compilation for specific hardware targets
    
    The class uses a multi-strategy approach:
    1. Direct conversion via TFLiteConverter.from_keras_model()
    2. Fallback to tf.Module wrapper for complex models
    3. Multiple resource variable handling strategies
    """
    
    def __init__(
        self,
        model,
        input_signature=None,
        verbose=False,
        aot_compile_targets=None,
        **kwargs,
    ):
        """Initialize the LiteRT exporter.

        Args:
            model: The Keras model to export
            input_signature: Input signature specification (auto-inferred if None)
            verbose: Whether to print progress messages during export
            aot_compile_targets: List of LiteRT targets for AOT compilation
            **kwargs: Additional export parameters for TFLiteConverter
        """


### Input Signature Handling

The implementation uses Keras's `InputSpec` for type-safe input specification:

```python
# From keras.src.export.export_utils

def get_input_signature(model):
    """Get input signature for model export.
    
    Handles three model types differently:
    1. Functional: Returns single-element list wrapping the nested input structure
    2. Sequential: Maps structure of model.inputs
    3. Subclassed: Infers from recorded build shapes
    
    Returns:
        Input signature suitable for model export (always a tuple or list).
    """

def make_input_spec(x):
    """Convert various input types to keras.layers.InputSpec.
    
    Supports:
    - layers.InputSpec (pass-through with validation)
    - backend.KerasTensor (extract shape and dtype)
    - backend tensors (extract shape and dtype)
    
    Always sets batch dimension to None for flexibility.
    """

def make_tf_tensor_spec(x):
    """Convert InputSpec to tf.TensorSpec for TFLite conversion."""
```

### Converter Configuration

The implementation uses specific TFLite converter settings:

```python
converter = tf.lite.TFLiteConverter.from_keras_model(model)
# OR
converter = tf.lite.TFLiteConverter.from_concrete_functions(
    [concrete_func], trackable_obj=wrapper
)

# Common configuration
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,  # Standard ops
    tf.lite.OpsSet.SELECT_TF_OPS,     # TF ops fallback for unsupported ops
]

# Resource variable handling (tries both)
converter.experimental_enable_resource_variables = False  # Try first
converter.experimental_enable_resource_variables = True   # Fallback
```


class OpsSet:
    """Operation sets supported by target devices.
    
    Attributes:
        TFLITE_BUILTINS: Standard TFLite operations. Supported on all devices.
        TFLITE_BUILTINS_INT8: Quantized operations using int8. Requires
            representative dataset for calibration.
        SELECT_TF_OPS: Include select TensorFlow operations. Increases model
            size and may not be supported on all platforms.
        EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8: 16-bit
            activations with 8-bit weights for better accuracy.
    
    Examples:
    >>> # Standard operations only
    >>> target = litert.TargetSpec(
    ...     supported_ops=[litert.OpsSet.TFLITE_BUILTINS]
    ... )
    >>> 
    >>> # Quantized model
    >>> target = litert.TargetSpec(
    ...     supported_ops=[litert.OpsSet.TFLITE_BUILTINS_INT8]
    ... )
    >>> 
    >>> # Fallback to TF ops when needed
    >>> target = litert.TargetSpec(
    ...     supported_ops=[
    ...         litert.OpsSet.TFLITE_BUILTINS,
    ...         litert.OpsSet.SELECT_TF_OPS
    ...     ]
    ... )
    """
    TFLITE_BUILTINS = "TFLITE_BUILTINS"
    TFLITE_BUILTINS_INT8 = "TFLITE_BUILTINS_INT8"
    SELECT_TF_OPS = "SELECT_TF_OPS"
    EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8 = (
        "EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8"
    )


class TargetSpec:
    """Specification for target device constraints.
    
    Args:
        supported_ops: List of `OpsSet`. Operation sets available on target
            device. Defaults to `[OpsSet.TFLITE_BUILTINS]`.
        supported_types: Optional list of str. Data types supported by target.
            If None, uses default types for the operation set.
        experimental_supported_backends: Optional list of str. Hardware
            backends to target (e.g., 'GPU', 'DSP', 'NPU').
    
    Examples:
    >>> # CPU with standard ops
    >>> target = litert.TargetSpec(
    ...     supported_ops=[litert.OpsSet.TFLITE_BUILTINS]
    ... )
    >>> 
    >>> # Quantized model for edge TPU
    >>> target = litert.TargetSpec(
    ...     supported_ops=[litert.OpsSet.TFLITE_BUILTINS_INT8],
    ...     experimental_supported_backends=['EDGE_TPU']
    ... )
    """
    
    def __init__(
        self,
        supported_ops=None,
        supported_types=None,
        experimental_supported_backends=None,
    ):
        self.supported_ops = supported_ops or [OpsSet.TFLITE_BUILTINS]
        self.supported_types = supported_types
        self.experimental_supported_backends = experimental_supported_backends


class LiteRTModel:
    """Converted model ready for inference or serialization.
    
    This class wraps a converted LiteRT model and provides methods for
    inference, inspection, and serialization.
    
    Attributes:
        input_details: List of dicts describing model inputs.
        output_details: List of dicts describing model outputs.
        
    Examples:
    >>> model = litert.export(keras_model)
    >>> 
    >>> # Inspect model
    >>> print(model.input_details)
    >>> print(model.output_details)
    >>> 
    >>> # Run inference
    >>> outputs = model.predict(inputs)
    >>> 
    >>> # Save to file
    >>> model.save('model.tflite')
    """
    
    def __init__(self, flatbuffer_data):
        """Initialize from flatbuffer data.
        
        Args:
            flatbuffer_data: bytes. Serialized flatbuffer model.
        """
        self._flatbuffer_data = flatbuffer_data
        self._interpreter = None
    
    @property
    def input_details(self):
        """List of dicts describing model inputs.
        
        Returns:
            List of dicts with keys: 'name', 'shape', 'dtype', 'index'.
        """
        self._ensure_interpreter()
        return self._interpreter.get_input_details()
    
    @property
    def output_details(self):
        """List of dicts describing model outputs.
        
        Returns:
            List of dicts with keys: 'name', 'shape', 'dtype', 'index'.
        """
        self._ensure_interpreter()
        return self._interpreter.get_output_details()
    
    def predict(self, inputs, **kwargs):
        """Run inference on input data.
        
        Args:
            inputs: Input tensor(s). Can be:
                - Single numpy array for single-input models
                - List of numpy arrays for multi-input models
                - Dict mapping input names to numpy arrays
            **kwargs: Additional inference options (e.g., num_threads).
        
        Returns:
            Model outputs. Format matches input format:
                - Single array for single-output models
                - List of arrays for multi-output models
        
        Examples:
        >>> # Single input/output
        >>> output = model.predict(input_array)
        >>> 
        >>> # Multiple inputs
        >>> outputs = model.predict([input1, input2])
        >>> 
        >>> # Named inputs
        >>> outputs = model.predict({
        ...     'input_ids': ids,
        ...     'attention_mask': mask
        ... })
        """
        self._ensure_interpreter()
        # Implementation details
        pass
    
    def save(self, filepath):
        """Save model to file.
        
        Args:
            filepath: str. Path where model will be saved. Should end with
                '.tflite' extension.
        
        Examples:
        >>> model.save('model.tflite')
        >>> model.save('/path/to/model.tflite')
        """
        with open(filepath, 'wb') as f:
            f.write(self._flatbuffer_data)
    
    def get_signature_list(self):
        """Get list of available signatures in the model.
        
        Returns:
            List of str. Signature names.
        """
        # Implementation details
        pass
    
    def _ensure_interpreter(self):
        """Lazily initialize interpreter."""
        if self._interpreter is None:
            # Implementation details
            pass
```

## Conversion Pipeline

### Architecture

The actual implementation follows this flow:

```
┌─────────────────────────┐
│   Keras Model           │
│ (Sequential/Functional/ │
│    Subclassed)          │
└──────────┬──────────────┘
           │
           v
┌─────────────────────────┐
│  Ensure Model Built     │
│  (auto-build if needed) │
└──────────┬──────────────┘
           │
           v
┌─────────────────────────┐
│  Infer Input Signature  │
│  (if not provided)      │
└──────────┬──────────────┘
           │
           v
┌─────────────────────────┐
│  Try Direct Conversion  │
│  (from_keras_model)     │
└──────────┬──────────────┘
           │
           ├─ Success ────────────┐
           │                      │
           └─ Failure             │
                │                 │
                v                 │
   ┌─────────────────────────┐   │
   │  Wrapper-Based          │   │
   │  Conversion             │   │
   │  (tf.Module wrapper)    │   │
   └──────────┬──────────────┘   │
              │                  │
              v                  │
   ┌─────────────────────────┐   │
   │  Get Concrete Function  │   │
   │  with Input Signature   │   │
   └──────────┬──────────────┘   │
              │                  │
              v                  │
   ┌─────────────────────────┐   │
   │  Try Conversion         │   │
   │  Strategies (w/ & w/o   │   │
   │  resource vars)         │   │
   └──────────┬──────────────┘   │
              │                  │
              └──────────────────┤
                                 v
                    ┌─────────────────────────┐
                    │  TFLite Flatbuffer      │
                    │  (.tflite file)         │
                    └──────────┬──────────────┘
                               │
                               v
                    ┌─────────────────────────┐
                    │  Optional AOT           │
                    │  Compilation            │
                    │  (if targets specified) │
                    └─────────────────────────┘
```

### Conversion Strategies

The implementation uses two main conversion strategies:

#### 1. Direct Conversion (Preferred)

```python
def _convert_to_tflite(self, input_signature):
    """Try direct conversion first for Sequential and Functional models."""
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS,
        ]
        converter.experimental_enable_resource_variables = False
        return converter.convert()
    except Exception as direct_error:
        # Fall back to wrapper-based conversion
        return self._convert_with_wrapper(input_signature)
```

#### 2. Wrapper-Based Conversion (Fallback)

```python
def _convert_with_wrapper(self, input_signature):
    """Use tf.Module wrapper for complex models."""
    
    class KerasModelWrapper(tf.Module):
        """Clean tf.Module wrapper for TFLite conversion.
        
        Key design decisions:
        1. Store model without TF tracking (via object.__setattr__)
        2. Manually track all model variables
        3. Single @tf.function decorated __call__ method
        4. Handle multi-input/output cases explicitly
        """
        
        def __init__(self, model):
            super().__init__()
            # Avoid TF tracking to prevent _DictWrapper errors
            object.__setattr__(self, "_model", model)
            
            # Track variables for proper SavedModel serialization
            with self.name_scope:
                for i, var in enumerate(model.variables):
                    setattr(self, f"model_var_{i}", var)
        
        @tf.function
        def __call__(self, *args, **kwargs):
            """Entry point supporting multiple input formats."""
            # Handle positional, keyword, and mixed arguments
            # Support multi-input functional models
            # ...implementation details...
```

### Variable Tracking

Critical for stateful layers (BatchNorm, LSTM):

```python
# Inside KerasModelWrapper.__init__
with self.name_scope:
    for i, var in enumerate(model.variables):
        # Use unique attribute names to avoid tf.Module property conflicts
        setattr(self, f"model_var_{i}", var)
```

This ensures:
- All model variables are included in the SavedModel
- Proper variable initialization in TFLite runtime
- Correct behavior for stateful layers

## Error Handling

### Error Categories and Handling

#### 1. Backend Availability Check

```python
# In model.export()
if format == "litert" and export_litert is None:
    raise RuntimeError(
        "LiteRT export requires TensorFlow to be installed. "
        "Install it via `pip install tensorflow`."
    )
```

#### 2. Model Build State

```python
def _ensure_model_built(self):
    """Ensures the model is built before conversion."""
    if self.model.built:
        return
    
    try:
        # Try various build strategies
        if self.input_signature:
            input_shapes = tree.map_structure(
                lambda spec: spec.shape, self.input_signature
            )
            self.model.build(input_shapes)
        elif hasattr(self.model, "inputs") and self.model.inputs:
            # ... build from model.inputs
        else:
            raise ValueError(
                "Cannot export model to the litert format as the "
                "input_signature could not be inferred. Either pass an "
                "`input_signature` to `model.export()` or ensure that the "
                "model is already built (called once on real inputs)."
            )
    except Exception as e:
        raise ValueError(
            f"Failed to build model: {e}. Please ensure the model is "
            "properly defined or provide an input_signature."
        )
```

#### 3. Input Signature Validation

```python
# In get_input_signature()
if not model.built:
    raise ValueError(
        "The model provided has not yet been built. It must be built "
        "before export."
    )

# For subclassed models
if not input_signature or not model._called:
    raise ValueError(
        "The model provided has never called. "
        "It must be called at least once before export."
    )
```

#### 4. Conversion Failures

```python
# Multi-strategy approach with detailed logging
conversion_strategies = [
    {"experimental_enable_resource_variables": False, 
     "name": "without resource variables"},
    {"experimental_enable_resource_variables": True, 
     "name": "with resource variables"},
]

for strategy in conversion_strategies:
    try:
        # ... conversion attempt
        return tflite_model
    except Exception as e:
        if self.verbose:
            io_utils.print_msg(
                f"Conversion failed {strategy['name']}: {e}"
            )
        continue

# If all fail
raise RuntimeError(
    "All conversion strategies failed for wrapper-based conversion"
)
```

#### 5. File Path Validation

```python
assert filepath.endswith(".tflite"), (
    "The LiteRT export requires the filepath to end with '.tflite'. "
    f"Got: {filepath}"
)
```

## Testing Strategy

## Testing Strategy

### Test Infrastructure

The implementation includes comprehensive tests in `litert_test.py`:

#### Backend and Interpreter Setup

```python
# Conditional test execution
@pytest.mark.skipif(
    backend.backend() != "tensorflow",
    reason="`export_litert` currently supports the TensorFlow backend only.",
)
class ExportLitertTest(testing.TestCase):
    """Test suite for LiteRT (TFLite) model export functionality."""

# Interpreter selection with fallback
AI_EDGE_LITERT_AVAILABLE = False
if backend.backend() == "tensorflow":
    if litert.available:
        try:
            from ai_edge_litert.interpreter import Interpreter as LiteRTInterpreter
            AI_EDGE_LITERT_AVAILABLE = True
        except (ImportError, OSError):
            LiteRTInterpreter = tensorflow.lite.Interpreter
    else:
        LiteRTInterpreter = tensorflow.lite.Interpreter
```

### Test Coverage

#### 1. Model Architecture Tests

```python
# Parameterized tests for different model types
model_types = ["sequential", "functional"]
if AI_EDGE_LITERT_AVAILABLE:
    model_types.append("lstm")  # LSTM requires AI Edge LiteRT

@parameterized.named_parameters(named_product(model_type=model_types))
def test_standard_model_export(self, model_type):
    """Test exporting standard model types to LiteRT format."""
    model = get_model(model_type)
    ref_input = np.random.normal(size=(1, 10)).astype("float32")
    ref_output = _convert_to_numpy(model(ref_input))
    
    # Export and validate
    model.export(temp_filepath, format="litert")
    self.assertTrue(os.path.exists(temp_filepath))
    
    # Test numerical equivalence
    interpreter = LiteRTInterpreter(temp_filepath)
    interpreter.allocate_tensors()
    # ... run inference and compare outputs
```

#### 2. Multi-Input/Output Tests

```python
def test_multi_input_model_export(self):
    """Test exporting models with multiple inputs."""
    model = get_model("multi_input")
    input1 = np.random.normal(size=(1, 10)).astype("float32")
    input2 = np.random.normal(size=(1, 10)).astype("float32")
    ref_output = model([input1, input2])
    
    # Export and validate with named inputs
    model.export(filepath, format="litert")
    # ... validate inference with input dictionaries

def test_multi_output_model_export(self):
    """Test exporting models with multiple outputs."""
    # Similar approach for multi-output models
```

#### 3. Numerical Validation

```python
def _convert_to_numpy(structure):
    """Convert outputs to numpy for comparison."""
    return tree.map_structure(
        lambda x: x.numpy() if hasattr(x, "numpy") else np.array(x), 
        structure
    )

# In tests
np.testing.assert_allclose(ref_output, tflite_output, atol=1e-5)
```

#### 4. Subclassed Model Tests

```python
def test_subclassed_model_export(self):
    """Test exporting subclassed models."""
    model = get_model("subclass")
    # Ensure model is traced before export
    dummy_input = np.zeros((1, 10), dtype=np.float32)
    _ = model(dummy_input)  # Trace the model
    
    # Export and validate
    model.export(filepath, format="litert")
```

#### 5. Edge Cases

```python
def test_batch_normalization_export(self):
    """Test models with BatchNormalization layers."""
    layer_list = [
        layers.Dense(10, activation="relu"),
        layers.BatchNormalization(),
        layers.Dense(1, activation="sigmoid"),
    ]
    model = get_model("sequential", layer_list=layer_list)
    # ... test conversion and inference

def test_lstm_export(self):
    """Test models with LSTM layers (requires AI Edge LiteRT)."""
    if not AI_EDGE_LITERT_AVAILABLE:
        self.skipTest("LSTM models require AI Edge LiteRT interpreter.")
    model = get_model("lstm")
    # ... test conversion
```

### Test Utilities

```python
def _normalize_name(name):
    """Normalize tensor names for input matching."""
    normalized = name.split(":")[0]
    if normalized.startswith("serving_default_"):
        normalized = normalized[len("serving_default_"):]
    return normalized

def _set_interpreter_inputs(interpreter, inputs):
    """Set interpreter inputs handling dicts and lists."""
    # Handles named inputs, positional inputs, and input matching

def _get_interpreter_outputs(interpreter):
    """Extract outputs from interpreter."""
    output_details = interpreter.get_output_details()
    outputs = [
        interpreter.get_tensor(detail["index"]) 
        for detail in output_details
    ]
    return outputs[0] if len(outputs) == 1 else outputs
```

### Integration Tests

Test end-to-end workflows:

```python
class EndToEndConversionTest(TestCase):
    @pytest.mark.large
    def test_mobilenet_v2_conversion(self):
        """Test conversion of MobileNetV2."""
        model = keras.applications.MobileNetV2(weights='imagenet')
        
        litert_model = litert.export(
            model,
            optimizations=[litert.Optimize.DEFAULT]
        )
        
        litert_model.save('/tmp/mobilenet_v2.tflite')
        
        # Verify file size is reasonable
        file_size = os.path.getsize('/tmp/mobilenet_v2.tflite')
        self.assertLess(file_size, 15 * 1024 * 1024)  # <15MB
    
    @pytest.mark.large
    def test_bert_conversion_with_quantization(self):
        """Test BERT model conversion with quantization."""
        model = create_bert_model()
        
        litert_model = litert.export(
            model,
            representative_dataset=bert_calibration_gen,
            target_spec=litert.TargetSpec(
                supported_ops=[litert.OpsSet.TFLITE_BUILTINS_INT8]
            )
        )
        
        # Verify model size reduction
        quantized_size = litert_model._flatbuffer_data.size
        original_size = get_model_size(model)
        
        # Quantization should reduce size by ~4x
        self.assertLess(quantized_size, original_size / 3)
```

### Compatibility Tests

Test across different framework versions:

```python
class CompatibilityTest(TestCase):
    @pytest.mark.extra_large
    @pytest.mark.parametrize('tf_version', ['2.13', '2.14', '2.15'])
    def test_tensorflow_version_compatibility(self, tf_version):
        """Test conversion works across TensorFlow versions."""
        # Test with specified TF version
        pass
    
    @pytest.mark.extra_large
    @pytest.mark.parametrize('torch_version', ['2.0', '2.1', '2.2'])
    def test_pytorch_version_compatibility(self, torch_version):
        """Test conversion works across PyTorch versions."""
        # Test with specified PyTorch version
        pass
```

## Performance Considerations

### Model Size

The implementation uses standard TFLite conversion which provides:
- **Flatbuffer serialization**: Efficient binary format
- **SELECT_TF_OPS fallback**: Ensures maximum compatibility but may increase size
- **Typical sizes**: Simple models (0.5-2 MB), Complex models (10-100 MB)

Model size can be viewed during export with `verbose=True`:

```
TFLite model converted successfully. Size: 0.45 MB
```

**Current Implementation**: TensorFlow backend only

```python
# Backend check in Keras Core
import keras
if keras.backend.backend() != "tensorflow":
    raise ValueError(
        "LiteRT export is only supported with TensorFlow backend. "
        f"Current backend: {keras.backend.backend()}"
    )
```

**Future**: JAX and PyTorch backends could be supported via:
- JAX: Convert to TensorFlow via `jax2tf`, then to TFLite
- PyTorch: Convert to ONNX, then use `onnx2tf` + TFLite converter

### 14.3 Conversion Strategy Selection

Keras Core automatically selects the optimal strategy:

```python
def _choose_conversion_strategy(model):
    """Determine conversion strategy based on model type."""
    
    # Strategy 1: Direct (fast, preferred)
    if isinstance(model, (keras.Sequential, keras.Functional)):
        return "direct"
    
    # Strategy 2: Wrapper (slower, for complex models)
    else:
        return "wrapper"
```

**Rationale**:
- **Direct conversion** is ~2x faster but only works for graph-based models
- **Wrapper conversion** is universal but adds overhead

### 14.4 Error Propagation

Errors flow through the layers with clear context:

```python
# Keras-Hub Layer Errors
try:
    config = ExporterRegistry.get_config(model)
except ValueError:
    # "No exporter config found for model type 'AudioClassifier'.
    #  Supported types: text_classifier, causal_lm, ..."
    
# Keras Core Layer Errors  
try:
    signature = infer_input_signature(model)
except ValueError:
    # "Model must be built before export. Call model.build(input_shape)."
    
# TFLite Converter Errors
try:
    tflite_model = converter.convert()
except Exception as e:
    # "TFLite conversion failed: Unsupported op 'CustomOp' in layer 'layer_name'"
```

---

## 15. Error Handling Strategy

### 15.1 Error Categories

The system defines clear error categories:

| Category | Responsible Layer | Example |
|----------|------------------|---------|
| **Model Type Unsupported** | Keras-Hub | AudioClassifier not implemented yet |
| **Model Not Built** | Keras Core | model.export() before model.build() |
| **Invalid Signature** | Keras Core | Signature mismatch with model inputs |
| **Unsupported Operations** | TFLite Converter | Custom ops not in TFLite |
| **Conversion Failure** | TFLite Converter | Complex control flow |

### 15.2 Error Message Guidelines

All error messages follow these principles:

1. **What went wrong**: Clear description of the error
2. **Why it happened**: Root cause explanation
3. **How to fix it**: Actionable next steps

**Example**:
```python
# Bad error (old approach)
# ValueError: Export failed

# Good error (current approach)
raise ValueError(
    "Model must be built before export. "  # What
    "The model has no defined input shape. "  # Why
    "Call model.build(input_shape) or pass sample data to model(x)."  # How
)
```

### 15.3 Keras-Hub Error Handling

```python
class LiteRTExporter:
    def export(self, model, filepath, **kwargs):
        # Validate model type
        try:
            adapter_type = self._get_model_adapter_class(model)
        except ValueError as e:
            # Re-raise with additional context
            raise ValueError(
                f"Cannot export {type(model).__name__} to LiteRT. "
                f"{str(e)}\n\n"
                f"If you need support for this model type, "
                f"please file an issue at: "
                f"https://github.com/keras-team/keras-hub/issues"
            )
```

### 15.4 Keras Core Error Handling

```python
def export_litert(model, filepath, **kwargs):
    # Validate model is built
    if not model.built:
        raise ValueError(
            "Model must be built before export. "
            "Call model.build(input_shape) or model(sample_input) first."
        )
    
    # Validate TensorFlow backend
    if keras.backend.backend() != "tensorflow":
        raise ValueError(
            f"LiteRT export requires TensorFlow backend. "
            f"Current backend: {keras.backend.backend()}. "
            f"Set KERAS_BACKEND=tensorflow environment variable."
        )
    
    # Try conversion with fallback
    try:
        # Try direct conversion first
        tflite_model = _direct_conversion(model, **kwargs)
    except Exception as direct_error:
        try:
            # Fallback to wrapper-based
            tflite_model = _wrapper_conversion(model, **kwargs)
        except Exception as wrapper_error:
            # Both strategies failed - provide detailed error
            raise ValueError(
                "TFLite conversion failed using both strategies:\n"
                f"1. Direct conversion error: {direct_error}\n"
                f"2. Wrapper conversion error: {wrapper_error}\n\n"
                "This may be due to:\n"
                "- Unsupported operations in the model\n"
                "- Dynamic control flow (if/while loops)\n"
                "- Custom layers without TFLite equivalents\n\n"
                "Try setting litert_kwargs={'verbose': True} for more details."
            )
```

---

## 16. Testing Strategy

### 16.1 Test Coverage Overview

| Component | Test File | Coverage |
|-----------|-----------|----------|
| **Keras Core Export** | `keras/src/export/litert_test.py` | Sequential, Functional, Subclassed models |
| **Keras-Hub CausalLM** | `keras_hub/src/export/causal_lm_export_test.py` | Mock + GPT-2 preset |
| **Keras-Hub TextClassifier** | `keras_hub/src/export/text_classifier_export_test.py` | Mock + BERT preset |
| **Keras-Hub Seq2SeqLM** | `keras_hub/src/export/seq2seq_lm_export_test.py` | Mock + T5 preset |
| **Keras-Hub ImageClassifier** | `keras_hub/src/export/image_classifier_export_test.py` | Mock + ResNet preset |
| **Keras-Hub ObjectDetector** | `keras_hub/src/export/object_detector_export_test.py` | Mock + YOLO preset |

### 16.2 Test Philosophy

**Three-Tier Testing**:

1. **Unit Tests** (Fast, ~1-2 seconds per test)
   - Mock models with minimal layers
   - Test export logic, adapter creation, config selection
   - Run on every commit

2. **Integration Tests** (Medium, ~10-30 seconds per test)
   - Small preset models (e.g., bert_tiny, gpt2_small)
   - Test end-to-end export pipeline
   - Run before merge

3. **E2E Tests** (Slow, ~1-5 minutes per test)
   - Full preset models (e.g., bert_base, gpt2_medium)
   - Test numerical equivalence
   - Run nightly or on-demand

### 16.3 Numerical Equivalence Testing

All tests verify exported model matches original:

```python
def test_numerical_equivalence(self):
    """Verify TFLite model produces same outputs as Keras model."""
    
    # Create and export model
    model = self._create_test_model()
    filepath = self.get_temp_dir() + '/model.tflite'
    model.export(filepath, format='litert')
    
    # Load TFLite interpreter
    interpreter = tf.lite.Interpreter(model_path=filepath)
    interpreter.allocate_tensors()
    
    # Test with random inputs
    test_input = self._get_test_input()
    
    # Keras output
    keras_output = model(test_input).numpy()
    
    # TFLite output
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Set inputs (handle dict or list)
    if isinstance(test_input, dict):
        for i, (key, value) in enumerate(test_input.items()):
            interpreter.set_tensor(input_details[i]['index'], value.numpy())
    else:
        interpreter.set_tensor(input_details[0]['index'], test_input.numpy())
    
    interpreter.invoke()
    tflite_output = interpreter.get_tensor(output_details[0]['index'])
    
    # Assert numerical equivalence (tolerance for float32 precision)
    self.assertAllClose(keras_output, tflite_output, atol=1e-5, rtol=1e-5)
```

### 16.4 Test Execution

```bash
# Run all export tests
pytest keras_hub/src/export/

# Run with production models (slow)
pytest keras_hub/src/export/ --run-large

# Run specific model type
pytest keras_hub/src/export/causal_lm_export_test.py

# Run with verbose output
pytest keras_hub/src/export/ -v -s
```

**Current Results**:
- ✅ 42 tests passing (all functional tests)
- ⏭️ 27 tests skipped (production model tests, requires --run-large)
- ❌ 0 tests failing

---

## 17. Performance Considerations

### 17.1 Export Time Benchmarks

| Model Type | Size | Conversion Time | Strategy |
|------------|------|-----------------|----------|
| Sequential (10 layers) | 0.5 MB | ~2-3 seconds | Direct |
| Functional (ResNet-like) | 25 MB | ~5-8 seconds | Direct |
| Subclassed (Custom) | 10 MB | ~10-15 seconds | Wrapper |
| BERT Base | 110 MB | ~20-30 seconds | Wrapper |
| GPT-2 Medium | 350 MB | ~45-60 seconds | Wrapper |

**Key Factors**:
- Direct conversion is ~2-3x faster than wrapper
- Model size affects serialization time linearly
- AOT compilation adds 1-5 minutes per target

### 17.2 Model Size Optimization

```python
# Without optimization
model.export('model.tflite', format='litert')
# Size: 25 MB (float32 weights)

# With quantization (use TFLite converter options)
model.export('model.tflite', format='litert',
             litert_kwargs={
                 'converter_options': {
                     'optimizations': [tf.lite.Optimize.DEFAULT]
                 }
             })
# Size: ~6 MB (int8 quantized)
```

### 17.3 Inference Performance

**Exported model characteristics**:
- **Precision**: Float32 by default (same as Keras)
- **Ops**: Mix of TFLITE_BUILTINS (fast) + SELECT_TF_OPS (compatibility)
- **Optimization**: Default TFLite optimizations applied

**Typical Inference Speed** (on mobile CPU):
- Image classification (ResNet-50): ~50-100ms
- Text classification (BERT-Base): ~200-500ms
- Object detection (YOLO): ~100-200ms

**AOT Compilation Benefits**:
- Qualcomm Hexagon DSP: ~2-5x speedup
- MediaTek APU: ~3-8x speedup  
- Requires hardware-specific runtime

---

# PART V: LIMITATIONS AND FUTURE WORK

> **Note**: For comprehensive usage examples including multi-input models, custom layers, and AOT compilation, see the [LiteRT Export User Guide](https://github.com/keras-team/keras-hub/blob/main/docs/guides/litert_export.md) *(placeholder link)*.

## 18. Limitations

1. **Backend Support**:
   - ✅ TensorFlow backend required for Keras export path
   - ℹ️ JAX models: Use [JAX to LiteRT](https://ai.google.dev/edge/litert/models/convert_jax) direct conversion
   - ℹ️ PyTorch models: Use [PyTorch to LiteRT](https://ai.google.dev/edge/litert/models/convert_pytorch) direct conversion

2. **Keras-Hub Model Types**:
   - ✅ Text: CausalLM, TextClassifier, Seq2SeqLM
   - ✅ Vision: ImageClassifier, ObjectDetector, ImageSegmenter
   - ❌ Audio: AudioClassifier, Wav2Vec2 (not implemented)
   - ❌ Multimodal: CLIP, LLaVA (not implemented)

3. **TFLite Operations**:
   - Some Keras layers require SELECT_TF_OPS (larger model size)
   - Custom layers may need manual TFLite op registration
   - Ragged/sparse tensors not fully supported

4. **Dynamic Features**:
   - Dynamic control flow (if/while loops) requires wrapper strategy
   - Variable batch size supported, but fixed shapes preferred
   - Dynamic sequence lengths work but less optimized

### 22.2 Known Issues

1. **Large Models**:
   - Models >2GB may hit TFLite file size limits
   - Workaround: Use model sharding or compression

2. **LSTM/GRU Inference**:
   - Exported models work but may need AI Edge LiteRT runtime
   - Standard TFLite runtime has limited support

3. **Quantization**:
   - Models exported as FP32 by default (even if trained in bfloat16)
   - Keras quantization API (`.quantize()`) incompatible with LiteRT export - weights revert to FP32
   - Use TFLite converter's post-training quantization for on-device deployment
   - Quantization-aware training must be done through TFLite workflow
   - **Note**: This is a limitation of the converter requiring LiteRT-specific quantization format

---

## 23. Future Enhancements

## 19. Future Enhancements

### Stage 2: Android/iOS Runtime Library (Planned)
Development of comprehensive mobile runtime libraries to simplify on-device inference:
- **C++/Java/Kotlin API Wrappers**: Native inference APIs for Android/iOS
- **Preprocessing Utilities**: Task-specific preprocessing (tokenization, image normalization)
- **Postprocessing Utilities**: Common operations (argmax, NMS, detokenization)
- **Unified Interface**: Consistent API across different model types
- See [Android Runtime Library Design](https://github.com/keras-team/keras-hub/issues/XXXXX) *(placeholder)*

### Additional Model Type Support
- **Audio Models**: AudioClassifier, Wav2Vec2, Whisper
- **Multimodal Models**: CLIP, LLaVA, Flamingo
- **JAX Backend**: Export for JAX-based models via `orbax-export`

### Export Optimizations
- Auto-quantization (int8/float16)
- Export validation (numerical equivalence checking)
- Model compression integration

---

## 20. References

**Related Pull Requests**:
- Keras Core LiteRT Export (PR #21674): https://github.com/keras-team/keras/pull/21674
- Keras-Hub LiteRT Export (PR #2405): https://github.com/keras-team/keras-hub/pull/2405

**Design Inspirations**:
- HuggingFace Optimum: Registry pattern for model-type configs
- TensorFlow Lite Converter: Multi-strategy conversion approach
- ONNX Export: Unified export interface

**Dependencies**:
- TensorFlow ≥2.13 (required for TFLite converter)
- AI Edge LiteRT (optional, for AOT compilation)

---

## Appendix: File Structure

**Keras Core Files** (`keras/src/export/`):
- `litert.py` - LiteRTExporter class, export_litert() function
- `export_utils.py` - Input signature utilities
- `litert_test.py` - Unit tests

**Keras-Hub Files** (`keras_hub/src/export/`):
- `base.py` - KerasHubExporter and KerasHubExporterConfig base classes
- `configs.py` - Task-specific config classes + get_exporter_config()
- `litert.py` - LiteRT exporter, ModelAdapter, export_litert()
- `*_test.py` - Per-model-type tests

---

**END OF DESIGN DOCUMENT**
