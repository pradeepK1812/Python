# Audience Intelligence System Architecture

## Project Goal

The Audience Intelligence System is a custom Transformer-based NLP model built from scratch in PyTorch.

The system analyzes audience comments and predicts the dominant audience emotion while also exposing secondary emotions and attention patterns learned by the model.

Example:

Input:

```
this tutorial was amazing but still confusing
```

Output:

```
confusion   -> 74%
excitement  -> 12%
curiosity   -> 7%
```

Audience Insight:

```
Audience mostly showed confusion.
```

---

# Supported Emotions

The model currently predicts six audience emotions:

| Label | Emotion      |
| ----- | ------------ |
| 0     | Appreciation |
| 1     | Confusion    |
| 2     | Curiosity    |
| 3     | Frustration  |
| 4     | Excitement   |
| 5     | Boredom      |

---

# High Level Pipeline

```
Raw Comment
      │
      ▼
Tokenization
      │
      ▼
CLS Token Addition
      │
      ▼
Embedding Layer
      │
      ▼
Multi-Head Self Attention
      │
      ▼
CLS Representation Extraction
      │
      ▼
Emotion Classifier
      │
      ▼
Softmax Probabilities
      │
      ▼
Top-K Emotion Analysis
      │
      ▼
Audience Insight Generation
```

---

# Step 1 — Tokenization

Input sentence:

```
this tutorial is excellent
```

Tokenized:

```
["this", "tutorial", "is", "excellent"]
```

Vocabulary lookup:

```
[1, 7, 3, 13]
```

Purpose:

Convert words into numerical IDs that can be processed by the neural network.

---

# Step 2 — CLS Token

A special classification token is inserted at the beginning.

Input:

```
["this", "tutorial", "is", "excellent"]
```

Becomes:

```
["[CLS]", "this", "tutorial", "is", "excellent"]
```

Token IDs:

```
[0, 1, 7, 3, 13]
```

Purpose:

The CLS token acts as a sentence-level representation.

After attention processing, the CLS embedding contains information from the entire sentence.

---

# Step 3 — Embedding Layer

Input Shape:

```
[batch_size, sequence_length]
```

Example:

```
[1, 5]
```

Embedding Output:

```
[1, 5, embedding_dim]
```

Example:

```
[1, 5, 64]
```

Purpose:

Convert token IDs into dense semantic vectors.

---

# Step 4 — Query, Key and Value Projections

Each token embedding is projected into:

```
Q = Query
K = Key
V = Value
```

Shapes:

```
Q : [1, 5, 64]

K : [1, 5, 64]

V : [1, 5, 64]
```

Purpose:

Allow tokens to determine:

* What information they need
* What information they contain
* What information should be passed forward

---

# Step 5 — Scaled Dot Product Attention

Attention Scores:

```
Q × Kᵀ
```

Scaled Attention:

```
(Q × Kᵀ) / √d
```

Purpose:

Prevent large attention values from causing unstable softmax outputs.

This is a core Transformer technique introduced in the original Transformer paper.

---

# Step 6 — Multi-Head Self Attention

The model uses:

```
4 Attention Heads
```

Each head learns different relationships.

Examples:

Head 0:

```
Local context
```

Head 1:

```
Sentiment words
```

Head 2:

```
Contrast words such as "but"
```

Head 3:

```
Long-range context
```

Output Shape:

```
[batch_size, sequence_length, embedding_dim]
```

Purpose:

Allow the model to learn multiple language patterns simultaneously.

---

# Step 7 — Context-Aware Token Representations

Attention weights are applied to Value vectors.

Result:

```
Context-aware embeddings
```

Each token now contains information from surrounding words.

Example:

```
confusing
```

can receive context from:

```
amazing
but
still
```

allowing the model to understand mixed emotions.

---

# Step 8 — CLS Representation Extraction

After attention:

```
attended[:, 0, :]
```

is extracted.

Shape:

```
[batch_size, embedding_dim]
```

Example:

```
[1, 64]
```

Purpose:

Use the CLS embedding as the sentence representation.

Modern Transformer models such as BERT use this approach.

---

# Step 9 — Emotion Classification Layer

CLS representation is passed through:

```
Linear Layer
```

Output:

```
[batch_size, 6]
```

Example:

```
[1, 6]
```

Each value represents one emotion class.

Example:

```
[2.3, 5.1, 0.9, 1.2, 3.8, 0.4]
```

These values are called logits.

---

# Step 10 — Softmax Probabilities

Softmax converts logits into probabilities.

Example:

```
appreciation : 0.57

confusion    : 0.12

curiosity    : 0.08

frustration  : 0.04

excitement   : 0.16

boredom      : 0.03
```

Properties:

* All probabilities sum to 1
* Highest probability becomes primary emotion

---

# Step 11 — Top-K Emotion Analysis

Instead of showing only one emotion:

```
torch.topk()
```

is used.

Example:

```
appreciation -> 57%

excitement   -> 16%

confusion    -> 12%
```

Purpose:

Capture mixed audience reactions.

---

# Step 12 — Audience Insight Generation

Business-friendly summary is generated.

Example:

```
Audience mostly showed appreciation.
```

or

```
Audience showed appreciation and also expressed excitement.
```

This makes the output useful for content creators.

---

# Step 13 — Attention Visualization

Attention matrices are displayed for each head.

Example:

```
confusing attends to:

still -> 0.29

but   -> 0.18
```

Purpose:

Provide explainability.

Allows inspection of what the model learned.

---

# Model Evaluation Framework

Evaluation Dataset:

```
run_evaluation.py
```

Purpose:

Measure performance on unseen examples.

Output:

```
Expected Emotion

Predicted Emotion

CORRECT / WRONG
```

Final Accuracy:

```
77.14%
```

(Current Best Checkpoint)

---

# Error Analysis Pipeline

Workflow:

1. Train Model
2. Run Evaluation
3. Capture Incorrect Predictions
4. Store Errors
5. Analyze Failure Patterns
6. Improve Dataset
7. Retrain Model

Generated File:

```
error_analysis.txt
```

Purpose:

Systematic model improvement.

---

# Key Concepts Learned

* Tokenization
* Vocabulary Building
* Embeddings
* CLS Tokens
* Self Attention
* Multi Head Attention
* Scaled Dot Product Attention
* Transformer Classification
* Softmax Probabilities
* Top-K Predictions
* Attention Visualization
* Error Analysis
* Dataset Balancing
* Mixed Emotion Detection
* Model Evaluation

---

# Current Status

Version:

```
Audience Intelligence v1
```

Architecture:

```
Custom Transformer
```

Framework:

```
PyTorch
```

Best Evaluation Accuracy:

```
77.14%
```

Next Planned Enhancements:

* Larger evaluation dataset
* Better attention visualization
* Comment batch processing
* FastAPI integration
* YouTube audience analytics
* BERT-based comparison model

