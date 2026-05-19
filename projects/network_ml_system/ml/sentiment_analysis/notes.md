# NLP / Transformer Learning Progress Notes

## 1. Basic Neural Network Understanding

Started with:

* Logistic Regression
* XOR problem
* Hidden layers
* Activation functions

Key learnings:

* Neural networks learn by adjusting weights using gradients
* ReLU can fail for XOR because XOR is non-linear
* tanh worked better because it supports both positive and negative activations

Important concepts understood:

* Forward pass
* Loss computation
* Backpropagation
* Gradient descent
* Weight updates

Core training flow:

```python
predictions = model(X)

loss = criterion(predictions, y)

optimizer.zero_grad()

loss.backward()

optimizer.step()
```

Understanding:

* `loss.backward()` computes gradients
* `optimizer.step()` updates parameters using gradient descent

---

# 2. Transition from Numerical ML to NLP

Started building a sentiment analysis system using PyTorch.

Goal:

* Learn NLP fundamentals
* Build toward transformers and LLM understanding

---

# 3. Vocabulary Mapping

Created:

```python
word_to_index
```

Purpose:

* Convert words into token IDs

Example:

```text
excellent -> 12
terrible -> 14
```

Important realization:

* Token IDs contain NO semantic meaning
* They are only identifiers

---

# 4. Sequence Tensor Preparation

Converted sentences into:

```python
X_tensor
```

Shape:

```text
[batch, sequence]
```

Added:

* padding
* fixed sequence lengths

Learned:

* Why tensors require equal-length sequences
* Why padding is needed

---

# 5. Embeddings

Introduced:

```python
nn.Embedding()
```

Concept:

```text
token ID -> trainable dense vector
```

Example:

```text
excellent ->
[0.2, -0.7, 1.3, ...]
```

Important understanding:

* Embeddings are trainable
* Each word gets a vector representation
* Similar-context words gradually move closer in vector space

---

# 6. Embedding Matrix Understanding

Embedding matrix shape:

```text
[vocab_size x embedding_dim]
```

Example:

```text
20 x 8
```

Meaning:

* 20 words
* each represented by 8-dimensional vector

Important realization:

* Each row corresponds to one token embedding
* Gradients update embedding vectors during training

---

# 7. Semantic Representation Learning

Observed:

* Positive words cluster together
* Negative words cluster together

Examples:

```text
great
excellent
love
```

vs

```text
terrible
bad
hate
```

Major realization:

```text
language can be represented geometrically
```

---

# 8. Embedding Visualization

Built:

```python
visualize_embeddings.py
```

Used matplotlib to visualize embeddings.

Learned:

* Embeddings form semantic geometry
* Similar words tend to occupy nearby regions
* Meaning emerges from training dynamics

Important limitation discovered:

* Visualization only shows 2 dimensions
* Actual embeddings are high-dimensional

---

# 9. Primitive Attention Mechanism

Initial issue:

```text
all tokens contributed equally
```

Old approach:

```python
embedded.mean(dim=1)
```

Problem:

* Important words diluted by less useful words

Introduced:

```text
token importance weighting
```

Concept:

```text
some words matter more than others
```

Example:

```text
excellent > this
```

---

# 10. Attention Weighting

Implemented:

```python
attention_scores
attention_weights
```

Used:

```python
softmax()
```

Learned:

* Attention converts scores into probability distributions
* Important tokens can dominate sentence representation

Important intuition:

```text
attention = information prioritization
```

---

# 11. Primitive Self-Attention

Major transition:

```text
tokens interacting with other tokens
```

Implemented:

```python
embedded @ embedded.transpose(1,2)
```

Realization:

```text
every token compares itself with every other token
```

Produced:

```text
[token x token]
```

attention matrix.

---

# 12. Tensor Shape Understanding

Learned transformer tensor reasoning.

Example:

Embedding tensor:

```text
[batch, sequence, embedding_dim]
```

Example:

```text
[1, 4, 8]
```

Transpose:

```python
transpose(1,2)
```

became:

```text
[1, 8, 4]
```

Important realization:

```text
matrix multiplication computes pairwise token similarities
```

---

# 13. Attention Matrix Interpretation

Observed:

* Tokens mostly attended to themselves

Example:

```text
excellent -> excellent
```

had strongest attention.

Major realization:

```text
raw embedding similarity causes self-similarity dominance
```

This naturally led to:

```text
need for Query/Key/Value projections
```

---

# 14. Static vs Contextual Meaning

Major conceptual breakthrough.

Current model:

```text
static embeddings
```

Meaning:

```text
same word -> same vector everywhere
```

Problem example:

```text
bank account
river bank
```

Current system struggles because:

```text
meaning mostly tied to training statistics
```

---

# 15. Transformer Insight

Real transformers:

* start with static embeddings
* repeatedly refine token meaning through attention layers

Key realization:

```text
meaning is dynamically constructed from context
```

NOT:

```text
permanently attached to tokens
```

---

# 16. Query / Key / Value (QKV) Intuition

Understanding developed before implementation.

Query:

```text
What information am I searching for?
```

Key:

```text
What information do I contain?
```

Value:

```text
What information should I contribute?
```

Attention mechanism becomes:

```text
relationship-driven contextual matching
```

instead of:

```text
raw embedding similarity
```

Core formula learned:

Q = XWQ

K = XWK

V = XWV

Attention:

```text
softmax(QK^T)V
```

---

# 17. Biggest Conceptual Learnings So Far

## Static Embeddings

* store general semantic tendencies

## Self-Attention

* enables token interaction

## Transformers

* dynamically refine meaning through context

## Language Understanding

* emerges from relational token interactions

---

# 18. Important Current Understanding

Current learning state:

* tokenization ✔
* embeddings ✔
* semantic geometry ✔
* attention weighting ✔
* self-attention ✔
* attention matrices ✔
* contextual interaction ✔
* static vs contextual meaning ✔
* transformer intuition ✔

---

# 19. Most Important Realization

Transformers are fundamentally:

```text
dynamic contextual representation systems
```

NOT:

```text
simple word lookup systems
```

Meaning emerges through:

* token relationships
* attention
* contextual refinement
* repeated representation transformation

