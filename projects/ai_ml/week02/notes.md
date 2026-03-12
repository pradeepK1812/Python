## Week 2 — Classification from First Principles (Logistic Regression)

Classification predicts probabilities instead of raw numeric values.

---

### Step 1: Start with the same linear model

z = w x + b  

This defines a geometric decision boundary:

w x + b = 0  

which separates input space into two regions.

---

### Step 2: Convert linear output into probability

Linear output z can take any real value.
We need a value in [0, 1] to represent probability.

Use the sigmoid function:

σ(z) = 1 / (1 + e⁻ᶻ)

Properties:
- maps (−∞, +∞) → (0, 1)
- smooth and differentiable
- large positive z → probability near 1
- large negative z → probability near 0

Sigmoid converts distance from decision boundary into confidence.

---

### Step 3: Why squared error is not ideal

Mean squared error works for regression but is poor for classification:

- slow convergence
- weak probabilistic meaning
- unstable gradients

Instead we use cross-entropy loss.

---

### Step 4: Cross-Entropy Loss (binary)

For one sample:

L = −[ y log(p) + (1 − y) log(1 − p) ]

Where:
p = σ(z)

Behavior:
- confident wrong predictions are punished strongly
- confident correct predictions have small loss

---

### Step 5: Elegant gradient simplification

Combining sigmoid + cross-entropy gives:

∂L/∂z = p − y

Which mirrors regression gradient:

Regression: y_pred − y  
Classification: p − y  

This makes optimization stable and efficient.

---

### Step 6: Decision rule

Predict class:

if p > 0.5 → class 1  
else → class 0  

Decision boundary remains:

w x + b = 0  

Same geometry as linear regression.

---

### Key Comparison

Regression:
- output = real number
- loss = mean squared error

Classification:
- output = probability
- loss = cross-entropy

Both use:
linear model + gradient descent

---

### Core Insight

Classification is:

Linear regression  
+ probability mapping (sigmoid)  
+ cross-entropy loss  

Same optimization engine, new interpretation.

---

### Connection to Deep Learning

Softmax + cross-entropy used in neural networks and GPT are direct generalizations of logistic regression.


######################################################################################################

## Week 2 — Logistic Regression: Scratch vs sklearn vs PyTorch

We implemented binary classification in three different ways.
All three solve the same mathematical problem.

---

# 1️⃣ From Scratch (NumPy)

### What we controlled manually:
- Linear model: z = wX + b
- Sigmoid activation
- Cross-entropy loss
- Gradient derivation
- Parameter updates
- Normalization

### Training loop:
forward → compute loss → compute gradients → update parameters

### Purpose:
Understand math + optimization deeply.

### Pros:
✔ Full control  
✔ Full visibility of gradients  
✔ Strong intuition building  

### Cons:
✖ Not scalable  
✖ Easy to introduce bugs  
✖ No GPU support  

---

# 2️⃣ sklearn Version

### What sklearn handled:
- Sigmoid internally
- Cross-entropy loss
- Optimization solver (LBFGS by default)
- Regularization
- Numerical stability

### What we controlled:
- Train/test split
- Scaling (StandardScaler)
- Evaluation

### Purpose:
Industry baseline model.

### Pros:
✔ Fast  
✔ Reliable  
✔ Few lines of code  
✔ Good for structured data  

### Cons:
✖ Limited flexibility  
✖ Harder to customize training dynamics  

---

# 3️⃣ PyTorch Version

### What PyTorch handled:
- Automatic differentiation (autograd)
- Stable BCEWithLogitsLoss
- Optimizer (SGD/Adam)
- Parameter tracking

### What we controlled:
- Model definition (nn.Linear)
- Training loop
- Learning rate
- Optimizer choice

### Purpose:
Deep learning foundation.

### Pros:
✔ Fully scalable  
✔ GPU compatible  
✔ Same framework used for deep networks  
✔ Maximum flexibility  

### Cons:
✖ More verbose than sklearn  
✖ Requires understanding of tensors & gradients  

---

# 🔁 Core Mathematical Comparison

| Component | Scratch | sklearn | PyTorch |
|-----------|----------|----------|----------|
| Linear model | Manual | Internal | nn.Linear |
| Sigmoid | Manual | Internal | Inside BCEWithLogitsLoss |
| Loss | Manual cross-entropy | Internal | BCEWithLogitsLoss |
| Gradients | Manual formula | Solver | autograd |
| Update rule | Gradient descent | Optimized solver | Optimizer.step() |
| Scaling | Manual | StandardScaler | Manual |
| Evaluation | Manual | accuracy_score | Tensor ops |

---

# 📐 Geometry Insight

All three learned the same decision boundary:

w x + b = 0

Parameter values may differ due to:
- normalization
- optimization method
- regularization

But the separating hyperplane is equivalent.

---

# 🧠 Engineering Insight

Scratch:
    Learning tool

sklearn:
    Baseline production model

PyTorch:
    Deep learning foundation

Real ML workflow:
    Scratch (understand) →
    sklearn (baseline) →
    PyTorch (scale & customize)

---

# 🎯 Big Takeaway

Classification =

Linear regression  
+ sigmoid (probability mapping)  
+ cross-entropy loss  

Same optimization engine across all frameworks.
Only abstraction level changes.
####################################################################################

## Probabilistic Foundations of Classification — MLE to Cross-Entropy

Classification loss functions arise from Maximum Likelihood Estimation (MLE).

---

# Binary Classification (Logistic Regression)

### Output and probability model

Labels:
y ∈ {0,1}

Each label is modeled as a Bernoulli random variable:

P(y | p) = pʸ (1 − p)¹⁻ʸ  

where:

p = σ(wx + b)

is the predicted probability of class 1.

---

### Likelihood over dataset

For samples {y₁, ..., yₙ}:

Likelihood:

P = ∏ᵢ pᵢʸⁱ (1 − pᵢ)¹⁻ʸⁱ  

---

### Why take log?

Products of probabilities become extremely small and hard to optimize.

Taking log:

log P = Σ [ yᵢ log pᵢ + (1 − yᵢ) log(1 − pᵢ) ]

This converts multiplication into addition and improves numerical stability.

---

### Why negative log?

MLE maximizes log-likelihood.

Gradient descent minimizes a loss.

So we define:

Loss = − log P

Which gives:

Loss = − Σ [ yᵢ log pᵢ + (1 − yᵢ) log(1 − pᵢ) ]

This is **binary cross-entropy loss**.

High predicted probability → small loss  
Low predicted probability → large loss  

---

### Why sigmoid is used

Linear output:

z = wx + b  

Sigmoid maps it to:

p ∈ (0,1)

which matches Bernoulli probability requirements.

---

# Multi-class Classification (Softmax)

For K classes, outputs follow a Categorical distribution:

p = (p₁, ..., pₖ),  Σpᵢ = 1  

Softmax converts logits into valid probabilities:

pᵢ = exp(zᵢ) / Σ exp(zⱼ)

---

### Categorical MLE

Likelihood:

P = ∏ pᵢʸⁱ  

Log-likelihood:

log P = Σ yᵢ log pᵢ  

Loss:

Loss = − Σ yᵢ log pᵢ  

This is **softmax cross-entropy loss**.

---

# Unified principle

Choose output distribution → write likelihood → maximize likelihood → minimize negative log-likelihood.

| Output | Distribution | Activation | Loss |
|-------|-------------|-----------|------|
| real | Gaussian | identity | MSE |
| binary | Bernoulli | sigmoid | BCE |
| multi-class | Categorical | softmax | CE |

---

### Core takeaway

Cross-entropy losses are negative log-likelihoods derived from probabilistic models, not heuristic error measures.
###########################################################################################################

Concept	   Logistic Regression	             GPT
Input	     features x	                     context embeddings 
Linear layer	wx + b	                  W h + b
Activation	    sigmoid	                       softmax
Distribution	Bernoulli	               Categorical
Loss	        BCE	                                Cross-entropy
Output	        probability of class	             probability of each token
################################################################################################

GPT predicts the next token by applying softmax to linear scores and minimizing 
categorical cross-entropy — exactly multi-class logistic regression.

#########################################################################################################################
## What is actually stored inside a trained language model (LLM)

After training on large text datasets, a transformer model does **not store sentences or documents**.  
Instead, it stores **parameters (θ)** that define a function approximating the probability distribution of language.

The model ultimately learns a mapping:

P(w_t | w_1, ..., w_{t-1})

i.e.

context → probability distribution of next token

Below is a conceptual breakdown of what is physically stored and how the functional mapping works.

---

### 1. What is physically stored in the model

After training, the model contains **large matrices of learned parameters**.

Typical components include:

- Token embedding matrix
- Query, Key, Value matrices for attention
- Output projection matrices
- Feed-forward (MLP) layer weights
- Layer normalization parameters
- Final language modeling head (lm_head)

Symbolically:

θ = {W_emb, W_Q, W_K, W_V, W_O, W1, W2, W_lm}

These parameters define the neural network.

---

### 2. Functional mapping learned by the model

The transformer learns a function:

f_θ(context) → probability distribution over vocabulary

Pipeline:

tokens  
↓  
token embeddings  
↓  
attention layers  
↓  
feed-forward (MLP) layers  
↓  
hidden vector h  
↓  
linear projection  
↓  
softmax  
↓  
P(next token)

Mathematically:

z = W_lm h  
P(token) = softmax(z)

---

### 3. What knowledge looks like inside the weights

The model stores **statistical patterns encoded as geometry in vector space**.

Examples:

Word relationships:

king − man + woman ≈ queen

Semantic clusters:

dog, cat, horse → animal cluster  
car, bus, train → vehicle cluster

Syntactic features encoded in hidden vectors:

- singular vs plural
- tense information
- grammatical expectations

---

### 4. What attention matrices encode

Attention layers learn **relationships between tokens** in a sequence.

Example sentence:

"The dogs that bark loudly are noisy"

Attention allows the model to connect:

dogs ↔ are

even though they are far apart in the sequence.

Thus attention captures **long-range dependencies**.

---

### 5. What MLP layers encode

Feed-forward layers learn **nonlinear patterns** in language.

Examples of patterns learned:

plural noun → expect plural verb  
question word → expect interrogative structure  
function definition → expect return statement

These layers transform hidden vectors to capture higher-level abstractions.

---

### 6. What the final layer stores

The final projection layer (language modeling head) acts as a **multi-class classifier**.

z = W_lm h

Each row of W_lm corresponds to a token in the vocabulary.

The dot product measures how compatible the hidden state is with each token.

Softmax converts logits into probabilities.

---

### 7. What the model's "knowledge" represents

The model stores a **compressed statistical representation of language**.

It encodes patterns such as:

- syntax
- semantics
- world associations
- writing styles
- reasoning heuristics

These are learned automatically from training data.

---

### 8. Mathematical summary

The model approximates the conditional probability distribution:

P_θ(w_t | w_1, ..., w_{t-1})

where

f_θ : token sequence → ℝ^V

is implemented by a deep neural network consisting of attention and feed-forward transformations.

---

### 9. Intuitive mental model

You can imagine the model learning a **probability landscape of language**.

Given context:

"I love"

the hidden state moves to a region in vector space where tokens like:

cats, dogs, you

have high probability.

---

### 10. Final takeaway

After training, the model stores **parameters of a transformer network that implement a function mapping context → probability distribution over next tokens**, capturing statistical patterns of language rather than memorizing text.
#########################################################################################################

## How MLE → Cross Entropy → Softmax → GPT Training Connect

Modern language models (like GPT) are trained using a probabilistic principle called **Maximum Likelihood Estimation (MLE)**.  
The objective is to learn a model that assigns high probability to sequences that appear in the training data.

---

### 1. Language Modeling Objective

Given a sequence of tokens:

w₁, w₂, w₃, ..., w_T

The model learns to predict the probability of the next token given the previous ones:

P(w_t | w₁, ..., w_{t-1})

The training objective is:

maximize likelihood of the observed data

Equivalent minimization objective:

minimize negative log likelihood.

---

### 2. Chain Rule of Language Probability

The probability of an entire sentence is decomposed using the **chain rule of probability**:

P(w₁, w₂, ..., w_T)  
= P(w₁)  
× P(w₂ | w₁)  
× P(w₃ | w₁, w₂)  
× ...  
× P(w_T | w₁, ..., w_{T-1})

Language models learn these conditional probabilities.

This allows the model to generate text token by token.

---

### 3. From Likelihood to Cross-Entropy Loss

For each token prediction, we define likelihood:

L = P(correct token | context)

MLE maximizes this likelihood.

In practice we minimize the **negative log likelihood**:

Loss = - log P(correct token)

For a vocabulary of size V this becomes:

L = - Σ y_k log(p_k)

where

y_k = one-hot target vector  
p_k = predicted probability of token k

This is the **cross-entropy loss**.

---

### 4. Softmax Converts Scores to Probabilities

The model first produces **logits**:

z = W h

where

h = hidden representation from transformer  
W = output projection matrix

Softmax converts logits to probabilities:

p_k = exp(z_k) / Σ exp(z_j)

This ensures

p_k ≥ 0  
Σ p_k = 1

so the output is a valid probability distribution over tokens.

---

### 5. Final Training Objective

Combining everything:

Loss = - log softmax(z_correct)

Minimizing this pushes the model to increase probability of the correct next token.

This is equivalent to **maximum likelihood estimation of the categorical distribution**.

---

### 6. Connection to the tiny_gpt Example

In the tiny_gpt implementation:

logits = linear(x, lm_head)

This computes:

z = W_lm h

Then probabilities are computed:

probs = softmax(logits)

Loss is calculated as:

loss = -log(probs[target_token])

Which matches exactly:

L = - log P(correct token | context)

The backward pass uses the gradient:

∂L / ∂z = p - y

This pushes the model to increase probability of the correct token.

---

### 7. Training Loop in tiny_gpt

Each training step performs:

1. Read context tokens  
2. Compute hidden state using transformer layers  
3. Compute logits with linear layer  
4. Apply softmax to obtain probabilities  
5. Compute cross-entropy loss  
6. Backpropagate gradients  
7. Update parameters

Repeated over millions/billions of tokens.

---

### 8. Generation (Inference)

Once trained, the model generates text using the learned conditional probabilities:

context → logits → softmax → probabilities

Then a token is selected:

- greedy: highest probability
- sampling: probabilistic draw
- temperature / top-k / top-p sampling

The selected token is appended to the context and the process repeats.

---

### 9. Key Insight

GPT does not memorize sentences.

Instead it learns parameters θ that approximate:

P_θ(w_t | w₁, ..., w_{t-1})

This allows it to generate new sentences that follow the statistical structure of language.

---

### 10. Summary

Training GPT involves the following pipeline:

Chain rule of language probability  
→ Maximum Likelihood Estimation  
→ Negative log likelihood  
→ Cross entropy loss  
→ Softmax probability distribution  
→ Gradient descent to update model parameters.

This is the core mathematical principle behind modern language models.
################################################################################################################
## Conceptual Progression: Linear Regression → Logistic Regression → Softmax → GPT

A useful way to understand modern language models is to see them as an extension of the same ideas used in classical machine learning.

The core idea evolves as follows.

---

### 1. Linear Regression (Continuous Prediction)

Goal: predict a real number.

Model:

ŷ = wx + b

Loss function:

Mean Squared Error (MSE)

L = (ŷ − y)²

Interpretation:

The model learns a function mapping input features to a continuous output.

---

### 2. Logistic Regression (Binary Classification)

Goal: predict probability of a binary class.

Model:

z = wx + b

p = sigmoid(z)

Loss derived from Bernoulli MLE:

L = − [y log(p) + (1 − y) log(1 − p)]

Key idea:

Instead of predicting a number, we predict **probability of class 1**.

---

### 3. Softmax Classifier (Multi-Class Classification)

Goal: predict probability across multiple classes.

Model:

z = W x

p_k = exp(z_k) / Σ exp(z_j)

Loss derived from categorical MLE:

L = − Σ y_k log(p_k)

Interpretation:

The model outputs a probability distribution across many classes.

---

### 4. Language Modeling (Token Prediction)

In language modeling, each token is treated as a class.

Example vocabulary:

["cat", "dog", "runs", "walks", ...]

Given context:

"I love"

The model predicts:

P(next token | context)

Example output:

cats → 0.52  
dogs → 0.41  
pizza → 0.02  

This is exactly a **softmax classifier over the vocabulary**.

---

### 5. Transformer Feature Extractor

In GPT models:

The transformer acts as a **feature extractor**.

Pipeline:

tokens  
↓  
embeddings  
↓  
self-attention layers  
↓  
MLP layers  
↓  
hidden vector h

The hidden vector encodes context information.

---

### 6. Final Softmax Layer

The hidden representation is mapped to vocabulary scores:

z = W_lm h

Softmax converts logits to probabilities:

p_k = exp(z_k) / Σ exp(z_j)

This produces the probability distribution of the next token.

---

### 7. Training Objective

Training minimizes cross-entropy loss:

L = − log P(correct token | context)

This is equivalent to **maximum likelihood estimation**.

---

### 8. Full GPT Training Pipeline

context tokens  
↓  
transformer network  
↓  
hidden representation  
↓  
linear projection  
↓  
softmax  
↓  
cross-entropy loss  
↓  
gradient descent update

Repeated over billions of tokens.

---

### 9. Key Insight

GPT is fundamentally a **very large softmax classifier** applied repeatedly across a sequence.

The transformer architecture simply produces better contextual features.

---

### 10. Final Mental Model

Linear Regression  
→ predict numbers

Logistic Regression  
→ predict probability of two classes

Softmax Classifier  
→ predict probability across many classes

GPT  
→ predict probability of the **next word among thousands of vocabulary tokens** using a transformer to represent context.
================================================================================================================================
