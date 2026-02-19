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

