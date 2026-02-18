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

