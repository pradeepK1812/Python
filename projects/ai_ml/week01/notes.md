# Week 1 Notes – Linear Regression

## Observations
- Dominant operation: vectorized matrix ops
- Time complexity scales with N

## Architecture Thoughts
- Training is batch-oriented
- Memory becomes bottleneck before compute
- NumPy speed comes from C + contiguous memory

## Questions
- How would this change with streaming data?
- Where would distributed training enter?


## Timing Experiments (Full Batch GD)

N = 1,000     → 0.0066 seconds
N = 10000     -> 0.0064 seconds
N = 100,000   → 0.0250 seconds 
N = 1,000,000 →  0.2407 seconds

Observations:
- Time scales roughly linearly with N
- CPU dominates
- Memory still fits


Scaling behavior shows clear constant-overhead region at small N.
Linear O(N) compute dominates beyond ~10k samples.
Vectorized NumPy ops scale efficiently up to 1M samples.
This validates batching strategies used in production ML systems.

Whole ML model working:
guess model
→ predict
→ measure error
→ compute gradient
→ update model
→ repeat
============================================================

The whole ML training loop (in one sentence)

We assume a model 
𝑦
=
𝑤
𝑥
+
𝑏
y=wx+b,
compare its predictions to real data,
measure how wrong it is,
and use gradients to adjust 
𝑤
w and 
𝑏
b to reduce that error.

That’s machine learning at its simplest and most honest form.

Step-by-step, mapped to your code
1️⃣ Assume a model (hypothesis)
y_pred = w * X + b


You’re saying:

“I believe the world looks roughly like a straight line.”

This is a modeling assumption, not a fact.

2️⃣ Measure error (how wrong we are)
loss = np.mean((y_pred - y) ** 2)


This answers:

“On average, how far off am I?”

Squaring:

removes sign

penalizes big mistakes more

3️⃣ Compute gradients (direction to fix it)
dw = 2 * np.mean((y_pred - y) * X)
db = 2 * np.mean(y_pred - y)


This answers:

“If I slightly change w or b, will the error increase or decrease?”

Gradient = sensitivity of error.

4️⃣ Update parameters (learning)
w -= learning_rate * dw
b -= learning_rate * db


You move parameters opposite to the gradient, because that’s downhill.

=================================================================================


## Chain Rule in Linear Regression (Backpropagation Explained)

Model:
y_pred = w * x + b

Loss (single sample):
L = (y_pred - y)^2

Goal:
Find how loss changes with parameters w and b.

---

### Chain rule structure

dL/dw = (dL/dy_pred) × (dy_pred/dw)

Meaning:
How loss changes with prediction × how prediction changes with weight

---

### Step 1: Loss sensitivity

L = (y_pred - y)^2

dL/dy_pred = 2(y_pred - y)

---

### Step 2: Prediction sensitivity

y_pred = w x + b

dy_pred/dw = x  
dy_pred/db = 1

---

### Step 3: Combine (chain rule)

dL/dw = 2(y_pred - y) x  
dL/db = 2(y_pred - y)

---

### Average over dataset

dw = 2 * mean((y_pred - y) * X)  
db = 2 * mean(y_pred - y)

---

### Mapping to NumPy code

dw = 2 * np.mean((y_pred - y) * X)  
db = 2 * np.mean(y_pred - y)

---

### Interpretation

Gradient = error × parameter influence

Large error + strong influence → big update  
Small error + weak influence → small update

This is the core of backpropagation.
All deep learning uses the same principle, repeated across many layers.
=============================================================================


## NumPy vs PyTorch Training Loop (Step-by-Step Mapping)

This note compares a manual NumPy training loop with an equivalent PyTorch loop.
Both implement the same learning algorithm.

---

### Step 1: Data Representation

NumPy:
- Uses `ndarray`
- CPU only
- No gradient tracking

PyTorch:
- Uses `Tensor`
- CPU or GPU
- Can track gradients automatically

---

### Step 2: Model Parameters

NumPy:
- Parameters are plain scalars or arrays
- No built-in gradient storage

Example:
w = np.random.randn()
b = 0.0

PyTorch:
- Parameters are tensors with `requires_grad=True`
- Gradients stored in `.grad`

Example:
w = torch.randn(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

---

### Step 3: Forward Pass

NumPy:
y_pred = w * X + b

PyTorch:
y_pred = w * X + b

Same math, different tensor engine.

---

### Step 4: Loss Computation

NumPy:
loss = np.mean((y_pred - y) ** 2)

PyTorch:
loss = torch.mean((y_pred - y) ** 2)

Same loss formula.

---

### Step 5: Gradient Computation (Backpropagation)

NumPy:
- Gradients derived manually
- Explicit formulas

Example:
dw = 2 * mean((y_pred - y) * X)
db = 2 * mean(y_pred - y)

PyTorch:
- Gradients computed automatically
- Uses autograd + chain rule

Example:
loss.backward()

Results:
w.grad → ∂loss/∂w
b.grad → ∂loss/∂b

---

### Step 6: Parameter Update (Optimization)

NumPy:
w -= learning_rate * dw
b -= learning_rate * db

PyTorch:
w -= learning_rate * w.grad
b -= learning_rate * b.grad

Same gradient descent step.

---

### Step 7: Gradient Reset

NumPy:
- Gradients overwritten each iteration
- No accumulation

PyTorch:
- Gradients accumulate by default
- Must reset explicitly

Example:
w.grad.zero_()
b.grad.zero_()

---

### Step 8: Performance & Scaling

NumPy:
- Fast vectorized CPU execution
- Best for classical ML & small models

PyTorch:
- CPU + GPU execution
- Scales to deep networks & large datasets

---

### Key Insight

NumPy loop:
Predict → Loss → Manual gradients → Update

PyTorch loop:
Predict → Loss → Autograd → Update

Same algorithm.
PyTorch automates gradient computation and enables GPU acceleration.

---

### Architect Takeaway

- Use NumPy to understand the math and mechanics
- Use PyTorch to scale models and systems
- All deep learning training loops follow this structure
======================================================================================

## Full Machine Learning Lifecycle (Production View)

Machine learning is not just training a model.
It is a continuous system with multiple stages.

---

### 1. Data Collection

Sources:
- logs
- sensors
- databases
- user interactions
- APIs

Key concerns:
- data quality
- missing values
- bias
- volume growth

Architectural impact:
- storage systems
- streaming vs batch ingestion
- schema evolution

---

### 2. Data Preparation (Feature Engineering)

Tasks:
- cleaning
- normalization
- encoding categories
- handling outliers
- train/test splits

Why it matters:
Bad data beats good models every time.

Architectural impact:
- preprocessing pipelines
- reproducibility
- versioned datasets

---

### 3. Model Training

Activities:
- choosing algorithms
- optimizing loss
- tuning hyperparameters
- validating performance

Outputs:
- trained model artifacts
- metrics

Architectural impact:
- compute scaling (CPU/GPU)
- batch processing
- experiment tracking

---

### 4. Model Evaluation

Metrics:
- accuracy
- precision/recall
- RMSE
- ROC/AUC

Goals:
- generalization
- robustness
- failure analysis

Architectural impact:
- automated testing
- validation gates

---

### 5. Model Deployment (Inference)

Forms:
- REST APIs
- batch prediction jobs
- real-time streaming

Key constraints:
- latency
- throughput
- cost
- reliability

Architectural impact:
- microservices
- load balancing
- autoscaling

---

### 6. Monitoring & Observability

Track:
- prediction accuracy over time
- data drift
- concept drift
- latency
- error rates

Why it matters:
Models degrade silently.

Architectural impact:
- dashboards
- alerts
- logging pipelines

---

### 7. Retraining & Continuous Learning

Triggers:
- performance drop
- new data
- seasonality
- product changes

Approaches:
- scheduled retraining
- event-driven retraining

Architectural impact:
- automated pipelines
- version rollback
- CI/CD for ML

---

### Key Insight

ML systems are living systems.

Training is only a small part.
Most real-world complexity lives in:

data pipelines + deployment + monitoring.

---

### Architect Mindset

Focus not just on models, but on:

data flow → reliability → scalability → maintainability

That is where production ML succeeds or fails.
#####################################################################################

The linear algebra view of what you’re doing

You are trying to solve:

y=wx+b

or in matrix form:
Aθ=y

where:

𝐴 = [X,1]
A=[X1],
θ=[
w
b]

he key reality

Most of the time:

y does NOT lie in the column space of A

Meaning:

There is no exact solution to:

𝐴
𝜃
=
𝑦
Aθ=y

because:

• noise
• imperfect model
• real-world randomness

📉 So what do we do?

We find the closest possible vector in the column space of A to y.

That’s:

👉 the projection of y onto Col(A)

This is exactly least squares regression.

🧠 Geometric interpretation (important)

You’re solving:

min
  ⁡
𝜃∥Aθ−y∥ **2


Which means:

find the point in the model space that is closest to actual data

In words:

Approximate an unsolvable system with the best possible solution.


Two equivalent ways to find that projection
✅ Linear algebra method (closed form)

Normal equation:


θ=(ATA)−1 ATy

This is what sklearn uses internally (or SVD).

✅ Calculus method (gradient descent)

Minimize squared error by:

• computing gradients
• iteratively improving parameters

This is what WE HAVE  implemented.

==============================================================================

We usually can’t solve y = Aθ exactly because y is outside the column space,
 so we project y onto the column space of A by minimizing squared error,
 which can be done either with linear algebra (normal equations) or calculus (gradient descent).

====================================================================================================


Big unifying idea (very important):
===================================================
View	What’s happening
==========================================
Linear algebra	projecting onto column space
Calculus	    minimizing squared error
ML	            training a model
Geometry	    finding closest point

All the same thing.
