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
