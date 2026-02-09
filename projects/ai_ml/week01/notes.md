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

