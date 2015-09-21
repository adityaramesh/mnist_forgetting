# Overview

The current goal is the answer the following question:

Does fixing all but a subset of the weights of a CNN during training allow that
particular subset of weights to change faster than it would provided all of the
weights were allowed to vary simultaneously?

# Agenda

- Train the fused models to convergence.
- Train a baseline model for 200 epochs so that we can compare train/validation
accuracy plots.
- Create plots.

- Also fix a subset of the weights of the converged baseline model, and see if
optimizing a smaller set of weights allows us to escape local minima.
  - Instead of fixing half the weights, try fixing smaller fractions to see if
  this helps. We would need to try different subsets this time, instead of just
  the first half.

- Explore other ways of fixing half the weights, and note patterns in the
strategies that work best:
  - First half of kernels/weights in each layer, except the FC layers (current
  strategy).
    - Maybe we should include the FC layers?
  - Random kernels instead of fixed ones.
  - Random sampling.

- Parititon the kernels in each layer into sets, based on the sum of the
magnitudes of the components of the hessian corresponding to the interaction
between each pair of kernels.
  - We can find the sum of the values of the components (not absolute values)
  by performing hessian-vector multiplications, which can be implemented
  efficiently.
  - This is related to the pruning strategy discussed in OBD.

- Visualization of weights over time.
- Changing the subsets of weights that are optimized over time.
