# Overview

The current goal is the answer the following question:

Does fixing all but a subset of the weights of a CNN during training allow that
particular subset of weights to change faster than it would provided all of the
weights were allowed to vary simultaneously?

# Agenda

- modified version of model arch file to use fused halves (don't use seeded
parameters for the linear layer before the softmax)
- above for baseline models
- run the shit on device 2

- difference between fixing half the parameters and just training twos model,
each half the size? it may make more sense to do the latter, since we won't
have to compute gradients based on parameters that are random and frozen. think
about implementing this.
- idea based on this: if we have an ensemble, then we could consider fusing all
of them together into a "mutant" ensemble and optimizing it. note: major
problems with doing this naively: memory issues, as well as overfitting since
the mutant model will be very "wide"
  - approach 1: need some way to get rid of redundant parameters along the way
  (pruning over feature maps?) and restructuring the network dynamically.
  - approach 2: only fuse the models at the final linear + softmax layer, i.e.
  don't the convolutional layers together. this makes training tractible and
  parallelizable. this idea is likely very similar to the "inception" module
  technique.

# Future Ideas

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
- Using elastic averaging or soft weight sharing.
