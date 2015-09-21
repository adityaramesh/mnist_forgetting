# Overview

The current goal is the answer the following question:

Does fixing all but a subset of the weights of a CNN during training allow that
particular subset of weights to change faster than it would provided all of the
weights were allowed to vary simultaneously?

# Agenda

- Create a version of `run_model.lua` file that saves the state of the model
every 10 epochs to a new file. Train this model for 80 epochs so that we have a
converged model and validation accuracy graph to compare against.
- Use each of the partially trained to train a new model for 10, 20 and 30
epochs (separate trials, start first with 20), but only allow half of the
weights in this new model to change. Save each "stage 2" model to a new file.
- Train each of the "stage 2" models to convergence, but now allow all of the
weights to change.
- Also fix a subset of the weights of the converged model, and see if
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

- Visualization of weights over time.
