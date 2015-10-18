# Overview

Experiments involving catastrophic forgetting with MNIST.

# Agenda

- Parameters of function that defines model arch:
  - Depth.

- Other model arch files:
  - One file to simply load a model given the path as a command-line argument.
  - One file to fuse two models together given the paths to the left and right
  halves. Should work regardless of depth of models.
  - One file with implementation of nested submodels.

- Optimizer: use the same adadelta configuration for everything.
- Rake tasks for preprocessing data, and one rake task for each experiment.
- Use 30 epochs for all models.
- Perhaps make the grad modification function an extra CLI argument.

- Extra parameter: how many samples of digits 5--9 to use.
  - Experiment by hand first to see how the accuracy changes before doing this.

# Further Questions

- Is it always best to fix half of the weights? What if we fix a smaller or
larger fraction?

- For CNNs, what's the difference between fixing half the feature maps and
choosing half of the weights arbitrarily?

- Changing the subsets of weights that are optimized over time. Coordinate
descent?

- When two model halves are fused together, does the optimizer change all of
the feature maps by an equal amount, or are some changed disproportionately
compared to others? If the latter case holds, what qualitative properties do
these feature maps have?
  - Hypothesis: the feature maps that change should fall into two categories:
    1. Those that are changed to eliminate redundancy.
    2. Those that are changed to allow the model to discriminate between the
    first task and the second.

- Partition the kernels in each layer into sets, based on the sum of the
magnitudes of the components of the Hessian that correspond to the interaction
between each pair of kernels.
  - We can find the sum of the values of the components (not absolute values)
  by performing Hessian-vector multiplications, which can be implemented
  efficiently.
  - This is related to the pruning strategy discussed in OBD.

- Using elastic averaging or soft weight sharing to prevent models or subsets
of weights trained on different objectives from diverging from one another.
  - This is the same as Yann's suggestion to use elastic net regularization.

- Application to combining ensemble models into a single model. How to fuse the
individual models together?
  - Approach 1: get rid of redundant parameters while optimizing (pruning over
  feature maps?), and restructure the network dynamically.
  - Approach 2: only fuse the models at the final linear + softmax layer, i.e.
  don't the join intermediate layers. this makes training tractible and
  parallelizable. this idea is likely very similar to the "inception" module
  technique.
