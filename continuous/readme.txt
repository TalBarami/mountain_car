Our policy is defined as follows:
- The input is the location and the speed of the car.
- We use a 3-layer FCNN to featurize the input. We use Elu as the activation function.
- Each layer contains 64 neurons.
- Network weights are updated during the training process.
- The loss is as described in class for using Actor Critic learning.
- The network outputs a normal distribution parameters, mu and sigma.
- Given a state, we sample an action from this distribution.

Our value function is defined as follows:
- 3-layer FCNN, with a similar architecture as the policy network.
- Each layer contains 512 neurons.
- Network weights are updated during the training process.
- The loss is as described in class for using Actor Critic learning.
- The network outputs the value of a given state.
