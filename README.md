# Sense

Project Checkpoint Summary:
In this phase of the **SENSE** (**S**ch**E**duli**N**g **S**tructual l**E**arning), the [DARTS](https://arxiv.org/abs/1806.09055) algorithm was applied to DQN agents in CartPole. To our knowledge, DARTS has not been applied to Reinforcement Learning or implemented in Julia before this project.

[CartPole](https://github.com/openai/gym/wiki/CartPole-v0) is a relative simple task, with four observable enivronment variables and two agent control variables. Networks frequently used for DQN agents to solve Cartpole are usually fully connected networks with two hidden layers. 

To apply DARTS to CartPole, the network architectures were initially searched completely in parallel. Each operation was a full network. 

In all following experiments, both the first and second order approximations were used to compare the efficiency and effectiveness of each. The second order approximation is significantly harder to implement and slower to train, due to the multiple forward and backward passes required per step. 

In the following figure, six different dense network architectures were used as operations. The hidden dimensions are listed in the legend. The first five architectures in each training regime (first order approximation and second order approximation) had two hidden layers, while the final one had a single hidden layer. The softmax weight of each architecture is plotted over the course of training. In both training regimes, both the architecture with two hidden layers each of size 128 as well as the architecture with a single hidden layer of size 1024 initially increased in architecture weight, while the other four smaller networks decreased. However, later in training, the 1024 architecture fell back down while the 128x128 architecture continued to dominate. It is generally preferable in DARTS for a single operation to dominate, since that operation would be used for the sole operation for that portion of the macroarchitecture after the algorithm concludes. 
![Initial dense network experiment](https://github.com/d9w/Sense/blob/master/DARTS/Plots/init_dense.png)


Since the 128x128 architecture was very successful and is very commonly used in DQN networks for CartPole, multiple very similar architectures were tested as operations. In the following figures, three networks each with double hidden dimensions of 125, 126, and 127 were used as the operations, and the training was run much past the initial "convergence". The architecture weights are plotted first, and the reward per episode is plotted second. Both training regimes initally approached the same architecture weights with 126x126 having the highest weight, but over time after this convergence, the first order architecture weights stayed relatively static while the second order architecture weights slowly changed. In particular, the architecture weight for 127x127 eventually surpassed that of 126x126, thus changing the architecture found via DARTS.
![Similar dense networks - architecture weights](https://github.com/d9w/Sense/blob/master/DARTS/Plots/sim_dense_a.png)
![Similar dense networks - rewards](https://github.com/d9w/Sense/blob/master/DARTS/Plots/sim_dense_r.png)

In the next set of experiments, small convolutional network architectures were used as parallel operations. For 1d convolutions, the input environment observation variables were used similarly to a 4x1x1 image, while for 2d convolutions, the in environment observation variables were repeated to a 4x4x1 grid. The output of each architecture, 1x1x2, was squeezed down to the expected output shape of 2. In the first experiment, three different architectures were searched over, and the regimes showed similar architecture weights but very different rewards per episode. 
![Three parallel convolutional operations - architecture weights](https://github.com/d9w/Sense/blob/master/DARTS/Plots/par_conv_3_a.png)
![Three parallel convolutional operations - reward](https://github.com/d9w/Sense/blob/master/DARTS/Plots/par_conv_3_r.png)

When a fourth operation was added, the training is much less successful. This may be due to bad initialization.
![Four parallel convolutional operations - architecture weights](https://github.com/d9w/Sense/blob/master/DARTS/Plots/par_conv_4_a.png)
![Four parallel convolutional operations - reward](https://github.com/d9w/Sense/blob/master/DARTS/Plots/par_conv_4_r.png)

Now that convolutions are implemented, individual layers (or small networks) can be used as operations, as in the original DARTS paper. In the following experiements, five layer operations were used. The identity layer up or downsampled the space as necessary, the following three layer operations have the specified kernel size in a single 2D convolutional layer, and the final layer operation had two 2D convolutions, separated by the nonlinearity. All layers were followed by a nonlinearity, which was ReLU in all applications. The input and output shapes were generated as described in the previous experiment for 2D convolutions.  The first layer of weighted operations increased channels, the second layer downsampled space, and the third downsampled both space and channels. These experiments showed less stability overall, particularly in the first and third layers. The order approximation training has a particularly interesting shift around episode 210 .
![Series convolutional operations - architecture weights](https://github.com/d9w/Sense/blob/master/DARTS/Plots/ser_conv_a.png)
![Series convolutional operations - reward](https://github.com/d9w/Sense/blob/master/DARTS/Plots/ser_conv_r.png)

In general, the main difference between the first order approximation and second order approximation is training speed. Both training regimes typically approached similar architecture weights, but the second order often approached them slightly faster. Also, the second order approximaiton was generally less stable in the architecture weights, although future experiments could investigate whether this instability leads to eventual convergence or indefinite oscillation.

These experiments raise a lot of further questions, particularly now that DARTS can be studied in a much lower dimensional space. How sensitive is DARTS to the initializaiton of the architectural weights, standard weights, and operation choices themselves? Why does the second order approximation seem to be less stable: is it escaping a local minima? How different are the solutions found by the first and second order approximations? What other tasks can DARTS be applied to?
