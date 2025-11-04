# Probabilistic Safety-Certified Deep Reinforcement Learning for Discrete Time Systems Using High-Order Discrete-Time Control Barrier Functions
Tensorflow2 implementation of the RL-HODTCBF algorithm.

What is this algorithm all about?

A Reinforcement Learning (RL) controller learning efficiently and safely an optimal policy, relying on safety guarantees utilizing high-order discrete-time control barrier functions (HODTCBFs) with online learned knowledge of unknown dynamics through Gaussian process (GP) regression, has been proposed.
This new, to the knowledge of the author, approach led to this RL-HODTCBF algorithm, which aims to guarantee safety with high probability during the learning process, for, ideally, any RL algorithm, demonstrating also efficient policy exploration.

Method is tested on the type 1 diabetes patient model, the Bergman model.

For an in depth explanation feel free to contact me.

### Usage
Experiments can be run by calling:
```
python main.py
```

Hyper-parameters can be modified with different arguments to main.py.

### Results
Learning curves are found under /plots.

Data obtained from the training of the algorithm and evaluation of the obtained policy are found in /data.
