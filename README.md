# How to read my repository
1. utils: includes helper functions
2. script: includes file runs (main files)
3. models: includes all algorithms 
    * single_solution: 
    * multiple_solution: 4 folders
        * human_based
        * physics_based
        * swarm_based
        * evolutionary_based
4. How to run?
    * run files in script folder, your environment need 2 package: copy and numpy
    * also change the parameters of models in scripts's file

# Notes
* This repository includes all optimization algorithms coded in python (Numpy) in my research time
* If you want to know how to implement optimization with neural networks, take a look at this repos:
    * https://github.com/chasebk/code_FLNN

* If you see my code and data useful and use it, please cites us here

    * Nguyen, T., Tran, N., Nguyen, B. M., & Nguyen, G. (2018, November). A Resource Usage Prediction System Using Functional-Link and Genetic Algorithm Neural Network for Multivariate Cloud Metrics. In 2018 IEEE 11th Conference on Service-Oriented Computing and Applications (SOCA) (pp. 49-56). IEEE.

    * Nguyen, T., Nguyen, B. M., & Nguyen, G. (2019, April). Building Resource Auto-scaler with Functional-Link Neural Network and Adaptive Bacterial Foraging Optimization. In International Conference on Theory and Applications of Models of Computation (pp. 501-517). Springer, Cham.

* If you want to know more about code, or want a pdf of both above paper, contact me: nguyenthieu2102@gmail.com


# Meta-heuristics
- Implement algorithms based on papers

### Single Solution
```code
None
```

### Multiple Solution
1._. Physics-based
```code
1. (done) TWO (2016) - A novel meta-heuristic algorithm: tug of war optimization
2. (done) NRO (2019) - Nuclear Reaction Optimization- A novel and powerful physics-based algorithm for global optimization
    + So many equations and loops - take time to run on larger dimension 
    + General O (g * n * d) 
    + Good convergence curse because the used of gaussian-distribution and levy-flight trajectory
    + Use the variant of Differential Evolution
...
```

2._. Evolutionary-based
```code 
1. (done) GA (1989) - Genetic algorithms in search
2. (done) DE (2006) - Differential evolution: A survey of the state-of-the-art
3. (done) CRO (2013) - The Coral Reefs Optimization Algorithm: A Novel Metaheuristic for Efficiently Solving Optimization Problems
4. (done) OCRO (2019) - In reviewing process

...
```

3._. Swarm-based
```code
1. (done) PSO (1995) - Particle swarm optimization
2. (done) CSO (2006) - Cat swarm optimization
3. (done) WOA (2016) - The whale optimization algorithm
4. (done) BFO (2002) - Biomimicry of bacterial foraging for distributed optimization and control
5. (done) ABFOLS (2012) - An adaptive bacterial foraging optimization algorithm with lifecycle and social learning
6. (done) ABC (2007) - Artificial bee colony (ABC) optimization algorithm for solving constrained optimization problems
7. (done) PFA (2019) - A new meta-heuristic optimizer: Pathfinder algorithm
8. check (done) - HHO (2019) - 11 cites - Harris Hawks optimization: Algorithm and applications
9. check - (done) - SFO (2019) - 1 cites - The Sailfish Optimizer: A novel nature-inspired metaheuristic algorithm for solving constrained engineering optimization problems
10. not good - (done) - SOA (2019) - Sandpiper optimization algorithm: a novel approach for solving real-life engineering problems
    + Cant even update the position itself
    + So many unclear operators and unclear parameters
    + Can't converge
...
```

4._. Human activity-based
```code
1. (done) - QSA (2018) - Queuing search algorithm: A novel metaheuristic algorithm for solving engineering optimization problems

... 
```

#### The list will continue update...