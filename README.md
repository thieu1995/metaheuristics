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

## Single Solution
```code
None
```

## Multiple Solution


### 1._. Physics-based

* __Tug Of War Optimization__: Kaveh, A., & Zolghadr, A. (2016). A novel meta-heuristic algorithm: tug of war optimization. Iran University of Science & Technology, 6(4), 469-492.
* __Nuclear Reaction Optimization__: Wei, Z., Huang, C., Wang, X., Han, T., & Li, Y. (2019). Nuclear Reaction Optimization: A novel and powerful physics-based algorithm for global optimization. IEEE Access. 
```code
    + So many equations and loops - take time to run on larger dimension 
    + General O (g * n * d) 
    + Good convergence curse because the used of gaussian-distribution and levy-flight trajectory
    + Use the variant of Differential Evolution
```
* __Henry Gas Solubility Optimization__: Hashim, F. A., Houssein, E. H., Mabrouk, M. S., Al-Atabany, W., & Mirjalili, S. (2019). Henry gas solubility optimization: A novel physics-based algorithm. Future Generation Computer Systems, 101, 646-667.
```code 
    + Too much constants and variables
    + Still have some unclear point in Eq. 9 and Algorithm. 1
    + Can improve this algorithm by opposition-based and levy-flight
    + A wrong logic code in line 91 "j = id % self.n_elements" => to "j = id % self.n_clusters" can make algorithm converge faster. I don't know why?
    + Good results come from CEC 2014
```


### 2._. Evolutionary-based

* __Genetic Algorithms__: Holland, J. H. (1992). Genetic algorithms. Scientific american, 267(1), 66-73.
* __Differential Evolution__: Storn, R., & Price, K. (1997). Differential evolution–a simple and efficient heuristic for global optimization over continuous spaces. Journal of global optimization, 11(4), 341-359.
* __Coral Reefs Optimization Algorithm__: Salcedo-Sanz, S., Del Ser, J., Landa-Torres, I., Gil-López, S., & Portilla-Figueras, J. A. (2014). The coral reefs optimization algorithm: a novel metaheuristic for efficiently solving optimization problems. The Scientific World Journal, 2014.



### 3._. Swarm-based

* __Particle Swarm Optimization__: Eberhart, R., & Kennedy, J. (1995, October). A new optimizer using particle swarm theory. In MHS'95. Proceedings of the Sixth International Symposium on Micro Machine and Human Science (pp. 39-43). Ieee.
* __Cat Swarm Optimization__: Chu, S. C., Tsai, P. W., & Pan, J. S. (2006, August). Cat swarm optimization. In Pacific Rim international conference on artificial intelligence (pp. 854-858). Springer, Berlin, Heidelberg.
* __Whale Optimization__: Mirjalili, S., & Lewis, A. (2016). The whale optimization algorithm. Advances in engineering software, 95, 51-67.
* __Bacterial Foraging Optimization__: Passino, K. M. (2002). Biomimicry of bacterial foraging for distributed optimization and control. IEEE control systems magazine, 22(3), 52-67.
* __Adaptive Bacterial Foraging Optimization__: Yan, X., Zhu, Y., Zhang, H., Chen, H., & Niu, B. (2012). An adaptive bacterial foraging optimization algorithm with lifecycle and social learning. Discrete Dynamics in Nature and Society, 2012.
* __Artificial Bee Colony__: Karaboga, D., & Basturk, B. (2007, June). Artificial bee colony (ABC) optimization algorithm for solving constrained optimization problems. In International fuzzy systems association world congress (pp. 789-798). Springer, Berlin, Heidelberg.
* __Pathfinder Algorithm__: Yapici, H., & Cetinkaya, N. (2019). A new meta-heuristic optimizer: Pathfinder algorithm. Applied Soft Computing, 78, 545-568.
* __Harris Hawks Optimization__: Heidari, A. A., Mirjalili, S., Faris, H., Aljarah, I., Mafarja, M., & Chen, H. (2019). Harris hawks optimization: Algorithm and applications. Future Generation Computer Systems, 97, 849-872.
* __Sailfish Optimizer__: Shadravan, S., Naji, H. R., & Bardsiri, V. K. (2019). The Sailfish Optimizer: A novel nature-inspired metaheuristic algorithm for solving constrained engineering optimization problems. Engineering Applications of Artificial Intelligence, 80, 20-34.



### 4._. Human activity-based

* __Queuing Search Algorithm__: Zhang, J., Xiao, M., Gao, L., & Pan, Q. (2018). Queuing search algorithm: A novel metaheuristic algorithm for solving engineering optimization problems. Applied Mathematical Modelling, 63, 464-490.



# List of trash meta-heuristics (totally fake and unethical)

1._. Physics-based

2._. Evolutionary-based

3._. Swarm-based

* __Sandpiper Optimization Algorithm__: Kaur, A., Jain, S., & Goel, S. (2019). Sandpiper optimization algorithm: a novel approach for solving real-life engineering problems. Applied Intelligence, 1-38.
* __Sooty Tern Optimization Algorithm__: Dhiman, G., & Kaur, A. (2019). STOA: A bio-inspired based optimization algorithm for industrial engineering problems. Engineering Applications of Artificial Intelligence, 82, 148-174.
```code
+ Cant even update the position itself
+ So many unclear operators and unclear parameters
+ Can't converge
```

* __Butterfly Optimization Algorithm__: Arora, S., & Singh, S. (2019). Butterfly optimization algorithm: a novel approach for global optimization. Soft Computing, 23(3), 715-734.
```code 
+ Cant converge as the paper said
+ The code public from the author different with the pseudo-code in the paper
+ Totally trash and unethical, plagiarize from Flower Pollination Algorithm developed by Yang
```

* __Butterfly Optimization Algorithm__: Arora, S., & Singh, S. (2019). Butterfly optimization algorithm: a novel approach for global optimization. Soft Computing, 23(3), 715-734.
```code 
+ Cant converge as the paper said
+ The code public from the author different with the pseudo-code in the paper
+ Totally trash and unethical, plagiarize from Flower Pollination Algorithm developed by Yang
```

* __Blue Monkey Optimization__: (2019) The Blue Monkey: A New Nature Inspired Metaheuristic Optimization Algorithm. DOI: http://dx.doi.org/10.21533/pen.v7i3.621
```code 
+ Really? The idea look like "Chicken Swarm Optimization"
+ The pseudo-code totally shit in my opinion, just read the paper you will understand.
+ The unclear point here is the "Rate equation": really confuse because It's contain the position. As you know,
    The position is the vector, but finally, the author conclude that Rate is random number in range [0, 1]
    Luckily, using number we can plus/add number and vector or vector and vector.
    So at first, Rate is random number then after the 1st loop, its become vector. 
+ Morever, both equtions movement of blue monkey and children is the same.
+ In addition, they don't check the bound after update position.
+ Keep going, they don't tell you the how to find the global best (I mean from blue monkey group or child group)
+ The code I have done here is the version which I used my knowledge about meta-heuristics to do it. 
+ The results good for some CEC2014 functions, but can't even converge with the square function.
+ Can't get the results same at the paper.
After all, I think this is completely trash algorithm, need to remove from our comunity.
```


4._. Human activity-based

# The list will continue update...





