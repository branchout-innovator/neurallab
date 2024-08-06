#### Optimization is where the model is trained iteratively so it produces a maximum and minimum function. Optimization algorithms are a division of algorithms used to find the most suitable possible solution to a problem. What makes optimization algorithms so powerful is their ability to make decisions dependent on reliable data and models acquired by physical experiments or simulations, meaning they can swiftly solve problems without relying only on manual processes. 

<br>

Maxima and minima are the highest and lowest values of a function within a specific range. Global maxima and minima are the maximum and minimum values of the entire function while local maxima and minima are the maximum and minimum values within a specific range of the function. In a function, there is only one global maxima and one global minima but there can be several local maxima and minima.

<br>

## **Forms of Optimization Algorithms** 
#### Gradient descent: minimizes a function and finds the local minima of a differentiable function using the formula:  X=X-lr * d/dx f(X)
* X is the input 
* F(X) is the output based on X
* LR is the learning rate which is what determines the size of each step to reach the minimum of the function 

<br>

### Adam optimizer:
The Adam optimizer is an iterative algorithm that can be used to minimize the loss function while neural networks are being trained. The Adam optimizer utilizes the squared gradients to gauge the learning rate. The optimizer also helps adjust the settings of the network to improve its performance when doing its job such as recognizing images or understanding text. 

<br>

**Particle swarm optimization** - 
Particle swarm optimization uses the notion of group behavior of organisms such as birds to optimize solutions. (works effectively with complex systems such as robotic systems and control systems engineering systems)

<br>

**Hessian Matrix-based techniques** - 
Hessian matrix-based techniques are often used when the optimization of multiple objectives needs to be performed 

<br>

**Genetic algorithms** -
Genetic algorithms are helpful when solving discrete combinatorial search problems like scheduling tasks. 

<br>

**Newton’s method** -  
Newton’s method is an iterative refinement technique. After finding the approximate solution it can be refined by using steps of Newton’s method. 

<br>

**Simulated annealing** - 
Simulated annealing is used when local minima could become global minima 

<br>

**Ant colony optimization** - 
Ant colony optimization can be useful when addressing routing problems 

<br>

## **Optimization algorithms can be divided into 3 general categories** 
* Search procedures
* Loss functions 
* Convex programming 

<br>

#### What makes each type of algorithm unique is the approach it uses to find the most suitable solution to an optimization problem. 

<br>

### *References*
Guide, S. (n.d.). Understanding Optimization Algorithms in Machine Learning | by Supriya Secherla. Towards Data Science. Retrieved July 16, 2024, from https://towardsdatascience.com/understanding-optimization-algorithms-in-machine-learning-edfdb4df766b
Optimization Algorithms. (n.d.). Complexica. Retrieved July 16, 2024, from https://www.complexica.com/narrow-ai-glossary/optimization-algorithms
What is Adam Optimizer? (2024, April 26). Analytics Vidhya. Retrieved July 16, 2024, from https://www.analyticsvidhya.com/blog/2023/09/what-is-adam-optimizer/#
BISWA NATH DATTA, CHAPTER 13 - NUMERICAL SOLUTIONS AND CONDITIONING OF ALGEBRAIC RICCATI EQUATIONS, Editor(s): BISWA NATH DATTA, Numerical Methods for Linear Control Systems, Academic Press, 2004, Pages 519-599, ISBN 9780122035906, https://doi.org/10.1016/B978-012203590-6/50017-3. (https://www.sciencedirect.com/science/article/pii/B9780122035906500173)
