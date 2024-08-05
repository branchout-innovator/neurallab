#### Activation functions are mathematical functions applied to the output of a neuron, similar to a filter. They introduce non-linearity (where input changes are not proportional to output changes) to the model, allowing it to learn and predict patterns more accurately. In a computer's processing system, activation switches help it make decisions by turning "on" or "off" based on the information it receives. 

<br>

## **Network Layers**

<br>

### Output Layers:
#### **Binary Classification:**
The Sigmoid function is used for the output layer because it outputs a value from 0 to 1: 
σ(x) = 1/(1 + e-x). This function is differentiable and monotonic. 

<br>

#### **Multi-class Classification:**
The Softmax function is used because it converts the outputs to probabilities that sum to 100%: 
softmax(xi​) = exi/∑jexj. The Softmax function is a more generalized logistic activation function. 

<br>

### Hidden Layers: 
#### **ReLU (Rectified Linear Unit):**
ReLU is a simple and effective activation function that helps to alleviate the vanishing gradient problem (derivatives of positive inputs are 1): 
ReLU(x) = max(0,x). ReLU currently remains the most used activation function. However, all negative values become zero, refraining ReLU models from accurately modeling data sets. 

<br>

#### **Leaky ReLU:**
To overcome the issue of a restricted domain when modeling with ReLU, Leaky ReLU allows for negative values to be modeled. f(x) = ax for all negative x. This allows the Leaky ReLU function to have a range of (neg inf, infinity). 

<br>

#### **Tanh:** 
Tanh functions are similar to sigmoid but with a range of [-1, 1]. Tanh can be more effective at training the network because of a larger gradient:  tanh(x) = ex-e-x/ex+e-x. The range of tanh is (-1,1) instead of sigmoid (0,1). Tanh is differentiable and monotonic. 

<br>

## **Why Are Activation Functions Important?**
Activation functions enhance the ability of computational models to process information and make decisions. These functions allow neural networks to discern patterns and make precise choices. Activation functions introduce non-linear transformations, support gradient-based optimation, and enable complex decision-making.  

<br>

### *References*
SHARMA, S. (n.d.). Activation Functions in Neural Networks | by SAGAR SHARMA. Towards Data Science. Retrieved July 16, 2024, from https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6
