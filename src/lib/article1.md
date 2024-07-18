## What are Activation Functions? (Neural Nets)

Activation functions are mathematical functions applied to the output of a neuron (like a filter). They introduce non-linearity (where input changes are not proportional to output changes) to the model, which allows it to learn and predict patterns more accurately.
<br>
![Image0](/articleimages/image0)
<br>

## When to use Different Activation Functions:

#### **Output Layers:**

### Binary Classification:

The Sigmoid function is used for the output layer because it outputs a value from 0 to 1:
σ(x) = 1/(1 + e<sup>-x</sup>)

### Multi-class Classification:

The Softmax function is used because it converts the outputs to probabilities that sum to 100%:
softmax(x<sub>i</sub>) = e<sup>x<sub>i</sub></sup>/∑<sub>j</sub>e<sup>x<sub>j</sub></sup>

### Regression:

Either linear activation functions or no activation function is used.

#### **Hidden Layers:**

### ReLU:

Simple and effective activation function that helps to alleviate the vanishing gradient problem (derivatives of positive inputs are 1):
ReLU(x) = max(0,x)

### Tanh:

Similar to sigmoid but with a range of [-1, 1]. Can be more effective at training the network because of a larger gradient:
tanh(x) = e<sup>x</sup>-e<sup>-x<sup>/e<sup>x</sup>+e<sup>-x</sup>
<br>

<br>
# Chatgpt response: 
Imagine your brain as a big, super-smart machine with lots of tiny switches inside. These switches help you decide what to do based on the information you get, like deciding whether to jump when you see a puddle or to say "hello" when you see a friend.
## Activation Functions
In a computer's brain (like in robots or apps that learn), there are also tiny switches called activation functions. These switches help the computer make decisions by turning "on" or "off" based on the information it receives. Here are some examples of how these activation functions work:
- ReLU (Rectified Linear Unit):
- Imagine a switch that stays off (at 0) if it gets a negative signal, but turns on (to the same value as the signal) when it gets a positive signal.
- It's like if you decided only to do something if it was fun (positive), and  you'd do exactly how much fun it seemed.
- Sigmoid:
- This switch smoothly turns on more and more as the signal gets bigger, but it never fully reaches 1, and never fully turns off to 0.
- Think of it like a dimmer switch for a light; as you turn it, the light gets brighter slowly, but it never gets completely dark or super bright.
- Tanh (Hyperbolic Tangent):
- This one is like the sigmoid but a bit different: it can handle both positive and negative signals, turning on for positive ones and turning off for negative ones, and it does it more smoothly.
- Imagine a balance scale: it can tip to one side for good things and to the other for bad things, showing how strong each is.
Why Are They Important?
Activation functions help the computer's brain understand and decide things better by handling information in smart ways. Just like how you use different switches or decisions based on what you're doing, computers use these activation functions to learn and make choices more accurately.
