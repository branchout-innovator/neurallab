#### RNNs, or Recurrent Neural Networks, are neural networks specizlied in processing sequences of datas by maintaining a memory of previous inputs through directed cycles in their connections. This enables RNNs to capture temporal dependencies, making them suitable for tasks like language modeling, speech recognition, and time series analysis. 

<br>

## **How Recurrent Neural Networks work**

<br>

### Sequential Data Processing:

<br>

#### RNNs handle input sequences one element at a time. For each time step, they receive an input vector and update their internal state.

<br>

#### **Hidden State:**

The core of an RNN is its hidden state, which acts as a form of memory. At each time step, the hidden state is updated based on the current input and the previous hidden state. This allows the network to capture information about the sequence context.

<br>

#### **State Transition:**

The RNN updates the hidden state using a function, typically a tanh or ReLU activation function. This function takes the previous hidden state and the current input, combines them, and produces the new hidden state

<br>

#### **Output Generation and Training**

After updating the hidden state, the RNN can produce an output at each time step based on the current hidden state. This output can be used for tasks like classification or prediction. During training, the RNN adjusts its weights to minimize the error between its predictions and the actual target values. This is done using backpropagation through time (BPTT), which involves unfolding the network across time steps and applying the backpropagation algorithm.

<br>

## **Vocab**
* Sequence: An ordered list of data points or inputs that the RNN processes one element at a time.
* Hidden State: A vector representing the memory of the RNN, updated at each time step based on the current input and the previous hidden state.
* Activation Function: A function applied to the input and hidden state to produce the new hidden state. Common activation functions include tanh, ReLU, and sigmoid.
* Forward Propagation: The process of passing input data through the RNN to compute the hidden state and output at each time step.
* Backpropagation Through Time (BPTT): A variant of backpropagation used to train RNNs by unfolding the network through time steps and computing gradients to update weights.
* Weight Matrix: Parameters of the RNN that are learned during training. These matrices include weights for input-to-hidden connections, hidden-to-hidden connections, and hidden-to-output connections.
* Bias: An additional parameter in the RNN that helps to adjust the output along with the weighted sum of inputs.

<br>

### *Resources*
https://towardsdatascience.com/recurrent-neural-networks-rnns-3f06d7653a85 
