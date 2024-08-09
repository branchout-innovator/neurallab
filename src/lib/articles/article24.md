## Model Training Techniques

#### Neural network training requires numerous iterations. Each time, a pass through a model’s layers is executed to compute an output for each training example, then a pass backwards is executed to see how each parameter affects the final output. This is calculated by computing a “gradient” in response to each parameter. The average gradient is passed through an optimization algorithm which shows new parameters, and gradually, over many iterations, the model improves. 

<!-- ![Image0](/static/articleimages/model_training_technique) -->

#### There are various steps needed to train a neural network to make accurate predictions and classifications:
1. Defining the neural network: specify nodes and layers and choose an activation function (ie. sigmoid, ReLU, etc.) for the model 
2. Initialize weights; these weights control strength of connections between neurons and capture relationships between inputs and outputs, are first randomly initialized, then learned based on past data
3. Forward pass; input data is passed through the network, each layer is computed using weighted some of input and activation functions, and output is the final layer
4. Loss computation: error of function is computed using a loss function which determines how far the predictions are from target, loss functions include MSE, MAE, Huber, etc. 
5. Backwards pass (backpropagation): gradient is calculated (derivative vector) with respect to each weight, meaning how much each weight contributes to error is calculated, then gradient descent is calculated (weights are updated based on computed gradients to minimize loss using algorithms such as SGD)
6. Training cycles are repeated; 1 epoch is a full training cycle; however, training data may also be divided into smaller batches which helps maintain efficiency