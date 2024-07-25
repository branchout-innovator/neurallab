 #### CNNs, or Convolutional Neural Networks, are a type of neural net designed specifically for visual data, such as images. For example, they can be used to identify objects within images, locate objects within images, divide images into sections, and identify individuals through facial recognition. 

<br>

#### **CNNs are composed of 3 main layers:**
* Convolutional Layers
* Pooling Layers
* Fully Connected Layers

<br>

#### **Steps for training CNNs:**
* Gather data, ensuring images are correctly labeled
* Resize and normalize data
* Augmentation (transforming images to increase diversity and reduce overfitting)
* Define the input shape
* Add convolutional layers (filters (kernels) which slide across images summing up elements to produce single value, kernel size - dimensions of the filter, where smaller kernels capture finer details, strides determine how many units the filter moves at a time)
* Add pooling layers (Max Pooling - takes maximum value from patches of feature map, Average Pooling - takes average value from patches of feature map)
* Add 1+ dense layers near the end
* For the classification output layer, add a dense layer with a number of neurons equal to the number of classes (with softmax activation)
* Compile model - define loss function (categorical cross-entropy for classification), optimizer (Adam, SGD, etc. ), and metrics (evaluates performance)
* Training the model - batch size (number of samples processed during each parameter update) and epochs (number of complete passes through the dataset)
* Evaluate the model - Use a validation data set to check the model for overfitting, then test with a new data set after. 
* Adjust the learning rate to improve convergence, regularize the model, and experiment with other hyperparameters to improve the model. 

<br>

### *References*

https://www.ibm.com/topics/convolutional-neural-networks
https://www.techtarget.com/searchenterpriseai/definition/convolutional-neural-network