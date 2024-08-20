## **What are Pooling Layers?**

<br>

Pooling layers summarize the presence of features by reducing the spatial dimensions of the input feature maps. They exist in convolutional neural networks after the convolutional layers. The pooling layers in a neural network allow for only the important features to be kept. 

<br>

### **Pooling methods**

### Average Pooling
Computes the average value of each patch of the feature map. This is less commonly used than max pooling.
Example: For a 2x2 window, the average value of the values in that window is computed.
### Max pooling
Selects the maximum value from each patch of the feature map. This helps in capturing the most prominent features.
Example: For a 2x2 window, the maximum value in that window is selected.
### Global pooling
Applies a pooling operation over the entire feature map, reducing it to a single value per feature map.
<!-- ![Image1](/static/articleimages/activation_functions/image1) -->

<br>

### **Benefits of Pooling Layers**
Convolutional layers provide precise feature maps, meaning small movements in input would produce entirely different outputs. However, pooling methods sort through and keep only the most important features. This also reduces the number of parameters and the complexity of future layers in the model. 

<br>

### **References**
Brownlee, J. (2019, July 5). A Gentle Introduction to Pooling Layers for Convolutional Neural Networks - MachineLearningMastery.com. Machine Learning Mastery. Retrieved July 18, 2024, from https://machinelearningmastery.com/pooling-layers-for-convolutional-neural-networks/

Pooling Layers. (n.d.). Dremio. Retrieved July 18, 2024, from https://www.dremio.com/wiki/pooling-layers/





