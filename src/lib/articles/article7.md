<br>

Convolutional layers are a key innovation in convolutional neural networks (CNNs). They are the first layer in CNNs, followed by pooling layers and fully-connected (FC) layers. Most computation occurs in the convolutional layers, where data, often with grid-like topology is processed. Convolutional layers analyze these data sets using inputs, a filter, and a feature map.

![Image0](/static/articleimages/activation_functions/image0)

<br>

### Steps in Convolutional Layers
* The input will represented in corresponding data structures (ex. matrices) 
* The filter will created with the desired parameters in the feature
* A dot product will be calculated between the input and filter,
* The dot product value will be added to an output array. 
* The filter will shift and repeat the process across the entire input
* All dot products will be displayed, creating a feature map. 

<br>

### Weights

The parameters stay the same as the filter progresses through the data sets. However, some can be adjusted as training for the model continues. Some parameters must be defined before the training starts. These include: 
* Number of filters 
* Stride, or distance the filter moves as it shifts
* Zero-padding, or how the filter responds to various sizes that do not match the original data set

<br>

### Layer Complexity and Efficiency
As the amounts of layers increase the am\nalysis gets more and more complicated. Convolutional layers add to the efficiency of convolutional neural networks as the numerical data generated (dot products) can be easily analyzed further. The first convolutional layer is often one of many. 

<br>

### *References*
Darwish, D. (2024, January 22). Improving Techniques for Convolutional Neural Networks Performance. European Journal of Electrical Engineering and Computer Science. Retrieved July 17, 2024, from https://www.ejece.org/index.php/ejece/article/view/596
What are Convolutional Neural Networks? (n.d.). IBM. Retrieved July 17, 2024, from https://www.ibm.com/topics/convolutional-neural-networks





