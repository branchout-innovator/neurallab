CNNs (Convolutional Neural Networks) are usually used for image recognition. They essentially break each image up into small pieces and recognize patterns. They’re very useful in image pattern recognition and audio, time series, etc. classification.

Normal neural networks (breaking each image into grids and assigning each pixel to a specific node) are impractical in image recognition because of the sheer amount of pixels in most images and they cannot recognize similar images. 

By contrast, CNNs tolerate shifts in image pixels, recognize correlation (filter), can tolerate shifts in image and reduce input node number (convolution and max pooling). CNNs work in several steps: 
* First, they create a filter (kernels; smaller squares); pixel intensity in the filter is determined by backpropagation 
Filters are applied to the input image (convolved), the dot product between the filter and image is calculated, a bias term is applied, and the final value is put on a feature map, which is usually ran through an activation function (ReLU)
* Next, another filter is applied to the feature map; max pooling (maximum value of each region, “best” spot of filter matching) or mean pooling (average value of each region is calculated) is applied
* Finally, the resulting simplified pooled image is plugged into a neural network, classifying the image
