#### LeNet is one of the first popular convolutional neural networks, introduced by Yann LeCun, to recognize handwritten digits in images.

<br>

LeNet (LeNet-5) consists of two parts: (i) a convolutional encoder consisting of two convolutional layers; and (ii) a dense block consisting of three fully connected layers. 

<br>

In each convolutional block of LeNet, basic units consist of a convolutional layer, a sigmoid activation function, and an average pooling operation. These blocks map spatial inputs to two-dimensional feature maps, with the first convolutional layer producing 6 output channels and the second producing 16. Pooling operations reduce dimensionality by a factor of 2 via spatial downsampling. The output shape is (batch size, number of channels, height, width).
To pass the convolutional block's output to the dense block, the four-dimensional input is flattened into a two-dimensional format suitable for fully connected layers. LeNet's dense block has three fully connected layers with 120, 84, and 10 outputs, respectively, where the 10-dimensional output corresponds to the number of classification classes. Implementing LeNet with modern deep learning frameworks is straightforward, requiring only the instantiation of a Sequential block and appropriate layer chaining, using Xavier initialization.

<br>

LeNet served as the model for future CNNs, such as AlexNet, VGGNet, GoogLeNet, ResNet, and DenseNet. 

<br>

## **CNN architectures** 

<br>

#### Convolutional Neural Network architectures are networks architectures that learn straight from data that are used for deep learning. They are useful for searching for patterns in images so objects can be recognized. They’re also useful for data that is non image related such as audio and signal data. 
* The **Kernel** is a filter that extracts features from images. 
  * The formula for the kernel is [i-k]+1 where i refers to the input size and k refers to the kernel size. 
* The **stride** is part of the neural network’s filter that adjusts the movement over the video or over the image. 
  * The formula for the stride is [i-k/s]+1 where i refers to the input size and k refers to the kernel size s refers to the stride 
* The **padding** refers to the amount of pixels added to an image when being processed by the CNN. If the CNN is 0, then every pixel value added will equal 0. When the filter or kernel is used to scan an image the image will become smaller in size, so to avoid shrinking and preserve the original image size extra pixels are added outside the image. 
  * The padding formula is [i-k+2p/s]+1 where i refers to the input size and k refers to the kernel size s refers to the stride and p refers to the padding
* **Pooling** is a technique used to assist the network in recognizing features not dependent on their location in the image and for generalizing features that have been retracted by convolutional filters. 
* **Flattening** is used when converting 2D arrays from pooled feature maps into a singular continuous linear vector. 

<br>

### Layers used to build CNN
What makes CNNs different from other neural networks is their surpassing performance in relation to image, speech, or audio signal inputs. 
The 3 main types of CNN layers are 
* Convolutional layer 
  * The first layer, used to extract features from the input images. A filter or kernel is used to extract features from the input image. 
  * The formula for the convolutional layer is  W-F+2PS+1
* Pooling layer 
  * The goal of this layer is to reduce the size of convolved feature map to lower computational costs. 
  * The formula for the pooling layer is W-FS+1
* Fully connected (FC) layer 
  * Comprised of the weights and biases and neurons and is used to join the neurons between 2 layers 
  * The **dropout layer** is another characteristic of CNNs, it is a mask that voids the contribution of neurons to the next layer and doesn’t modify other neurons. 
  * The **activation function** determines whether or not the neuron should be activated. There are a few types of activation functions, such as Sigmoid, tanH, Softmax, RelU

<br>

### *References*
https://d2l.ai/chapter_convolutional-neural-networks/lenet.html 

https://medium.com/@draj0718/convolutional-neural-networks-cnn-architectures-explained-716fb197b243
