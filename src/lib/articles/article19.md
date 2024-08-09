#### VGG-16 is a 16-layer deep neural network that can classify images into 1000 object categories (ie. keyboard, pencil, mouse, etc.) This results in the network learning rich feature representations for many images. 

<br>

## **Implementation using python**

<br>

### Importing the libraries
* import numpy as np
* import pandas as pd
* import matplotlib.pyplot as plt
* from glob import glob
* import sklearn.metrics as metrics
* import tensorflow as tf
* from keras.preprocessing.image import ImageDataGenerator
* from tensorflow.keras.models import Model
* from tensorflow.keras.layers import Flatten, Dense
* from tensorflow.keras.applications import VGG16

<br>

### Initialization
* num_classes=3
* IMAGE_SHAPE = [224, 224]
* batch_size=32 #change for better accuracy based on your dataset
* epochs = 5 #change for better accuracy based on your dataset

<br>

### Load and compile the VGG model

#### vgg = VGG16(input_shape = (224,224,3), weights = ‘imagenet’, include_top = False)
#### for layer in vgg.layers:
#### layer.trainable = False
#### x = Flatten()(vgg.output)
#### x = Dense(128, activation = ‘relu’)(x)
#### x = Dense(64, activation = ‘relu’)(x)
#### x = Dense(num_classes, activation = ‘softmax’)(x)
#### model = Model(inputs = vgg.input, outputs = x)
#### model.compile(loss=’categorical_crossentropy’, optimizer=’adam’, metrics=[‘accuracy’])

<br>

### Image data generator
* Can be used to rotate images anywhere from 0-360 degrees through providing an integer range value through the rotation range argument 
* trdata = ImageDataGenerator()
* train_data_gen = trdata.flow_from_directory(directory=”Train”,target_size=(224,224), shuffle=False, class_mode=’categorical’)
* tsdata = ImageDataGenerator()
* test_data_gen = tsdata.flow_from_directory(directory=”Test”, target_size=(224,224),shuffle=False, class_mode=’categorical’)
* Found 385 images belonging the 3 classes 
* Found 138 images belonging to 3 classes

<br>

### Train the model
* training_steps_per_epoch = np.ceil(train_data_gen.samples / batch_size)
* validation_steps_per_epoch = np.ceil(test_data_gen.samples / batch_size)
* model.fit_generator(train_data_gen, steps_per_epoch = training_steps_per_epoch, validation_data=test_data_gen, validation_steps=validation_steps_per_epoch,epochs=epochs, verbose=1)
* print(‘Training Completed!’)
#### 13/13 [==============================] - 98s 8s/step - loss: 62.0823 - accuracy: 0.4364 - val_loss: 33.8267 - val_accuracy: 0.3406 
#### Epoch 2/5
#### 13/13 [==============================] - 98s 8s/step - loss: 14.0897 - accuracy: 0.6805 - val_loss: 6.7669 - val_accuracy: 0.7319 
#### Epoch 3/5
#### 13/13 [==============================] - 99s 8s/step - loss: 0.9217 - accuracy: 0.9299 - val_loss: 4.7415 - val_accuracy 0.7899 
#### Epoch 4/5
#### 13/13 [==============================] - 101s 8s/step - loss: 2.1274 - accuracy: 0.8987 - val_loss: 2.7657 - val_accuracy: 0.8768 
#### Epoch 5/5
#### 13/13 [==============================] - 102s 8s/step - loss: 3.3716 - accuracy: 0.8805 - val_loss: 2.6674 val_accuracy: 0.8841
#### Training Completed!

<br>

### Accuracy
* Y_pred = model.predict(test_data_gen, test_data_gen.samples / batch_size)
* val_preds = np.argmax(Y_pred, axis=1)
* import sklearn.metrics as metrics
* val_trues =test_data_gen.classes
* from sklearn.metrics import classification_report
* print(classification_report(val_trues, val_preds))
|  | Precision | Recall | f1-score | Support |
|--|--------------|----------|-------------|-------------|
| 0 | 0.90 |    0.90 |        0.90 |          59|
|---------------------|--------------|----------|-------------|-------------|
| 1 | 0.91 | 0.96 | 0.94 | 53 |
|---------------------|--------------|----------|-------------|-------------|
| 2 | 0.78 | 0.69 | 0.73 | 26 |
|---------------------|--------------|----------|-------------|-------------|
| Accuracy |  |  | 0.88 | 138 |
|---------------------|--------------|----------|-------------|-------------|
| Macro Avg | 0.86 | 0.85 | 0.86 | 138 |
|---------------------|--------------|----------|-------------|-------------|
| Weighted Avg | 0.88 | 0.88 | 0.88 | 138 |

<br>

### Confusion matrix
* A table usually used to describe the performance of a classification model on a data set where the true values are known 
* Y_pred = model.predict(test_data_gen, test_data_gen.samples / batch_size)
* val_preds = np.argmax(Y_pred, axis=1)
* val_trues =test_data_gen.classes
* cm = metrics.confusion_matrix(val_trues, val_preds)
#### Array([[53, 3, 3],
#### [ 0, 51, 2],
#### [ 6, 2, 18]], dtype=int64)

<br>

### Save the model** 
* keras_file=”Model.h5"
* tf.keras.models.save_model(model,keras_file)

<br>

### Prediction using new image
* create new file test.py and run this file
* from tensorflow.keras.models import load_model
* from tensorflow.keras.preprocessing import image
* from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
* import numpy as np
* load saved model
  * model = load_model(‘Model.h5’)
  * img_path = ‘fresh.jpg’
  * img = image.load_img(img_path, target_size=(224, 224))
  * x = image.img_to_array(img)
  * x = np.expand_dims(x, axis=0)
  * x = preprocess_input(x)
  * preds=model.predict(x)
* create a list containing the class labels
  * class_labels = [‘Apple’,’Banana’,’Orange’]
* find the index of the class with maximum score
  * pred = np.argmax(preds, axis=-1)
* print the label of the class with maximum score
  * print(class_labels[pred[0]])

<br>

### *References*
Dharmaraj. (2022b, April 3). Image classification and prediction using transfer learning. Medium. https://medium.com/@draj0718/image-classification-and-prediction-using-transfer-learning-3cf2c736589d 


