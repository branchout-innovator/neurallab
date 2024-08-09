#### For neural networks, data cleaning and preprocessing is used to ensure the data used is suitable for training.  These steps include handling missing data through imputation or deletion, normalizing or standardizing data, encoding categorical variables, handling imbalance data, sequence data, image data, and augmenting data to increase diversity. These preprocessing steps collectively enhance the data, making it optimal for training effective neural network models.

<br>

## **Common preprocessing techniques**

<br>

### Data Cleaning
* Fixing spelling and syntax errors
* Standardizing data sets
* Correcting mistakes such as empty fields
* Identifying duplicate data points

<br>

### Data Normalization/Standardization
* Normalization
  * Scale data to a standard range (often between 0 and 1)
* Standardization
  * scaling to unit variance (0 mean and 1 standard deviation) 
  * AKA z-score normalization

<br>

### Encoding Categorical Variables
* One-hot encoding
* Converting categorical values to binary values (0,1)
* Label encoding
  * Assigning categorical values/variables to numerical ones
  * Be cautious, as sometimes categorical values do not correlate in a specific order 

<br>

### Handling Imbalanced Data
* Sampling modification
  * Oversampling the minority class and undersampling the majority class to balance out imbalance classes

<br>

### Sequence Data (Text, Time Series)
* Tokenization
  * Breaking text into individual words
* Embeddings
  * Convert text data into numerical vectors 
  * Types: Word Embeddings (Word2Vec, GloVe) and BERT
* Time Series Handling:
  * Windowing 
  * Lag features
  * Rolling statistics

<br>

### Image Data
* Resizing and Cropping
  * Standardize image size
* Normalization
  * Scaling pixel values

<br>

## #Data Augmentation
* Strengthen the data set with random transformations	
* Ex: Rotations, Flips, Zooms

<br>

### Importance
Data preprocessing, including data cleaning, remains extremely important in neural networks, as it ensures the models are properly trained. This prevents the model from being biased and limits the amount of incorrect predications.  

<br>

### *Resources:*
https://www.obviously.ai/post/data-cleaning-in-machine-learning 

https://baotramduong.medium.com/data-preprocessing-for-neural-network-0b398b43d309 
