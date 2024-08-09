## **Normalization**

Normalization is a technique that can be used in data preprocessing in order to modify features that are varied and scale them in the common data set. Normalization can be beneficial when making and dealing with precise models, analyzing data, and reducing the impact of feature values on the data set. Transformation squeezes the n-dimensional data into an n-dimensional unit hypercube. Normalization cannot adjust to outliers so it is best used when there is no outlier present. Normalization can be applied during data preparation so numeric column values can be changed to use a standard scale. Normalization is not required for every data set in the model, it is only required when machine learning features have different ranges. 

<br>

2 of the most common normalization techniques used are min and max scaling, which assists the data set when shifting and rescaling attribute values in order for them to range from 0-1.

## **Standardization scaling**
Standardization is a scaling technique in which the values are centered around the mean with a unit of standard deviation. **Z score normalization** is the change of features by subtracting from the mean and dividing by standard deviation. Unlike normalization standardization is not affected by outliers due to the lack of a predefined range of transformed features.

Certain machine learning algorithms such as K-nearest neighbor or using linear models and interpreting their coefficient benefit from both standardization and normalization. 

### *References* 
https://medium.com/@meritshot/standardization-v-s-normalization-6f93225fbd84

