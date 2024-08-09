## Overfitting and Underfitting

### Overfitting
#### Overfitting occurs when the algorithm aligns too closely to the training data, resulting in a model which fails to make accurate conclusions from other data sets. In short, the model becomes overly complicated and solely tailored to the specific training data set patterns, causing the model to become “overfitted,” unable to properly generalize and perform intended tasks.

<br>

#### Indicators of Overfitting: 
* High training accuracy
* Low validation test accuracy
* Overly complex models
* High variance

## Detection Method: K-fold cross-validation

<!-- ![Image0](/static/articleimages/overfitting_and_underfitting) -->

#### In k-fold cross-validation, data is split equally into subsets, known as “folds.” In each interaction, one of the k-folds will act as the test set, while the remaining folds train the model. Once all interactions and scores have been evaluated, the scores are averaged to determine an overall assessment.

<br>

#### Prevention Methods
* Regularization
* Early stopping
* Ensemble methods (bagging & boosting)

## Underfitting
	
#### Underfitting occurs when the model algorithm is too simple to represent the entirety of the data structure, resulting in high error rate and poor performance shown in both the training data and test data. Similar to overfitting, underfitting is an inaccurate method in determining the dominant trend within the data.

#### Indicators of Underfitting
* Low training accuracy & low validation test accuracy
* Simple model
* High bias
* Low variance

<br>

#### Prevention Methods
* Decrease regularization
* Increase training duration
* Feature selections (increased complexity)

<br>

## Finding a Balance
#### By implementing a combination of such techniques and methods, the most ideal mastery is to find a balance between overfitting and underfitting. The final product results in a model able to use sufficient data and insight to make accurate predictions applicable to real world settings.

<br>

#### *References*
What is Overfitting? (n.d.). IBM. Retrieved August 5, 2024, from https://www.ibm.com/topics/overfitting
What Is Underfitting? (n.d.). IBM. Retrieved August 5, 2024, from https://www.ibm.com/topics/underfitting (Hardesty, 2017)
Navigating the Learning Curve: Understanding Overfitting and Underfitting in Machine Learning. (2024, June 7). Medium. Retrieved August 5, 2024, from https://medium.com/@caterine/navigating-the-learning-curve-understanding-overfitting-and-underfitting-in-machine-learning-bcdf0dd03be6

