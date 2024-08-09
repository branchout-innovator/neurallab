## Cross-validation techniques

#### Cross-validation is a technique used to assess how well a neural network generalizes new data. This helps ensure that a neural network doesn’t just memorize a specific data set and instead learns and performs well given various data subsets. There are several cross-validation techniques used with neural networks: 

<!-- ![Image0](/static/articleimages/cross_validation/cross_val_kfold) -->

* K-fold and variations: there are three key steps in this cross-validation technique: 
  * Entire dataset is split into “k” subjects, or folds. If folds often preserve the proportion of class distributions (category or set of data), then it’s called stratified K-fold cross-validation
  * Model is trained and evaluated “k” times, but in each iteration, one fold is held out as a “validation” and the model is instead tested on the remaining “k-1” folds, after training the model is tested on the held-out fold to evaluate the model and avoid overfitting (learns old data set “too well” leading to poor performance on new data)
  * Performance is calculated based on performance metrics (ie. accuracy, precision, recall, etc.) for each “k” set, then the average of these metrics are calculated and the k-score is given
  * LOOCV (leave one out cross validation) is an extreme k-fold in which only one sample is used as a test set and the rest are used to train the model (ie. k=n)

<!-- ![Image0](/static/articleimages/cross_validation/cross_val_tseries) -->
<br>

* Time-series cross-validation is tailored for time-series data where order of observations is crucial. It respects the sequential nature of data. They’re used to make predictions. Here are some key steps to do this: 
  * Initial split: divide time series into training and test set, with training set for earlier observations and test set for recent observations
  * Expanding window: training set grows with each iteration, (ie. after testing on the initial training set, the next iteration is new data+initial training set, etc.)
  * Train model: For each iteration, train model on training set and evaluate it on test set to see how well the model learns
  * Aggregation: after performing cross-validation across all sets, measure performance (MSE, MAE, etc.)	 	

