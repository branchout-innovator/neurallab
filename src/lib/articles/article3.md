#### Loss functions are used to determine the difference between the predicted values and the actual values of a model. The smaller the error, or loss, the more accurate the model. The goal of a model is to minimize this loss. Different loss functions are used for different tasks:

<br>

## **Regression Loss Functions:**

### Mean Squared Error (MSE):

MSE measures the average of the squares of the errors. Because the error is squared, it is used because it penalizes larger errors. MSE functions are differentiable (good for gradient descent optimization) and easy to read. However, it is also greatly influenced by outliers and can be difficult to compare to other data sets depending on the scale.
MSE = (1/n)\*Σ(y~i~-ŷ~i~)^2^, (y~i~: actual value, ŷ~i~: predicted value)

<br>

### Mean Absolute Error (MAE):

MAE measures the average of the absolute errors. The errors are not squared, so it is used because outliers do not as heavily influence it. MAE functions are also easy to read and give equal weight to all errors. However, it is not differentiable (at 0) and doesn’t penalize large errors.
MAE = (1/n)\*Σ|y~i~-ŷ~i~|, (y~i~: actual value, ŷ~i~: predicted value)

<br>

### Huber Loss:

Huber loss combines MSE and MAE (quadratic for small errors, linear for large). It is used because it handles outliers effectively, is differentiable everywhere, and provides a balanced measurement, combining both MSE and MAE. However, it is more complex, and choosing the δ value can be challenging.
L~δ~(y~i~-ŷ~i~) = {(1/2)(y~i~-ŷ~i~)for |y~i~-ŷ~i~| ≤ δ
δ(|yi-ŷi| - (1/2)δ)for |y~i~-ŷ~i~| > δ
(δ determines where the function changes)

<br>

![Image0](/articleimages/image0)

<br>

## **Classification Loss Functions:**

### Binary Cross-Entropy Loss (Log Loss):

Binary cross-entropy loss functions are used for binary classification problems. It is used for its clarity, smooth gradient, and flexible weighting. However, it is sensitive to outliers, heavy computationally, and is undefined for certain values (logarithmic).
Loss = -(1/n)Σ[y~i~log(ŷ~i~) + (1 - y~i~)log(1 - ŷ~i~)] (y~i~: actual binary label, ŷ~i~: predicted probability).

<br>

### Categorical Cross-Entropy Loss:

Categorical cross-entry loss functions are used for multi-class classification problems. It is used because it can handle multiple classes with probabilities, scalability, and smooth gradients. However, it is sensitive to outliers, undefined for certain values (logarithmic), and is sensitive to imbalanced datasets.
Loss = -ΣΣy~i~c\*log(ŷ~i~c), (C is # of classes, yic is indicator if c is correct for i, and ŷic is predicted probability of c for i)

<br>

### Hinge Loss:

Hinge loss functions are used for training support vector machines (SVMs). It is used because it maximizes the margin, is robust to outliers, and is effective for linear models. However, it is non-differentiable at the hinge point, difficult to use with multiple classes, and is sensitive to class imbalance.
Loss = Σmax(0, 1 - y~i~\*ŷ~i~)

<br>

![Image1](/articleimages/image1)

<br>

## **Conclusions:**

Loss functions are important because different loss functions can display accuracies for different aspects of models. You can also compare model accuracies to others through loss functions. Optimization algorithms, such as gradient descent, can help minimize the loss function.

<br>

### *References*
Yathish, V. (2022, August 4). Loss Functions and Their Use In Neural Networks. Towards Data Science. Retrieved July 16, 2024, from https://towardsdatascience.com/loss-functions-and-their-use-in-neural-networks-a470e703f1e9
