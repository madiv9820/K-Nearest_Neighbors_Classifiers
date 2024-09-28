# K Nearest Neighbors Classification

__K-Nearest Neighbors (KNN)__ is a simple yet powerful algorithm used for classification and regression tasks. It classifies data points based on how closely they resemble their neighbors in the feature space.

![](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcR-4HbbmYYZR2WSdktnEQOKfDqgsb930DB-wQ&s)

## How KNN Works:
1. __Data Points:__ Imagine you have a dataset with different types of fruits characterized by features such as weight, color, and size.
2. __Choosing K:__ You decide on the number of neighbors (K) to consider when making predictions. For example, if K=3, the algorithm will look at the 3 closest fruits to the new fruit you want to classify.
3. __Calculating Distances:__ When you introduce a new fruit (let’s say a new orange), KNN calculates the distance between this new fruit and all the existing fruits in the dataset.
4. __Finding Neighbors:__ The algorithm identifies the K closest fruits based on the distance calculated.
5. __Majority Voting:__ Finally, it looks at the classifications of these K neighbors and assigns the classification that appears most frequently among them to the new fruit.

## Real-Life Example:
Let’s say you are a fruit vendor trying to classify fruits based on their features to determine which fruits to display for sale.

### Step-by-Step Scenario:
1. __Dataset:__ You have a dataset containing features of various fruits:
    - Apples: (Weight: 150g, Color: Red)
    - Bananas: (Weight: 120g, Color: Yellow)
    - Oranges: (Weight: 130g, Color: Orange)
    - Grapes: (Weight: 50g, Color: Purple)

    Each fruit is labeled with its type.

2. __New Fruit:__ A customer brings in a fruit they found in their garden, and you need to classify it based on the weight and color features.

3. __KNN Classification:__
    - __New Fruit Features:__ (Weight: 135g, Color: Orange)
    - You set __K = 3__.
    - You calculate the distance from the new fruit to all the fruits in your dataset. The distances might look something like this:
        - Distance to Apple: 15g
        - Distance to Banana: 15g
        - Distance to Orange: 5g
        - Distance to Grape: 85g

4. __Finding Neighbors:__ The three closest fruits are:
    - Orange (distance 5g)
    - Apple (distance 15g)
    - Banana (distance 15g)

5. __Majority Voting:__ The K neighbors consist of one Orange, one Apple, and one Banana. Since Orange appears most frequently (1 out of 3), the algorithm classifies the new fruit as an __Orange__.

## Approach
This implementation of the K-Nearest Neighbors (KNN) algorithm follows a straightforward approach to classify data points based on their proximity to other labeled instances in the feature space. Below are the key steps taken in the implementation:

### 1. Class Definition
The `KNeighborsClassifier` class encapsulates the functionality of the KNN algorithm. It initializes with parameters such as the number of neighbors (`n_neighbors`) and the distance metric (`p`).

### 2. Fitting the Model
- The `fit` method is responsible for storing the training data. It takes two arguments: 
  - `x`: A NumPy array containing the feature vectors of the training data.
  - `y`: A NumPy array containing the corresponding labels.
- The training data is stored as instance variables for later use during predictions.

### 3. Distance Calculation
- The algorithm calculates the distance between a new data point and all points in the training dataset.
- The Lp norm is used to compute distances, allowing for customization of the distance metric by varying the `p` parameter.

### 4. Finding Nearest Neighbors
- The `__predict_One` private method is called for each instance in the test set.
- It identifies the K closest training samples using the computed distances and retrieves their corresponding labels.

### 5. Voting Mechanism
- Once the nearest neighbors are identified, a majority voting mechanism is employed to determine the class label of the new data point. The label that appears most frequently among the K neighbors is assigned to the point.

### 6. Predicting Multiple Instances
- The `predict` method handles predictions for multiple instances. It iteratively calls the `__predict_One` method for each input sample and compiles the results into an output array.

### 7. Model Evaluation
- The `score` method computes the accuracy of the model by comparing the predicted labels against the true labels of the test dataset. The accuracy is calculated as the ratio of correctly predicted instances to the total number of instances.

### Example Workflow
1. **Training Phase**: Use the `fit` method to input training data.
2. **Prediction Phase**: Call the `predict` method with test data to obtain predicted labels.
3. **Evaluation**: Assess model performance using the `score` method to compute accuracy.

This structured approach ensures that the KNN classifier is both efficient and easy to use, while maintaining clarity in the implementation of the core algorithm.

## Documentation

### Parameters:
- `n_neighbors` (int): The number of nearest neighbors to consider for classification (default is 5).
- `p` (int): The power parameter for the Minkowski distance metric (default is 2, which corresponds to the Euclidean distance).

### Methods
- `fit(x, y)`
    - Trains the KNN model by storing the training data.
    - __Parameters__:
        - `x` (np.ndarray): A 2D array where each row represents a feature vector of a training sample.
        - `y` (np.ndarray): A 1D array containing the labels corresponding to each training sample.

- `predict(x)`
    - Predicts the class labels for the provided input samples.
    - __Parameters:__
        - `x` (np.ndarray): A 2D array where each row represents a feature vector of a test sample.
    - __Returns:__
        - `np.ndarray`: A 1D array containing the predicted class labels for each test sample.

- `score(x, y)` 
    - Calculates the accuracy of the KNN model on the provided test dataset.
    - __Parameters:__
        - `x` (np.ndarray): A 2D array containing feature vectors of the test samples.
        - `y` (np.ndarray): A 1D array containing the true labels corresponding to the test samples.
    - __Returns:__
        - `np.float64`: The accuracy score as a float, representing the proportion of correctly classified samples.

### Usage
```python3 []
import numpy as np
from neighbors import KNeighborsClassifier

# Sample data
X_train = np.array([[1, 2], [2, 3], [3, 1], [6, 5], [7, 8]])
y_train = np.array([0, 0, 0, 1, 1])

X_test = np.array([[2, 2], [5, 5]])

# Create and train the model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Make predictions
predictions = knn.predict(X_test)
print(predictions)  # Output will show the predicted classes

# Evaluate the model
accuracy = knn.score(X_test, np.array([0, 1]))  # Replace with true labels
print(f'Accuracy: {accuracy:.2f}')
```

## Conclusion:
KNN is intuitive and easy to understand, making it a great choice for beginners in machine learning. However, it can become less efficient with large datasets or high-dimensional data, as the computation of distances can be costly. It's also sensitive to the choice of K and the feature scaling of the dataset, so these aspects should be carefully considered when using KNN in real-life applications.
