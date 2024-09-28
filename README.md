# K Nearest Neighbors Classification

__K-Nearest Neighbors (KNN)__ is a simple yet powerful algorithm used for classification and regression tasks. It classifies data points based on how closely they resemble their neighbors in the feature space.

![](data:image/png;base64,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)

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
