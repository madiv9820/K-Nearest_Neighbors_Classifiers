import numpy as np
from collections import Counter

class KNeighborsClassifier:
    def __init__(self, n_neighbors = 5, p = 2) -> None:
        """
        Initializes the KNeighborsClassifier with specified number of neighbors and distance metric.

        Parameters:
        n_neighbors (int): Number of nearest neighbors to consider (default is 5).
        p (int): The power parameter for the Minkowski distance (default is 2, which corresponds to Euclidean distance).
        """
        self.__k = n_neighbors  # Store the number of neighbors
        self.__p = p            # Store the distance metric

    def __repr__(self) -> str:
        """Returns a string representation of the KNN classifier."""
        return f'neighbors.K_Nearest_Neighbors(n_neighbors: {self.__k}, p: {self.__p})'
    
    def __predict_One(self, x: np.ndarray):
        """
        Predicts the class label for a single input sample.

        Parameters:
        x (np.ndarray): A 1D array representing the feature vector of a single test sample.

        Returns:
        The predicted class label for the input sample.
        """
        # Calculate the distances from the input sample to all training samples
        distances = np.sum((np.abs(self.__Input - x)) ** self.__p, axis=1)
        
        # Get the indices of the K nearest neighbors
        top_k_indices = np.argsort(distances)[:self.__k]
        
        # Retrieve the labels of the K nearest neighbors
        output_top_k = self.__Output[top_k_indices]
        
        # Return the most common class label among the neighbors
        return Counter(output_top_k).most_common(1)[0][0] 

    def fit(self, x: np.ndarray, y: np.ndarray):
        """
        Fits the KNN model by storing the training data.

        Parameters:
        x (np.ndarray): A 2D array where each row represents a feature vector of a training sample.
        y (np.ndarray): A 1D array containing the labels corresponding to each training sample.
        """
        self.__Input = x  # Store the training feature vectors
        self.__Output = y  # Store the training labels

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predicts the class labels for the provided input samples.

        Parameters:
        x (np.ndarray): A 2D array where each row represents a feature vector of a test sample.

        Returns:
        np.ndarray: A 1D array containing the predicted class labels for each test sample.
        """
        # Initialize an output array to store predictions
        outputs = np.zeros(x.shape[0], dtype=self.__Output.dtype)
        
        # Iterate over each sample in the test set and predict its label
        for index in range(x.shape[0]):
            outputs[index] = self.__predict_One(x[index])  # Predict label for the current sample
            
        return outputs  # Return the array of predicted labels
    
    def score(self, x: np.ndarray, y: np.ndarray) -> np.float64:
        """
        Calculates the accuracy of the KNN model on the provided test dataset.

        Parameters:
        x (np.ndarray): A 2D array containing feature vectors of the test samples.
        y (np.ndarray): A 1D array containing the true labels corresponding to the test samples.

        Returns:
        np.float64: The accuracy score as a float, representing the proportion of correctly classified samples.
        """
        predictions = self.predict(x)  # Get predictions for the test samples
        # Calculate and return the accuracy as the ratio of correct predictions to total samples
        return np.sum(predictions == y) / len(y)