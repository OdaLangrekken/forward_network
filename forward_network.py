import numpy as np
import time

class ForwardNetwork:
    def __init__(self, data, target):
        self.data = self.add_bias(data)
        self.target = target
        self.num_inputs = data.shape[1]
        self.num_features = data.shape[0]
        self.weights = {}
        self.activations = {}
        self.activations[0] = self.data
        self.errors = {}
        self.error_matrix = {}
        
    def initialize_weights(self, size_layer, size_prev_layer, n):
        # Use the He initializing for weights. Weights drawn from guassian dist with std sqrt(2/n)
        return np.random.randn(size_layer, size_prev_layer) * np.sqrt(2/n)
        
    def add_layer(self, size_layer):
        """
        Adds layer of size size_layer (bias excluded)
        """
        # Check if this is first layer
        if len(self.weights) == 0:
            self.weights[0] = self.initialize_weights(size_layer, self.num_features + 1, self.num_inputs)
            self.error_matrix[0] = np.zeros((size_layer, self.num_features + 1))
        else:
            counter = len(self.weights)
            size_prev_layer = self.weights[counter - 1].shape[0]
            self.weights[counter] = self.initialize_weights(size_layer, size_prev_layer + 1, self.num_inputs)
            self.error_matrix[counter] = np.zeros((size_layer, size_prev_layer + 1))
            
    def add_bias(self, x):
        x = np.concatenate((np.ones((x.shape[1], 1)).T, x), axis=0)
        return x
            
    def transform(self, a, w):
        """
        Calculates matrix product w^T*x
        """
        return np.dot(w, a)
    
    def activation(self, z):
        return self.sigmoid(z)
        #return np.maximum(z, 0)

    def sigmoid(self, z):
        sigmoid = 1 / (1 + np.exp(-z))
        return sigmoid
            
    def forward_prop(self, x):
        """
        Propagates input through all layers and return output of final layer
        """
        current_layer = 0
        num_layers = len(self.weights)
        while current_layer < num_layers:
            #Get weights for current layer
            weights = self.weights.get(current_layer)
            inputs = self.activations.get(current_layer)
            z = self.transform(inputs, weights)
            a = self.sigmoid(z)
            current_layer += 1
            #If layer is the not the last layer we add bias
            if current_layer != num_layers:
                a = self.add_bias(a)
            self.activations[current_layer] = a
        return a
            
    def calculate_cost(self, y_pred, y):
        m = len(y)    #number of samples
        loss = y*np.log(y_pred) + (1 - y)*np.log(1 - y_pred)  # Should add regularization
        return -np.sum(loss)/m
    
    def sigmoid_derivative(self, a):
        return a * (1 - a)
    
    def back_prop1(self, x, y):
        current_layer = len(self.weights)
        a = self.activations.get(current_layer)
        error = (a - y) * self.sigmoid_derivative(a)
        self.errors[current_layer] = error.T
        current_layer -= 1
        while current_layer > 0:
            a = self.activations.get(current_layer)
            weights = self.weights.get(current_layer)
            error_prev = self.errors.get(current_layer + 1)
            if current_layer == len(self.weights) - 1:
                error = np.dot(weights.T, error_prev)
            else:
                error = np.dot(weights.T, error_prev[1:])
            self.errors[current_layer] = error
            current_layer -= 1
            
    def back_prop(self, x, y):
        current_layer = len(self.weights)
        a = self.activations.get(current_layer)
        error = (a - y)
        self.errors[current_layer] = error
        current_layer -= 1
        while current_layer > 0:
            a = self.activations.get(current_layer)
            weights = self.weights.get(current_layer)
            error_prev = self.errors.get(current_layer + 1)
            error = np.dot(weights.T, error_prev)*self.sigmoid_derivative(a)
            self.errors[current_layer] = error
            self.error_matrix[current_layer] = np.dot(error_prev, a.T)
            current_layer -= 1
        self.error_matrix[0] = np.dot(self.errors[1], self.activations[0].T)
        for i in range(len(self.weights) - 1):
            self.error_matrix[i] = self.error_matrix[i][1:, :]
            
    def update_weights(self, learning_rate, reg_coef):
        current_layer = 0
        while current_layer < len(self.weights):
            m = self.num_inputs
            size = self.weights[current_layer].shape
            gradient = np.zeros(size)
            gradient[:, 0] = 1/m * self.error_matrix[current_layer][:, 0]
            gradient[:, 1:] = 1/m * (self.error_matrix[current_layer][:, 1:] + reg_coef*self.weights[current_layer][:, 1:])
            self.weights[current_layer] -= learning_rate * gradient
            current_layer += 1
        
    def train(self, num_epochs = 100, learning_rate = 1, reg_coef = 0):
        start = time.time()
        for i in range(num_epochs):
            self.forward_prop(self.data)
            self.back_prop(self.data, self.target)
            self.update_weights(learning_rate, reg_coef)
            if i % 100 == 0:
                print(f'Epochs done: {i}/{num_epochs}')
        end = time.time()
        if (end - start)/60 < 1:
            print(f'Time elapsed: {end - start:.2f} s')
        else:
            print(f'Time elapsed: {(end - start)/60:.2f} m')
            
    def predict(self, x):
        self.activations[0] = self.add_bias(x)
        self.forward_prop(x)
        preds = self.activations[len(self.activations) - 1]
        return np.argmax(preds, axis = 0)
    
    def predict_proba(self, x):
        self.forward_prop(x)
        preds = self.activations[len(self.activations) - 1]
        return preds
    
    def accuracy(self, y_pred, y):
        acc = 0
        for i in range(len(y)):
            if y_pred[i] == y[i]:
                acc += 1
        return acc/len(y)
           
           
