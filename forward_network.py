import numpy as np

class ForwardNetwork:
    def __init__(self, data, target):
        self.data = self.add_bias(data)
        self.target = target
        self.num_inputs = data.shape[0]
        self.num_features = data.shape[1]
        self.weights = {}
        self.activations = {}
        self.activations[0] = self.data
        self.errors = {}
        
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
        else:
            counter = len(self.weights)
            size_prev_layer = self.weights[counter - 1].shape[0]
            self.weights[counter] = self.initialize_weights(size_layer, size_prev_layer + 1, self.num_inputs)
            
    def add_bias(self, x):
        x = np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)
        return x
            
    def transform(self, x, w):
        """
        Calculates matrix product w^T*x
        """
        # Add bias term 
        return np.dot(x, w.T)
    
    def activation(self, z):
        return self.sigmoid(z)
        #return np.maximum(z, 0)

    def sigmoid(self, z):
        sigmoid = 1 / (1 + np.exp(-z))
        return sigmoid
            
    def forward_prop(self, x, current_layer = 0):
        """
        Propagates input through all layers and return output of final layer
        """
        num_layers = len(self.weights)
        while current_layer < num_layers:
            #Get weights for current layer
            weights = self.weights.get(current_layer)
            inputs = self.activations.get(current_layer)
            z = self.transform(inputs, weights)
            a = self.sigmoid(z)
            current_layer += 1
            #If layer is the last layer we want to return the output
            if current_layer != num_layers:
                a = self.add_bias(a)
            self.activations[current_layer] = a
        return a
            
    def calculate_cost(self, x, y):
        m = len(y)    #number of samples
        y_pred = self.sigmoid(self.forward_prop(x))
        loss = y*np.log(y_pred) - (1 - y)*np.log(1 - y_pred)
        return np.sum(loss)/m
    
    def sigmoid_derivative(self, a):
        return a * (1 - a)
    
    def back_prop(self, x, y):
        current_layer = len(self.weights)
        a = self.activations.get(current_layer)
        error = (a - y) * self.sigmoid_derivative(a)
        self.errors[current_layer] = error.T
        current_layer -= 1
        while current_layer >= 0:
            a = self.activations.get(current_layer)
            weights = self.weights.get(current_layer)
            error_prev = self.errors.get(current_layer + 1)
            if current_layer == len(self.weights) - 1:
                error = np.dot(weights.T, error_prev)
            else:
                error = np.dot(weights.T, error_prev[1:])
            self.errors[current_layer] = error
            current_layer -= 1
            
    def update_weights(self):
        current_layer = 0
        while current_layer < len(self.weights):
            if current_layer == len(self.weights) - 1:
                grad = np.dot(self.errors[current_layer + 1], self.activations[current_layer][:, 1:])
                #print(np.dot(self.errors[current_layer + 1], self.activations[current_layer][:, 1:]).shape)
                self.weights[current_layer][:, 1:] -= 0.01 * np.dot(self.errors[current_layer + 1], self.activations[current_layer][:, 1:])
            else:
                self.weights[current_layer][:, 1:] -= 0.01 * np.dot(self.errors[current_layer + 1][1:, :], self.activations[current_layer][:, 1:])
            current_layer += 1
            
            
    def train(self, num_epochs):
        costs = []
        weight_test = []
        for i in range(num_epochs):
            self.forward_prop(self.data)
            self.back_prop(self.data, self.target)
            self.update_weights()
            weight_test.append(self.weights[1][3])
            costs.append(self.calculate_cost(self.data, self.target))
        return costs, weight_test
            
    def predict(self, x):
        self.forward_prop(x)
        preds = self.activations[len(self.activations) - 1]
        return np.argmax(preds, axis = 1)
    
    def accuracy(self, x, y):
        acc = 0
        preds = self.predict(x)
        for i in range(len(y)):
            if preds[i] == y[i]:
                acc += 1
        return acc/len(y)
           
