import numpy as np
x = np.array([[1, 5, 4, 6, 7], [3, 8, 1, 3, 1], [2, 9, 3, 8, 4]]).T

class ForwardNetwork:
    def __init__(self, data):
        self.data = data
        self.num_inputs = data.shape[0]
        self.num_features = data.shape[1]
        self.layers = {}
        
    def initialize_weights(self, size_layer, size_prev_layer, n):
        # Use the He initializing for weights. Weights drawn from guassian dist with std sqrt(2/n)
        return np.random.randn(size_layer, size_prev_layer) * np.sqrt(2/n)
        
    def add_layer(self, size_layer):
        """
        Adds layer of size size_layer (bias excluded)
        """
        # Check if this is first layer
        if len(self.layers) == 0:
            self.layers[0] = self.initialize_weights(size_layer, self.num_features + 1, self.num_inputs)
        else:
            counter = len(self.layers)
            size_prev_layer = self.layers[counter - 1].shape[0]
            self.layers[counter] = self.initialize_weights(size_layer, size_prev_layer + 1, self.num_inputs)
            
    def transform(self, x, w):
        """
        Calculates matrix product w^T*x
        """
        # Add bias term 
        x = np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)
        return np.dot(x, w.T)
    
    def activation(self, z):
        return np.maximum(z, 0)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
            
    def forward_prop(self, x, current_layer = 0):
        """
        Propagates input through all layers and return output of final layer
        """
        num_layers = len(self.layers)
        while current_layer < num_layers:
            #Get weights for current layer
            weights = self.layers.get(current_layer)
            z = self.transform(x, weights)
            current_layer += 1
            #If layer is the last layer we want to return the output
            if current_layer == num_layers:
                output = self.sigmoid(z)
                return output
            #If layer is not last layer we propagate one layer forward
            else:
                nodes = self.activation(z)
                return self.forward_prop(nodes, current_layer)
            
            
test = ForwardNetwork(x)        
test.add_layer(3)
test.add_layer(3)
test.add_layer(1)
print(test.forward_prop(x))
