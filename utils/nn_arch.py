import torch
from torch import nn

import numpy as np



# Small CNN, meant to be used with MNIST
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_stack = nn.Sequential(
                        nn.Conv2d(1, 4, 4),
                        nn.MaxPool2d(4, 4))
        
        self.lin_stack = nn.Sequential(
                        nn.Linear(4*6*6, 16),
                        nn.Linear(16, 10))

    def forward(self, x):
        x = self.conv_stack(x)
        x = torch.flatten(x, 1)
        logits = self.lin_stack(x)
        return logits
    

# Mid-sized CNN, meant to be used with MNIST
class MidCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_stack = nn.Sequential(
                        nn.Conv2d(1, 16, kernel_size=(3,3), padding='same'),
                        nn.MaxPool2d(2, 2),
                        nn.Conv2d(16, 32, kernel_size=(3,3)),
                        nn.MaxPool2d(2, 2),
                        nn.Conv2d(32, 64, kernel_size=(3,3)),
                        nn.MaxPool2d(2, 2))
        
        self.lin_stack = nn.Sequential(
                        nn.Linear(64*2*2, 50),
                        nn.Linear(50, 10))

    def forward(self, x):
        x = self.conv_stack(x)
        x = torch.flatten(x, 1)
        logits = self.lin_stack(x)
        return logits
    

# Medium MLP, meant to be used with MNIST
class Medium_MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin_stack = nn.Sequential(
                        nn.Linear(784,100),
                        nn.Linear(100,20),
                        nn.Linear(20,10))
    
    def forward(self, x):
        logits = self.lin_stack(torch.flatten(x, start_dim=1))
        return logits
    

# wrapper around pytorch model to train in subspace
class Subspace_model(nn.Module):
    """
    Wraps a model in order to train it in a subspace
    """
    def __init__(self, model, vecs):
        # Set up subspace model parameters
        super(Subspace_model, self).__init__()
        self.device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
        self.model = model

        # Sets the subspace to train in
        if isinstance(vecs, np.ndarray):
            vecs = torch.from_numpy(vecs.copy()).to(self.device)
        self.register_buffer("vecs", vecs)
        self.d_dim = vecs.shape[1]

        # Initialize the subspace parameters
        self.params_d = nn.ParameterList([torch.zeros(self.d_dim)])
        self.params_d.to(self.device)
        # torch.nn.init.normal_(self.params_d[0], std=1/np.sqrt(self.d_dim))

        # Set model weights to be buffers
        self.register_buffer("params_0", self.get_params().to(self.device))

        # Unregister model parameters
        self.vector_to_parameters(self.params_0)

        self.params_D = self.vecs @ self.params_d[0] + self.params_0
    
    def get_params(self):
        # Function to retrieve model parameters as a tensor
        params = np.concat([p.data.cpu().numpy().flatten() for p in self.model.parameters()], axis=0)
        return torch.from_numpy(params)
    
    def vector_to_parameters(self, all_params):
        # Function to set model weights based on parameter tensor
        start_idx = 0
        for layer in self.model.modules():
            # Sets values in a layer if they exist as a weight or bias
            if hasattr(layer, 'weight'):
                num_params = layer.weight.numel()
                end_idx = start_idx + num_params
                new_params = torch.reshape(all_params[start_idx:end_idx], layer.weight.size())
                del layer.weight
                layer.weight = new_params
                start_idx = end_idx
                
            if hasattr(layer, 'bias'):
                num_params = layer.bias.numel()
                end_idx = start_idx + num_params
                new_params = torch.reshape(all_params[start_idx:end_idx], layer.bias.size())
                del layer.bias
                layer.bias = new_params
                start_idx = end_idx

    def new_subspace(self, vecs):
        # Apply a new subspace for the model to train in
        self.params_0 = self.params_D.to(self.device)
        self.vector_to_parameters(self.params_0)

        if isinstance(vecs, np.ndarray):
            vecs = torch.from_numpy(vecs.copy()).to(self.device)
        self.vecs = vecs
        self.params_d[0].detach().zero_()

    def forward(self, x):
        # Forward pass of subspace model
        self.params_D = self.vecs @ self.params_d[0] + self.params_0
        self.vector_to_parameters(self.params_D)
        return self.model.forward(x)