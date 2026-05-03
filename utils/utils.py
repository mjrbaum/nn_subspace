import torch

import numpy as np



def count_parameters(model):
    # Function to count total parameters and trainable parameters in a model
    params = sum(p.numel() for p in model.parameters())
    train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params, train_params


def get_params(model):
    # Function to retrieve vector with odel parameters
    params = np.concat([p.data.cpu().numpy().flatten() for p in model.parameters()], axis=0)
    return params


def train(dataloader, model, loss_fn, optimizer, device='cpu', return_loss=False, return_params=False):
    # Train loop for a single epoch
    size = len(dataloader.dataset)
    model.train()
    params_list = []
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if return_params:
            params_list.append(get_params(model))

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    # Return loss values and/or parameters based on flags
    if return_loss:
        if return_params:
            return loss.item(), np.stack(params_list, axis=1)
        else:
            return loss.item()
        
    if return_params:
        return np.stack(params_list, axis=1)
    

def test(dataloader, model, loss_fn, device='cpu'):
    # Test loop for a single epoch
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    return test_loss, correct


def run_svd(params, remove_first=10, trick=True):
    # Function to perform KL expansion on weight iterations
    if params.shape[1] < remove_first:
        raise('Removal iterations is greater than total iterations')
    
    # Remove first few iterations to remove bias from initialization
    params = params[:,remove_first:]

    if trick and params.shape[0] > params.shape[1]:
        trick = True
    else:
        trick = False

    # Remove the mean
    params = params - np.mean(params, axis=1, keepdims=True)
    
    # Linear algebra trick
    if trick:
        params = params.T

    # Find eigenvectors and values of covariance matrix
    Cov = np.matmul(params, params.T)
    vals, vecs = np.linalg.eigh(Cov)
    if trick:
        params = params.T
        vecs = np.matmul(params, vecs)
        vecs = vecs/np.linalg.norm(vecs, axis=0, keepdims=True)

    return np.flip(vecs, axis=1), np.flip(vals)


class Subspace_Generator():
    # Class to generate subspaces from space of eigenvectors
    def __init__(self, vecs, vals, size, std=False):
        # Initialize generator with vecs, vals, size
        self.vecs = vecs
        if std:
            sign = np.sign(vals)
            self.vals = sign*np.sqrt(np.abs(vals))
        else:
            self.vals = vals
        self.size = size
        self.idxs = None
        self.subspace = None

    def max_subspace(self):
        # Generate subspace of eigenvectors with highest eigenvalues
        self.subspace = self.vecs[:,0:self.size]
        self.idxs = range(self.size)

        return self.subspace

    def rand_subspace(self):
        # Generate weighted random subspace
        vals = self.vals[self.vals > 0]
        vals = vals/np.sum(vals)
        idxs = range(len(vals))

        # Randomly generate subspace indices
        filtered_idxs = np.random.choice(idxs, size=self.size, replace=False, p=vals)
        self.idxs = filtered_idxs
        self.subspace = self.vecs[:, filtered_idxs]

        return self.subspace
    
    def nonweighted_rand_subspace(self):
        # Generate a nonweighted random subspace
        # vals = self.vals[self.vals > 0]
        vals = self.vals
        idxs = range(len(vals))

        # Randomly generate subspace indices
        filtered_idxs = np.random.choice(idxs, size=self.size, replace=False)
        self.idxs = filtered_idxs
        self.subspace = self.vecs[:, filtered_idxs]

        return self.subspace