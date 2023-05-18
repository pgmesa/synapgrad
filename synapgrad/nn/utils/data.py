
from abc import ABC, abstractmethod 
import numpy as np


def split_dataset(X, y, test_split:float=0.2, val_split=None, shuffle:bool=False):
    
    def get_split_indices(data_size, test_split, val_split):
        # Creating data indices for training and validation splits:
        indices = list(range(data_size))
        split = int(np.floor(test_split * data_size))
        if shuffle:
            np.random.shuffle(indices)
        train_val_indices, test_indices = indices[split:], indices[:split]
        
        if val_split is not None:
            val_split = int(np.floor(val_split * len(train_val_indices)))
            train_indices, val_indices = train_val_indices[val_split:], train_val_indices[:val_split]
        else:
            train_indices = train_val_indices
            val_indices = None
        
        return train_indices, test_indices, val_indices

    train_indices, test_indices, val_indices = get_split_indices(len(X), test_split, val_split)
    
    X_train = np.array([ X[ind] for ind in train_indices ], dtype=np.float32)
    y_train = np.array([ y[ind] for ind in train_indices ], dtype=np.float32)
    train = (X_train, y_train)
    
    X_test = np.array([ X[ind] for ind in test_indices ], dtype=np.float32)
    y_test = np.array([ y[ind] for ind in test_indices ], dtype=np.float32)
    test = (X_test, y_test)
    
    validation = None 
    if val_indices is not None:
        X_val = np.array([ X[ind] for ind in val_indices ], dtype=np.float32)
        y_val = np.array([ y[ind] for ind in val_indices ], dtype=np.float32)
        validation = (X_val, y_val)
    
    check_sum = (len(X_train) + len(X_test) + (0 if validation is None else len(X_val)))
    assert check_sum == len(X), f"Bad split '{check_sum}' != '{len(X)}'"
    
    return train, test, validation


def one_hot_encode(y) -> np.ndarray:
    uniques = list(np.unique(y))
    
    encoded = []
    for label in y:
        zeros = [0]*len(uniques)
        zeros[uniques.index(label)] = 1
        encoded.append(zeros)
        
    return np.array(encoded)


class DataLoaderCallback(ABC):
    
    @abstractmethod
    def __call__(self, data_loader:'DataLoader', X_batch:np.ndarray, y_batch:np.ndarray):
        pass
    
        
class DataLoader:
    
    def __init__(self, X, y, batch_size, transform:DataLoaderCallback=None) -> None:
        self.X = X; self.y = y
        self.batach_size = batch_size
        self.step = 0
        self.transform = transform
    
    def __len__(self):
        return len(self.y) // self.batach_size
    
    def __iter__(self):
        self.step = 0
        return self
    
    def __next__(self):
        if self.step < self.__len__():
            item = self.__getitem__(self.step)
            self.step += 1
            return item
        
        raise StopIteration
        
    def __getitem__(self, idx) -> tuple:
        start = idx*self.batach_size
        end = (idx*self.batach_size) + self.batach_size
        
        X_batch = self.X[start:end]
        y_batch = self.y[start:end]
        
        return self.transform(self, X_batch, y_batch)