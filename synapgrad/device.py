from enum import Enum

class Device(Enum):
    """ 
    Devices supported by synapgrad.
    Currently it only supports CPU.
    """
    CPU = 'cpu'
    #GPU = 'gpu