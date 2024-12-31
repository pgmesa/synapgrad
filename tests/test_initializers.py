
import synapgrad

import numpy as np


def verify_tensor_properties(tensor, expected_shape, expected_dtype=np.float32):
    assert isinstance(tensor, synapgrad.Tensor), f"Expected Tensor type, got {type(tensor)}"
    assert tensor.data.dtype == expected_dtype, f"Expected data dtype {expected_dtype}, got {tensor.data.dtype}"
    assert tensor.dtype == expected_dtype, f"Expected tensor dtype {expected_dtype}, got {tensor.dtype}"
    assert tensor.data.shape == expected_shape, f"Expected shape {expected_shape}, got {tensor.data.shape}"

def test_tensor():
    # Test with default float dtype
    t = synapgrad.tensor([[1, 2], [3, 4]])
    verify_tensor_properties(t, (2, 2))
    assert np.array_equal(t.data, np.array([[1, 2], [3, 4]], dtype=np.float32))
    
    # Test with explicit dtype
    t = synapgrad.tensor([[1, 2], [3, 4]], dtype=np.float64)
    verify_tensor_properties(t, (2, 2), np.float64)
    assert np.array_equal(t.data, np.array([[1, 2], [3, 4]], dtype=np.float64))

def test_empty():
    # We can't test exact values for empty as they're undefined,
    # but we can verify the allocated memory has the right shape and type
    t = synapgrad.empty(2, 3)
    verify_tensor_properties(t, (2, 3))
    assert t.data.size == 6  # verify allocated size
    
    t = synapgrad.empty((2, 3), dtype=np.float64)
    verify_tensor_properties(t, (2, 3), np.float64)
    assert t.data.size == 6

def test_ones():
    t = synapgrad.ones(2, 3)
    verify_tensor_properties(t, (2, 3))
    assert np.all(t.data == 1)  # verify all elements are 1
    
    t = synapgrad.ones((2, 3), dtype=np.float64)
    verify_tensor_properties(t, (2, 3), np.float64)
    assert np.all(t.data == 1)
    assert t.data.dtype == np.float64

def test_ones_like():
    base = synapgrad.ones(2, 3)
    t = synapgrad.ones_like(base)
    verify_tensor_properties(t, (2, 3))
    assert np.all(t.data == 1)
    
    t = synapgrad.ones_like(base, dtype=np.float64)
    verify_tensor_properties(t, (2, 3), np.float64)
    assert np.all(t.data == 1)

def test_zeros():
    t = synapgrad.zeros(2, 3)
    verify_tensor_properties(t, (2, 3))
    assert np.all(t.data == 0)  # verify all elements are 0
    
    t = synapgrad.zeros((2, 3), dtype=np.float64)
    verify_tensor_properties(t, (2, 3), np.float64)
    assert np.all(t.data == 0)

def test_zeros_like():
    base = synapgrad.zeros(2, 3)
    t = synapgrad.zeros_like(base)
    verify_tensor_properties(t, (2, 3))
    assert np.all(t.data == 0)
    
    t = synapgrad.zeros_like(base, dtype=np.float64)
    verify_tensor_properties(t, (2, 3), np.float64)
    assert np.all(t.data == 0)

def test_arange():
    t = synapgrad.arange(5)
    verify_tensor_properties(t, (5,))
    assert np.array_equal(t.data, np.array([0, 1, 2, 3, 4], dtype=np.float32))
    
    t = synapgrad.arange(1, 5)
    verify_tensor_properties(t, (4,))
    assert np.array_equal(t.data, np.array([1, 2, 3, 4], dtype=np.float32))
    
    t = synapgrad.arange(0, 10, 2)
    verify_tensor_properties(t, (5,))
    assert np.array_equal(t.data, np.array([0, 2, 4, 6, 8], dtype=np.float32))

def test_rand():
    t = synapgrad.rand(2, 3)
    verify_tensor_properties(t, (2, 3))
    assert np.all((t.data >= 0) & (t.data <= 1))  # verify range [0, 1]
    
    t = synapgrad.rand((2, 3), dtype=np.float64)
    verify_tensor_properties(t, (2, 3), np.float64)
    assert np.all((t.data >= 0) & (t.data <= 1))

def test_randn():
    t = synapgrad.randn(2, 3)
    verify_tensor_properties(t, (2, 3))
    # For normal distribution, we can check if values are within reasonable bounds
    # (99.7% of values should be within 3 standard deviations)
    assert np.all(np.abs(t.data) < 10)  
    
    t = synapgrad.randn((2, 3), dtype=np.float64)
    verify_tensor_properties(t, (2, 3), np.float64)
    assert np.all(np.abs(t.data) < 10)

def test_normal():
    mean, std = 5, 2
    t = synapgrad.normal(mean, std, 2, 3)
    verify_tensor_properties(t, (2, 3))
    # Check if mean and std are roughly as expected (within reasonable bounds)
    assert abs(np.mean(t.data) - mean) < 2  # allow some deviation
    assert abs(np.std(t.data) - std) < 1
    
    t = synapgrad.normal(mean, std, 2, 3, dtype=np.float64)
    verify_tensor_properties(t, (2, 3), np.float64)
    assert abs(np.mean(t.data) - mean) < 2

def test_randint():
    low, high = 0, 10
    t = synapgrad.randint(low, high, (2, 3))
    verify_tensor_properties(t, (2, 3), np.int32)
    assert np.all((t.data >= low) & (t.data < high))  # verify range
    assert np.all(t.data.astype(int) == t.data)  # verify integers
    
    t = synapgrad.randint(low, high, (2, 3), dtype=np.int64)
    verify_tensor_properties(t, (2, 3), np.int64)
    assert np.all((t.data >= low) & (t.data < high))

def test_eye():
    t = synapgrad.eye(3)
    verify_tensor_properties(t, (3, 3))
    # Verify diagonal is 1 and rest is 0
    assert np.array_equal(t.data, np.eye(3, dtype=np.float32))
    
    t = synapgrad.eye(3, dtype=np.float64)
    verify_tensor_properties(t, (3, 3), np.float64)
    assert np.array_equal(t.data, np.eye(3, dtype=np.float64))