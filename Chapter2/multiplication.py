import numpy as np

def matrix_product(A, B):
	"""Take two numpy arrays of the same shape and return
	the product of them. 

	Input
	-----
	A
	- A numpy array of shape m x n
	B
	- A numpy array of shape n x p

	Output
	------
	C
	- A numpy array which equals to AB and is of shape
	  m x p

	For 2-D arrays it is equivalent to matrix
	multiplication, and for 1-D arrays to inner product
	of vectors (without complex conjugate). For N 
	dimensions it is a sum product over the last axis of 
	A and the second-to-last of B.
	"""
	return np.dot(A, B)

def elementwise_product(A, B):
	"""Take two numpy arrays and return the 
	elementwise-product(also known as Hadamard product) of 
	them. If the two arrays are not of the same shape by 
	accident, it will return A*B in terms of array 
	broadcasting.

	Input
	-----
	A
	- A numpy array of shape m x n
	B
	- A numpy array of shape m x n

	Output
	------
	C
	- A numpy array of shape m x n, whose entries are the
	  products of the corresponding entries of A and B.
	"""
	return np.multiply(A, B)

def dot_product(A, B):
	"""Take two vectors(one-dimensional numpy arrays) and
	return the dot product of them.

	Input
	-----
	A
	- A one dimensional numpy array which contains the 
	  same number of entries.
	B
	- A one dimensional numpy array which contains the 
	  same number of entries.

	Output
	------
	C
	- A scalar which is the dot product of A and B.
	"""
	return matrix_product(A, B)
