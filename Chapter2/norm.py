import numpy as np
from multiplication import dot_product

def lp_norm(A, p):
	"""Returns the Lp norm of a numpy array implemented vector.

	Input
	-----
	A
	- an one dimensional numpy array
	p
	- an integer 

	Output
	------
	a scalar that is the Lp norm of A
	"""
	if A.ndim != 1:
		print("Error, the array should be of 1-dimension.")
		return 
	else:
		A = np.absolute(A)
		A = np.power(A, p)
		return A.sum() ** (1./p)

def max_norm(A):
	"""Returns the max norm of a numpy array implemented vector.

	Input
	-----
	A
	- an one dimensional numpy array

	Output
	------
	a scalar that is the max norm of A
	"""
	if A.ndim != 1:
		print("Error, the array should be of 1-dimension.")
		return 
	else:
		A = np.absolute(A)
		return A.max()

def frobenius_norm(A):
	"""Returns the Frobenius norm of a numpy array implemented matrix.

	Input
	-----
	A
	- a numpy array implemented vector or matrix

	Output
	------
	a scalar that is the Frobenius norm of the matrix/vector
	"""
	d = A.ndim
	if d != 1 and d != 2:
		print("Error, the array should be of 1 or 2 dimension.")
		return 
	elif d == 1:
		return lp_norm(A, 2)
	else:
		A = np.power(A, 2)
		return A.sum().sum() ** (1./2.)

def angle(x, y):
	"""Returns the angle between two vectors.

	Input
	-----
	x, y
	- two numpy array implemented vectors

	Output
	------
	a scalar that is the angle between vectors x and y, in the range 
	of [0, Pi]
	"""
	return np.arccos(dot_product(x, y)/(lp_norm(x, 2) * 
		lp_norm(y, 2)))
