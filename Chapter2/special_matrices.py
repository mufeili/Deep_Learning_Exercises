import numpy as np

from norm import lp_norm
from multiplication import dot_product, matrix_product
from transpose import transpose
from identity import identity

def is_symmetric(A):
	"""Test if an numpy array is a symmetric matrix, 
	i.e. if A is of dimension 2 and A == A.T returns True.

	Input
	-----
	A
	- an numpy array

	Output
	------
	True/False
	- If A is/is not a symmetric matrix.
	"""
	if A.ndim == 1:
		print("This is a vector, not a matrix!")
		return False
	if A.ndim != 2:
		print("This is not a matrix!")
		return False
	else:
		return np.array_equal(A, A.T)

def is_unit_vector(A):
	"""Test if an numpy array is a unit vector, i.e. 
	if A is of dimension 1 and its L2 norm is 1.

	Input
	-----
	A
	- an numpy array

	Output
	------
	True/False
	- If A is/is not a unit vector.
	"""
	if A.ndim != 1:
		print("This is not a vector!")
		return False
	else:
		l2_norm = lp_norm(A, 2)
		if l2_norm == 1:
			return True
		else:
			return False

def orthogonal_to_each_other(A, B):
	"""If two numpy arrays are both vectors, 
	test if they are orthogonal to each other.

	Input
	-----
	A, B
	- two numpy arrays

	Output
	------
	True/False
	- If A and B are/are not orthogonal to each other.
	"""
	if A.ndim != 1 or B.ndim != 1:
		print("At least one of the numpy array is not a vector!")
		return False
	else:
		if dot_product(A, B) == 0:
			return True
		else:
			return False

def are_orthonormal(A, B):
	"""If two numpy arrays are both vectors and they are
	orthogonal to each other, test if they have unit norm.

	Input
	-----
	A, B 
	- two numpy arrays

	Output
	------
	True/False
	- If A and B are/are not orthonormal.
	"""
	if orthogonal_to_each_other(A, B):
		if lp_norm(A, 2) == 1 and lp_norm(B, 2) == 1:
			return True
		else:
			return False
	else:
		return False

def is_orthogonal(A):
	"""Test if an numpy array is an orthogonal matrix.

	Input
	-----
	A
	- a numpy array

	Output
	------
	True/False
	- If A is/is not an orthogonal matrix.
	"""
	if A.ndim != 2:
		print("This is not a matrix!")
		return False
	if A.shape[0] != A.shape[1]:
		print("This is not a square matrix!")
		return False

	product = matrix_product(A, transpose(A))

	if np.array_equal(product, identity(A.shape[0])):
		return True
	else:
		return False
