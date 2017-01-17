import numpy as np

def Lp_norm(A, p):
	"""Returns the Lp norm of a numpy array implemented 
	vector.

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
	"""Returns the max norm of a numpy array implemented 
	vector.

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
	"""Returns the Frobenius norm of a numpy array
	implemented matrix.

	Input
	-----
	A
	- a numpy array implemented vector or matrix

	Output
	------
	a scalar that is the Frobenius norm of the 
	matrix/vector
	"""
	d = A.ndim
	if d != 1 and d != 2:
		print("Error, the array should be of 1 or 2 \
			dimension.")
		return 
	elif d == 1:
		return Lp_norm(A, 2)
	else:
		A = np.power(A, 2)
		return A.sum().sum() ** (1./2.)
