import numpy as np 

def identity(n):
	"""Returns an n dimensional identity matrix.

	Input
	-----
	n
	- an integer that indicates the dimension of 
	the identity matrix

	Output
	------
	- A numpy array representation of n-dimensional 
	identity matrix.
	"""
	return np.identity(n)
