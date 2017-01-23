from numpy.linalg import inv

def inverse(A):
	"""Returns the inverse of an invertible n-dimensional square 
	matrix in the representation of a numpy array.

	Input
	-----
	A
	- An numpy array of shape(n, n) which is invertible 
	in the sense of matrix

	Output
	------
	An numpy array of shape(n, n) which is the inverse of A
	"""
	return inv(A)