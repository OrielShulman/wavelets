import pywt
import numpy as np
from signal import Signal


class SignalWaveletAnalysis:
	"""
	a class that performs on a given 1D signal wavelets analysis: decomposition, reconstruction, Etc.
	"""
	
	def __init__(self, signal: np.ndarray):
		"""
		initiates the class for a given signal
		:param signal: an input 1D signal
		"""
		self.signal = signal
	
	def one_level_wavelet_transform(self, wavelet: str = 'haar') -> tuple[np.ndarray, np.ndarray, np.ndarray]:
		"""
		compute a single level depth wavelet transform
		:param wavelet: wavelet method to use
		:return: arrays of the:
		- approximation coefficients
		- details coefficients
		- reconstructed signal
		"""
		# compute the wavelets approximation and details coefficients:
		cA, cD = pywt.dwt(data=self.signal, wavelet=wavelet)
		
		# reconstruct the signal:
		y = pywt.idwt(cA=cA, cD=cD, wavelet='haar')
		
		# Approximation coefficient, Details coefficients, reconstructed signal
		return cA, cD, y
		