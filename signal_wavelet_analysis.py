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
	
	def compute_dwt(self, wavelet: str = 'haar') -> tuple[np.ndarray, np.ndarray]:
		"""
		computes Discrete Wavelet Transform (DWT) and Inverse Discrete Wavelet Transform (IDWT) to the signal.
		:param wavelet: wavelet method
		:return: the deconstructed wavelet DWT coefficients and a reconstructed signal.
		"""
		# Perform DWT decomposition:
		coefficients = pywt.wavedec(data=self.signal, wavelet=wavelet)
		
		# Perform IDWT reconstruction:
		reconstructed_signal = pywt.waverec(coeffs=coefficients, wavelet=wavelet)
		
		# signal DWT coefficients, reconstructed signal
		return coefficients, reconstructed_signal
	
	def compute_swt(self, wavelet: str = 'haar') -> tuple[list, np.ndarray]:
		"""
		computes Stationary Wavelet Transform (SWT) and Inverse Stationary Wavelet Transform (ISWT) to the signal.
		:param wavelet: wavelet method
		:return: the deconstructed wavelet SWT coefficients and a reconstructed signal.
		"""
		# Perform SWT decomposition:
		coefficients = pywt.swt(data=self.signal, wavelet=wavelet)
		
		# Perform ISWT reconstruction:
		reconstructed_signal = pywt.iswt(coeffs=coefficients, wavelet=wavelet)
		
		# signal SWT coefficients, reconstructed signal
		return coefficients, reconstructed_signal
	
	@ staticmethod
	def compute_dwt_coeffs_energy(coeffs: np.ndarray) -> float:
		"""
		computes the energy of a DWT wavelet coefficients
		:param coeffs: DWT coefficients.
		:return: the energy of the signal coefficients.
		"""
		# flatten the coefficients and compute the total energy
		coefficients_energy = Signal.compute_signal_energy(signal=np.concatenate(coeffs))
		return coefficients_energy
	