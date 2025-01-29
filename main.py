from signal import Signal
import signal_display
import numpy as np
import pywt


def simple_haar_transform(signal: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""
	compute 1 level haar wavelet transform
	:param signal: input signal
	:return: arrays of the:
	- approximation coefficients
	- details coefficients
	-reconstructed signal
	"""
	cA, cD = pywt.dwt(data=signal, wavelet='haar')
	signal_display.display_2_signals(signals=[cA,
	                                          cD],
	                                 titles=["Approximation coeffs",
	                                         "Detailed coeffs"])
	y = pywt.idwt(cA=cA, cD=cD, wavelet='haar')
	signal_display.display_2_signals(signals=[signal,
	                                          y],
	                                 titles=["Signal",
	                                         "Reconstructed signal"])
	
	return cA, cD, y


def compute_dwt(signal: np.ndarray, wavelet: str = 'haar') -> tuple[np.ndarray, np.ndarray]:
	"""
	computes multy-level DWT for a signal
	:param signal: input signal
	:param wavelet: wavelet method
	:return: the approximation and details coefficients and a reconstructed signal.
	"""
	coefficients = pywt.wavedec(data=signal, wavelet=wavelet)
	reconstructed_signal = pywt.waverec(coeffs=coefficients, wavelet=wavelet)
	return coefficients, reconstructed_signal
	

if __name__ == "__main__":
	base_signal = Signal.pulse_1d(n_samples=64, pulse_start=5, pulse_duration=32, amplitude=10)
	un_signal = Signal.add_uniform_noise_1d(signal=base_signal)
	gn_signal = Signal.add_gaussian_noise_1d(signal=base_signal)
	
	simple_haar_transform(signal=base_signal)
	
	# cA, cD = pywt.dwt(data=[3, 7, 1, 1, -2, 5, 4], wavelet='haar')
	# y = pywt.idwt(cA=cA, cD=cD, wavelet='haar')
	# exit(0)