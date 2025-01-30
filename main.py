from signal import Signal
import numpy as np
import pywt


def simple_haar_transform(signal: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""
	compute 1 level haar wavelet transform
	:param signal: input signal
	:return: arrays of the:
	- approximation coefficients
	- details coefficients
	- reconstructed signal
	"""
	# compute the wavelets coefficients: approximation and details
	cA, cD = pywt.dwt(data=signal, wavelet='haar')
	# reconstruct the signal:
	y = pywt.idwt(cA=cA, cD=cD, wavelet='haar')
	
	return cA, cD, y


def compute_dwt(signal: np.ndarray, wavelet: str = 'haar') -> tuple[np.ndarray, np.ndarray]:
	"""
	computes multy-level DWT for a signal
	:param signal: input signal
	:param wavelet: wavelet method
	:return: the approximation and details coefficients and a reconstructed signal.
	"""
	# apply decomposition to wavelet coefficients:
	coefficients = pywt.wavedec(data=signal, wavelet=wavelet)
	# reconstruct the signal:
	reconstructed_signal = pywt.waverec(coeffs=coefficients, wavelet=wavelet)
	
	return coefficients, reconstructed_signal


def compute_wavelet_coefficients_energy(coeffs: np.ndarray) -> float:
	"""
	computes the energy of a signals wavelet coefficients
	:param coeffs: DWT coefficients.
	:return: the energy of the signal coefficients.
	"""
	# flatten the coefficients and compute the total energy
	coefficients_energy = Signal.compute_signal_energy(signal=np.concatenate(coeffs))
	return coefficients_energy


if __name__ == "__main__":
	import signal_display as sd
	
	# TODO: implement Shifted Variant Wavelet Transform (with visualization).
	# TODO: implement end to end simulator with results for SNR in dB.
	# TODO: Read about Noise Suppression.
	# TODO: Read about 'haar' Vs. 'Duvashi'.
	# TODO: data - find good signals.
	
	# pulse signal parameters:
	signal_size = 2 ** 7
	pulse_width = 2 ** 4
	pulse_start = (2 ** 6)
	pulse_amplitude = 10
	# generate a pulse signal:
	base_pulse_signal = Signal.pulse_1d(n_samples=signal_size,
	                                    pulse_start=pulse_start,
	                                    pulse_duration=pulse_width,
	                                    amplitude=pulse_amplitude)
	un_signal = Signal.add_uniform_noise_1d(signal=base_pulse_signal)
	gn_signal = Signal.add_gaussian_noise_1d(signal=base_pulse_signal)
	
	
	def example_haar_dwt(signal: np.ndarray) -> None:
		"""
		execute visual example of a single level haar transform
		:param signal: input signal to analyze
		"""
		cA, cD, y = simple_haar_transform(signal=signal)
		sd.display_2_signals(signals=[cA, cD],
		                     titles=["Approximation coefficients", "Detailed coefficients"])
		sd.display_2_signals(signals=[signal,
		                              y],
		                     titles=["Signal",
		                             "Reconstructed signal"])
		
	# example_haar_dwt(signal=base_pulse_signal)
	
	
	def example_multy_level_haar_dwt(signal: np.ndarray) -> None:
		"""
		execute visual example of a multy level haar transform with coefficients scalogram
		:param signal: input signal to analyze
		"""
		coeffs, reconstructed_signal = compute_dwt(signal=signal, wavelet='haar')
		signal_info = f"pulse signal " \
		              f"[signal size: {signal_size}, pulse width: {pulse_width}, pulse start: {pulse_start}]"
		sd.plot_scalogram_and_signal(signal=signal, coeffs=coeffs, main_title_info=signal_info, wavelet_method='haar')
		
		# signal reconstruction inspection:
		# sd.display_2_signals(signals=[signal, signal - reconstructed_signal],
		#                      titles=["Original Signal", "Original Signal - Reconstructed signal"])
		
		return
		
	# example_multy_level_haar_dwt(signal=base_pulse_signal)
	
	def example_signal_energy(signal: np.ndarray) -> None:
		"""
		for a given signal, its wavelet decomposition and the reconstructed - compute the energies.
		:param signal: input signal.
		"""
		# compute haar wavelet decomposition:
		dwt_coefficients, reconstructed_signal = compute_dwt(signal=signal)
		
		# signal energy:
		s_e = Signal.compute_signal_energy(signal=signal)
		
		# DWT coefficients energy:
		c_e = compute_wavelet_coefficients_energy(coeffs=dwt_coefficients)
		
		# reconstructed signal energy:
		rs_e = Signal.compute_signal_energy(signal=reconstructed_signal)
		
		print(f"signal energy: {s_e}\n"
		      f"DWT coefficients energy: {c_e}\n"
		      f"reconstructed signal energy: {rs_e}")
		return
	
	# example_signal_energy(signal=base_pulse_signal)
	
	
