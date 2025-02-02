import signal_display as sd
from signal import Signal
from signal_wavelet_analysis import SignalWaveletAnalysis

import numpy as np
import pywt


def example_single_level_haar_dwt(signal: np.ndarray) -> None:
	"""
	execute visual example of a single level haar transform
	:param signal: input signal to analyze
	"""
	cA, cD, y = SignalWaveletAnalysis(signal=signal).one_level_wavelet_transform()
	sd.display_2_signals(signals=[cA, cD],
	                     titles=["Approximation coefficients", "Detailed coefficients"])
	sd.display_2_signals(signals=[signal,
	                              y],
	                     titles=["Signal",
	                             "Reconstructed signal"])

	
def example_multy_level_haar_dwt(signal: np.ndarray, wavelet: str = 'haar', signal_info: str = None) -> None:
	"""
	execute visual example of a multy level haar DWT with visualization of coefficients scalogram
	:param signal: input signal to analyze.
	:param wavelet: wavelet method.
	:param signal_info: signal information to attach with the plot title.
	"""
	# Apply DWT:
	coeffs, reconstructed_signal = SignalWaveletAnalysis(signal=signal).compute_dwt(wavelet=wavelet)
	
	# Plot results scalogram:
	sd.plot_dwt_scalogram(signal=signal, coeffs=coeffs, signal_info=signal_info, wavelet_method=wavelet)
	
	# Plot signal reconstruction inspection:
	sd.display_2_signals(signals=[signal, np.abs(signal - reconstructed_signal)],
	                     titles=["Original Signal", "| Original Signal - IDWT |"])
	return


def example_multy_level_haar_swt(signal: np.ndarray, wavelet: str = 'haar', signal_info: str = None) -> None:
	"""
	execute visual example of a haar SWT transform with visualization of coefficients scalogram
	:param signal: input signal to analyze
	:param wavelet: wavelet method.
	:param signal_info: signal information to attach with the plot title.
	"""
	# Apply SWT:
	coeffs, reconstructed_signal = SignalWaveletAnalysis(signal=signal).compute_swt(wavelet=wavelet)
	
	# Plot results scalogram:
	sd.plot_swt_scalogram(signal=signal, coeffs=coeffs, signal_info=signal_info, wavelet_method=wavelet)
	
	# Plot signal reconstruction inspection:
	sd.display_2_signals(signals=[signal, np.abs(signal - reconstructed_signal)],
	                     titles=["Original Signal", "| Original Signal - ISWT |"])
	return


def example_compare_dwt_signal_energy(signal: np.ndarray, wavelet: str = 'haar') -> None:
	"""
	compare the signal evergy with its DWT coefficients and its IDWT energy.
	:param signal: input signal.
	:param wavelet: wavelet method.
	"""
	signal_analysis_mod = SignalWaveletAnalysis(signal=signal)
	
	# compute haar wavelet DWT decomposition:
	dwt_coefficients, reconstructed_signal = signal_analysis_mod.compute_dwt(wavelet=wavelet)
	
	# signal energy:
	s_e = Signal.compute_signal_energy(signal=signal)
	
	# DWT coefficients energy:
	c_e = signal_analysis_mod.compute_dwt_coeffs_energy(coeffs=dwt_coefficients)
	
	# reconstructed signal energy:
	rs_e = Signal.compute_signal_energy(signal=reconstructed_signal)
	
	print(f"signal energy: {s_e}\n"
	      f"DWT coeffs energy: {c_e}\n"
	      f"IDWT energy: {rs_e}")
	return


def example_haar_SWT(signal: np.ndarray) -> None:
	"""
	perform SWT and plot its coefficients along the signal (as plots)
	"""
	# Perform SWT
	coeffs = pywt.swt(signal, 'haar')
	n = len(coeffs)
	# Plot the original signal and the SWT coefficients
	plt.figure(figsize=(12, (1 + n) * 2))
	plt.suptitle('SWT\n')
	plt.subplot(n+1, 1, 1)
	plt.plot(signal)
	plt.title('Original Signal')
	for i in range(n):
		plt.subplot(n+1, 1, i + 2)
		plt.plot(coeffs[i][0])
		plt.title(f'Approximation Coefficients - Level {n - i}')
	plt.tight_layout()
	plt.show()
	return


def example_haar_DWT(signal: np.ndarray) -> None:
	"""
	perform DWT and plot its coefficients along the signal (as plots)
	"""
	# Perform DWT
	coeffs = pywt.wavedec(signal, 'haar')
	n = len(coeffs)
	# Plot the original signal and the DWT coefficients
	plt.figure(figsize=(12, n*2))
	plt.suptitle('DWT\n')
	plt.subplot(n, 1, 1)
	plt.plot(signal)
	plt.title('Original Signal')
	for i in range(1, n):
		plt.subplot(n, 1, i+1)
		plt.plot(coeffs[i], marker='x', markerfacecolor='r', markeredgecolor='r')
		plt.xticks([])
		plt.title(f'Approximation Coefficients - Level {n - i}')
	plt.tight_layout()
	plt.show()


if __name__ == "__main__":
	import matplotlib.pyplot as plt
	
	# TODO: implement end to end simulator with results for SNR in dB.
	# TODO: Read about 'haar' Vs. 'Duvashi'.
	# TODO: data - find good signals.
	
	# pulse signal parameters:
	signal_size = 2 ** 7
	pulse_width = 2 ** 5
	pulse_start = (2 ** 5) + 1
	pulse_amplitude = 10
	# generate a pulse signal:
	base_pulse_signal = Signal.pulse_1d(n_samples=signal_size,
	                                    pulse_start=pulse_start,
	                                    pulse_duration=pulse_width,
	                                    amplitude=pulse_amplitude)
	un_signal = Signal.add_uniform_noise_1d(signal=base_pulse_signal)
	gn_signal = Signal.add_gaussian_noise_1d(signal=base_pulse_signal)
	example_signal_s = np.sin(2 * np.pi * 7 * np.linspace(0, 1, 400)) + np.sin(2 * np.pi * 13 * np.linspace(0, 1, 400))
	
	# Execute examples:
	
	# example_single_level_haar_dwt(signal=base_pulse_signal)
	#
	# example_multy_level_haar_dwt(signal=base_pulse_signal,
	#                              signal_info=f"pulse signal [signal size: {signal_size}, "
	#                                          f"pulse width: {pulse_width}, pulse start: {pulse_start}]")
	#
	# example_multy_level_haar_swt(signal=base_pulse_signal,
	#                              signal_info=f"pulse signal [signal size: {signal_size}, "
	#                                          f"pulse width: {pulse_width}, pulse start: {pulse_start}]")
	#
	# example_compare_dwt_signal_energy(signal=base_pulse_signal)
	#
	# example_haar_SWT(signal=base_pulse_signal)
	#
	# example_haar_DWT(signal=base_pulse_signal)
