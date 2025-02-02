from signal import Signal
import numpy as np
import pywt


if __name__ == "__main__":
	import signal_display as sd
	from signal_wavelet_analysis import SignalWaveletAnalysis as SWA
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
	
	
	def example_haar_dwt(signal: np.ndarray) -> None:
		"""
		execute visual example of a single level haar transform
		:param signal: input signal to analyze
		"""
		swt = SWA(signal=signal)
		cA, cD, y = swt.one_level_wavelet_transform()
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
		coeffs, reconstructed_signal = SWA(signal=signal).compute_dwt(wavelet='haar')
		signal_info = f"pulse signal [signal size: {signal_size}, " \
		              f"pulse width: {pulse_width}, pulse start: {pulse_start}]"
		sd.plot_dwt_scalogram(signal=signal, coeffs=coeffs, signal_info=signal_info, wavelet_method='haar')
		
		# signal reconstruction inspection:
		# sd.display_2_signals(signals=[signal, signal - reconstructed_signal],
		#                      titles=["Original Signal", "Original Signal - Reconstructed signal"])
		
		return
	
	# example_multy_level_haar_dwt(signal=base_pulse_signal)
	
	
	def example_multy_level_haar_swt(signal: np.ndarray) -> None:
		"""
		execute visual example of a haar SWT transform with coefficients scalogram
		:param signal: input signal to analyze
		"""
		coeffs, reconstructed_signal = SWA(signal=signal).compute_swt(wavelet='haar')
		signal_info = f"pulse signal [signal size: {signal_size}, " \
		              f"pulse width: {pulse_width}, pulse start: {pulse_start}]"
		sd.plot_swt_scalogram(signal=signal, coeffs=coeffs, signal_info=signal_info, wavelet_method='haar')
		
		return
	
	# example_multy_level_haar_swt(signal=base_pulse_signal)
	
	def example_signal_energy(signal: np.ndarray) -> None:
		"""
		for a given signal, its wavelet decomposition and the reconstructed - compute the energies.
		:param signal: input signal.
		"""
		# compute haar wavelet DWT decomposition:
		dwt_coefficients, reconstructed_signal = SWA(signal=signal).compute_dwt(wavelet='haar')
		
		# # compute haar wavelet SWT decomposition:
		# swt_coefficients, reconstructed_signal = compute_dwt(signal=signal)
		
		# signal energy:
		s_e = Signal.compute_signal_energy(signal=signal)
		
		# DWT coefficients energy:
		c_e = SWA(signal=signal).compute_dwt_coeffs_energy(coeffs=dwt_coefficients)
		
		# reconstructed signal energy:
		rs_e = Signal.compute_signal_energy(signal=reconstructed_signal)
		
		print(f"signal energy: {s_e}\n"
		      f"DWT coeffs energy: {c_e}\n"
		      f"IDWT energy: {rs_e}")
		return
	
	example_signal_energy(signal=base_pulse_signal)
	
	
	def example_shift_invariant_wavelet_transform(signal: np.ndarray):
		# Perform SWT
		coeffs = pywt.swt(signal, 'haar', level=4)
		# Plot the original signal and the SWT coefficients
		plt.figure(figsize=(12, 8))
		plt.suptitle('Shift invariant')
		plt.subplot(5, 1, 1)
		plt.plot(signal)
		plt.title('Original Signal')
		for i in range(4):
			plt.subplot(5, 1, i + 2)
			plt.plot(coeffs[i][0])
			plt.title(f'SWT Approximation Coefficients - Level {i + 1}')
		plt.tight_layout()
		plt.show()
		return
	
	def example_DWT(signal: np.ndarray):
		# Perform DWT
		coeffs = pywt.wavedec(signal, 'haar', level=4)
		# Plot the original signal and the DWT coefficients
		plt.figure(figsize=(12, 8))
		plt.suptitle('DWT')
		plt.subplot(5, 1, 1)
		plt.plot(signal)
		plt.title('Original Signal')
		for i in range(1, 5):
			plt.subplot(5, 1, i + 1)
			plt.plot(coeffs[i])
			plt.title(f'DWT Coefficients - Level {i}')
		plt.tight_layout()
		plt.show()
	
	
	# # Generate a sample signal
	# example_signal_s = np.sin(2 * np.pi * 7 * np.linspace(0, 1, 400)) + np.sin(2 * np.pi * 13 * np.linspace(0, 1, 400))
	
	# example_shift_invariant_wavelet_transform(signal=base_pulse_signal)
	# example_DWT(signal=base_pulse_signal)
