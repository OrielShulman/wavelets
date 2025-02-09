import numpy as np
import pandas as pd
from scipy.signal import convolve, find_peaks
import matplotlib.pyplot as plt
plt.style.use("ggplot")


class ConvolutionPulseDetection:
	"""
	detect pulses in a given signal by convolution of the signal with an approximation of a single pulse.
	"""
	
	def __init__(self, signal: np.ndarray, pulse_width: int):
		"""
		initiate class instance
		:param signal: input signal
		:param pulse_width: approximation of pulse width
		"""
		self.signal: np.ndarray = signal
		self.pulse_width: int = pulse_width
		
		# Generate a rect pulse signal:
		self.rect_kernel: np.ndarray = self.generate_rect_pulse()
		
		# Convolve the signal with the rect pulse:
		self.convolution_result: np.ndarray = self.convolve_signal()
		
		# Detect peaks in the convolution results:
		self.convolution_peaks: np.ndarray = self.detect_peaks_in_convolution_result()
		
		# Generate a dataframe for the results:
		self.detected_pulses_edge: pd.DataFrame = self.pulses_edge_from_peaks()
		
	def generate_rect_pulse(self) -> np.ndarray:
		"""
		generates a clean pulse signal for convolution
		:return: pulse signal
		"""
		return np.ones(self.pulse_width)
	
	def convolve_signal(self) -> np.ndarray:
		"""
		compute the convolution of the signal with the pulse rect
		:return: the convolution result
		"""
		convolution_result = convolve(in1=self.signal, in2=self.rect_kernel, mode='same')
		return convolution_result
	
	def detect_peaks_in_convolution_result(self) -> np.ndarray:
		"""
		detect local maxima in the convolution result, the convolution result should be as a line of triangles
		:return: the detected peaks
		"""
		# Find local maxima:
		peaks, _ = find_peaks(self.convolution_result, distance=self.pulse_width)
		# Make sure that the peaks indexes are sorted:
		peaks.sort()
		return peaks
	
	def pulses_edge_from_peaks(self) -> pd.DataFrame:
		"""
		generate a sequence of pulses out of the detected peaks.
		:return: a dataframe for pulse start and the pulse end.
		"""
		# Create a DataFrame to store detected pulses
		detected_pulses = pd.DataFrame(columns=['pulse_start', 'pulse_end'])
		
		for peak in self.convolution_peaks:
			pulse_start = peak - self.pulse_width // 2
			pulse_end = peak + self.pulse_width // 2
			detected_pulses = pd.concat(
				[detected_pulses, pd.DataFrame({'pulse_start': [pulse_start], 'pulse_end': [pulse_end]})],
				ignore_index=True)
		return detected_pulses
	
	def synthesize_pulse_signal(self) -> np.ndarray:
		"""
		reconstruct a signal based on the detected pulsed
		:return: the reconstructed signal
		"""
		gen_signal = np.ones_like(self.signal) * np.min(self.signal)
		for _, row in self.detected_pulses_edge.iterrows():
			gen_signal[int(row['pulse_start']):int(row['pulse_end'])] = np.max(self.signal)
		return gen_signal
		
	def display_detection_process(self) -> None:
		"""
		plot the detection proccess and results
		:return:
		"""
		fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(15, 8))
		plt.suptitle(f"convolution pulse detection [pulse width = {self.pulse_width}]", fontweight='bold')
		
		# Subplot 1: Input Signal
		axes[0].plot(self.signal)
		axes[0].set_title(f'Input signal ({len(self.signal)} samples)')
		
		# Subplot 2: Convolution Result with Detected Peaks
		axes[1].plot(self.convolution_result)
		axes[1].plot(self.convolution_peaks,
		             self.convolution_result[self.convolution_peaks],
		             "x",
		             label=f'{len(self.convolution_peaks)} detected Peaks')
		axes[1].set_title(f'Convolution result (rect kernel length: {len(self.rect_kernel)})')
		axes[1].legend()
		
		# Subplot 3: Input Signal with Detected Peaks
		axes[2].plot(self.signal)
		for i, row in self.detected_pulses_edge.iterrows():
			axes[2].axvline(x=row['pulse_start'], color='tab:green', linestyle='--', label='pulse start' if i == 0 else "")
			axes[2].axvline(x=row['pulse_end'], color='tab:blue', linestyle='--', label='pulse start' if i == 0 else "")
		axes[2].set_title('Input signal with detected pulse edges')
		axes[2].legend()
		
		plt.tight_layout()
		plt.show()


if __name__ == "__main__":
	from raw_data_file import RawDataFile
	from signal import Signal
	import os
	
	RAW_DATA_DIR_PATH = r'C:\Work\dym\2025-01-20 A2 SN1 stability\raw'
	DATA_FILE_NAME = 'raw_112'
	SIGNAL_PULSE_WIDTH = 50
	
	# Generate Pulse signal:
	pulse_signal_params = {'n_samples': np.power(2, 7),
	                       'pulse_start': np.power(2, 5) + 1,
	                       'pulse_width': np.power(2, 4),
	                       'pulse_amplitude': 10,
	                       }
	base_pulse_signal = Signal.pulse_1d(n_samples=pulse_signal_params['n_samples'],
	                                    pulse_start=pulse_signal_params['pulse_start'],
	                                    pulse_duration=pulse_signal_params['pulse_width'],
	                                    amplitude=pulse_signal_params['pulse_amplitude'])
	
	# Apply detection on the base signal:
	# detection_module = ConvolutionPulseDetection(signal=base_pulse_signal, pulse_width=pulse_signal_params['pulse_width'])
	
	# Read a data file:
	file_path = os.path.join(RAW_DATA_DIR_PATH, f"{DATA_FILE_NAME}.csv")
	data_file = RawDataFile(file_path=file_path)
	
	# Extract the main_pd channel:
	main_pd_signal = data_file.df['main_pd'].values
	# signa_partial = main_pd_signal
	# signa_partial = main_pd_signal[:4096]
	# signa_partial = main_pd_signal[:2048]
	signa_partial = main_pd_signal[:1024]
	
	# signa_partial = main_pd_signal[:512]  # 5 pulses, all pulses OK
	# signa_partial = main_pd_signal[:300]  # 3 pulses, all pulses OK
	# signa_partial = main_pd_signal[:170]  # 2 pulses, right side false
	# signa_partial = main_pd_signal[60:225]  # 2 pulses, left side false
	# signa_partial = main_pd_signal[60:250]  # 3 pulses, both sides false
	# signa_partial = main_pd_signal[15:115]  # 1 centered pulses, successful detection
	
	# Apply detection:
	detection_module = ConvolutionPulseDetection(signal=signa_partial, pulse_width=SIGNAL_PULSE_WIDTH)
	# synthesized_signal = detection_module.synthesize_pulse_signal()
	detection_module.display_detection_process()
	exit(0)
	