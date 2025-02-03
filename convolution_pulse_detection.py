import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import convolve, find_peaks


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
		self.detected_pulses: pd.DataFrame = self.pulses_from_peaks()
		
		# Generate a reconstructed signal based on the results:
		self.reconstructed_signal: np.ndarray = self.reconstruct_signal()
		
		# Check detected pulses margins:
		self.pulses_with_margins: pd.DataFrame = self.check_pulses_margins()
		
		# Display results:
		self.display_detection_process()
		
	def generate_rect_pulse(self) -> np.ndarray:
		"""
		generates a clean pulse signal for convolution
		:return: pulse signal
		"""
		# TODO: is odd sized kernel neccesary?
		# make the pulse odd sized for center aligned peak detection:
		# pulse_width = self.pulse_width + 1 if self.pulse_width % 2 == 0 else self.pulse_width
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
		# TODO: find_peaks can also take distance
		# find local maxima:
		peaks, _ = find_peaks(self.convolution_result)
		return peaks
	
	def pulses_from_peaks(self) -> pd.DataFrame:
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
	
	def reconstruct_signal(self) -> np.ndarray:
		"""
		reconstruct a signal based on the detected pulsed
		:return: the reconstructed signal
		"""
		gen_signal = np.ones_like(self.signal) * np.min(self.signal)
		for _, row in self.detected_pulses.iterrows():
			gen_signal[int(row['pulse_start']):int(row['pulse_end'])] = np.max(self.signal)
		return gen_signal
	
	def check_pulses_margins(self) -> pd.DataFrame:
		"""
		check margins between pulses, each pulse row will be added with a column for indication of margins
		:return:
		"""
		margin_df = pd.DataFrame(columns=['pulse_start', 'pulse_end', 'overlapping_pulses'])
		
		for i, row in self.detected_pulses.iterrows():
			overlapping_pulses = []
			for j, other_row in self.detected_pulses.iterrows():
				if i != j:
					if not (row['pulse_end'] < other_row['pulse_start'] or row['pulse_start'] > other_row['pulse_end']):
						overlapping_pulses.append(j)
			
			margin_df = pd.concat(
				[margin_df, pd.DataFrame({'pulse_start': [row['pulse_start']], 'pulse_end': [row['pulse_end']],
				                          'overlapping_pulses': [overlapping_pulses]})],
				ignore_index=True)
		
		return margin_df
	
	def display_detection_process(self) -> None:
		"""
		plot the detection proccess and results
		:return:
		"""
		fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(12, 10))
		plt.suptitle(f"convolution pulse detection [pulse width = {self.pulse_width}]", fontweight='bold')
		
		# Subplot 1: Input Signal
		axes[0].plot(self.signal)
		axes[0].set_title(f'Input signal ({len(self.signal)} samples)')
		
		# Subplot 3: Convolution Result with Detected Peaks
		axes[1].plot(self.convolution_result)
		axes[1].plot(self.convolution_peaks,
		             self.convolution_result[self.convolution_peaks],
		             "x",
		             label='Detected Peaks')
		axes[1].set_title(f'Convolution result (rect kernel length: {len(self.rect_kernel)}) with detected peaks')
		axes[1].legend()
		
		# Subplot 4: Input Signal with Detected Peaks
		axes[2].plot(self.signal)
		for i, row in self.detected_pulses.iterrows():
			axes[2].axvline(x=row['pulse_start'], color='g', linestyle='--', label='pulse start' if i == 0 else "")
			axes[2].axvline(x=row['pulse_end'], color='r', linestyle='--', label='pulse start' if i == 0 else "")
		axes[2].set_title('Input signal with detected peaks')
		axes[2].legend()
		
		# Subplot 5: Reconstructed Pulse Signal
		axes[3].plot(self.reconstructed_signal)
		axes[3].set_title(f'Reconstructed Pulse Signal (Pulse Width: {self.pulse_width})')
		
		plt.tight_layout()
		plt.show()


if __name__ == "__main__":
	from signal import Signal
	
	RAW_DATA_DIR_PATH = r'C:\Work\dym\2025-01-20 A2 SN1 stability\raw'
	
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
	c_detection = ConvolutionPulseDetection(signal=base_pulse_signal, pulse_width=pulse_signal_params['pulse_width'])
	
	# Read a data file:
	
	# Extract the main_pd channel:
	
	# Apply detection:
	
	exit(0)
	