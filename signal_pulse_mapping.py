from convolution_pulse_detection import ConvolutionPulseDetection

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# TODO: subtract the noise mean from the signal
# TODO: indicate signal SNR vs Coherent Migration SNR


class SignalPulseMapping:
	"""
	signal pulse map and pulse analysis.
	"""
	
	def __init__(self, signal: np.ndarray, pulse_width: int, pri: int):
		"""
		initiates the pulse map
		:param signal: input signal
		:param pulse_width: pulse width [in samples]
		:param pri: Pulse Repetition Interval [in samples]
		"""
		self.signal: np.ndarray = signal
		self.pulse_width: int = pulse_width
		self.pri: int = pri
		
		# Detect pulses in the signal:
		self.pulse_detection_mod = ConvolutionPulseDetection(signal=self.signal, pulse_width=self.pulse_width)
		
		# Detect pulse peaks in the signal
		self.pulse_peaks: np.ndarray = self.pulse_detection_mod.convolution_peaks
		
		# Build a pulse map based on the first detected pulse and the PRI:
		self.pulse_map: pd.DataFrame = self.build_pulse_map()
		# Apply coherent integration on the pulse map:
		self.integrated_pulse_signal: np.ndarray = self.integrate_pulse_map(pulse_map=self.pulse_map)
		
		# Build a centered pulse map based on the detected pulses centers:
		self.centered_pulse_map: pd.DataFrame = self.build_centered_pulse_map()
		# Apply coherent integration on the centered pulse map:
		self.integrated_centered_pulse_signal: np.ndarray = self.integrate_pulse_map(pulse_map=self.centered_pulse_map)
		
		pass
		
	def build_pulse_map(self) -> pd.DataFrame:
		"""
		Build a pulse map by splitting the signal pulses, split starting from the first detected peak with steps of PRI.
		
		Clip out the first and last signals if possible.
		:return:
		"""
		# start the intervals fom the second detected peak (first and last peak may be corrupt or with no edges):
		interval_start = self.pulse_peaks[1] if len(self.pulse_peaks) >= 2 else self.pulse_peaks[0]
		interval_start -= self.pri // 2
		
		# slice the signal into pulses:
		pulses = [self.signal[interval_start + i * self.pri: interval_start + (i + 1) * self.pri] for i in range(len(self.pulse_peaks) - 2)]
		
		# Create a pulse map
		pulse_map = pd.DataFrame(columns=['pulse'])
		pulse_map['pulse'] = pulses
		
		return pulse_map
	
	def build_centered_pulse_map(self) -> pd.DataFrame:
		"""
		Build a centered pulse map by splitting the signal pulses around the detected peaks.
		:return:
		"""
		# pulses centers:
		pulse_centers = self.pulse_peaks[1:-1] if len(self.pulse_peaks) >= 3 else self.pulse_peaks
		
		# Slice the array around each coordinate in pulse_centers and add each slice to a new row in the DataFrame
		pulses = []
		for center in pulse_centers:
			start = center - self.pri // 2
			end = center + self.pri // 2 + 1
			if start < 0 or end >= len(self.signal):
				continue
			pulses.append(self.signal[start:end])
		
		# Create a pulse map
		pulse_map = pd.DataFrame(columns=['pulse'])
		pulse_map['pulse'] = pulses
		
		return pulse_map
	
	@staticmethod
	def integrate_pulse_map(pulse_map: pd.DataFrame) -> np.ndarray:
		"""
		Apply coherent integration by summing up signals row wise of a pandas data frame
		and returns the sum of them as a 1D array.
		:param pulse_map: dataframe where each row contains a unique pulse signal
		:return: the coherently integrated signal
		"""
		# Calculate the element-wise sum of the pulses
		pulses_sum = np.sum(pulse_map['pulse'].values.tolist(), axis=0)
		
		# Calculate the element-wise average of the pulses
		pulses_average = np.mean(pulse_map['pulse'].values.tolist(), axis=0)
		
		return pulses_sum
	
	def display_noise_mean_estimation_process(self) -> None:
		"""
		display the mean values estimation process
		"""
		plt.figure(figsize=(12, 6))
		num_pulse_samples = len(self.pulse_noise_indices['pulse_idx'].dropna())
		num_noise_samples = len(self.pulse_noise_indices['noise_idx'].dropna())
		plt.suptitle(f"Pulse signal with {len(self.signal)} Samples "
		             f"({num_pulse_samples} pulse samples) "
		             f"({num_noise_samples} noise samples)", fontweight='bold')
		
		# Plot the signal
		plt.plot(self.signal, label='signal')
		
		# Plot the 'pulse' indexes in green
		plt.plot(self.pulse_noise_indices['pulse_idx'].dropna().astype(int),
		         self.signal[self.pulse_noise_indices['pulse_idx'].dropna().astype(int)],
		         color='green', label='pulse', marker='.', linestyle='None')
		
		# Plot the 'noise' indexes in orange
		plt.plot(self.pulse_noise_indices['noise_idx'].dropna().astype(int),
		         self.signal[self.pulse_noise_indices['noise_idx'].dropna().astype(int)],
		         color='red', label='noise', marker='.', linestyle='None')
		
		# Add title with the required information
		plt.title(f"pulse mean: {self.compute_pulse_mean():.2f}, "
		          f"noise mean: {self.compute_noise_mean():.2f}, "
		          f"signal SNR: {self.compute_signal_snr():.2f} dB")
		
		plt.tight_layout()
		plt.legend()
		plt.show()
	
if __name__ == "__main__":
	from raw_data_file import RawDataFile
	from signal import Signal
	import os
	
	RAW_DATA_DIR_PATH = r'C:\Work\dym\2025-01-20 A2 SN1 stability\raw'
	DATA_FILE_NAME = 'raw_112'
	SIGNAL_PULSE_WIDTH = 49
	SIGNAL_PRI = 100
	
	# Read a data file:
	file_path = os.path.join(RAW_DATA_DIR_PATH, f"{DATA_FILE_NAME}.csv")
	data_file = RawDataFile(file_path=file_path)
	
	# Extract the main_pd channel:
	# main_current_signal = data_file.df['main_current'][:4084].values
	# main_current_signal = data_file.df['main_current'][:1024].values
	main_current_signal = data_file.df['main_current'][:512].values
	# main_current_signal = data_file.df['main_current'][60:300].values
	# main_current_signal = data_file.df['main_current'][60:250].values
	
	# Apply detection:
	noise_estimation_mod = SignalPulseMapping(signal=main_current_signal, pulse_width=SIGNAL_PULSE_WIDTH, pri=SIGNAL_PRI)
	# noise_estimation_mod.display_noise_mean_estimation_process()
	exit(0)