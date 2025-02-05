from convolution_pulse_detection import ConvolutionPulseDetection
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class PulseSignalNoiseEstimation:
	"""
	estimates the noise of a pulse signal
	"""
	
	def __init__(self, signal: np.ndarray, pulse_width: int):
		"""
		initiates the class, applies pulse detection on the signal and extracts the noise
		:param signal: the pulse signal
		:param pulse_width: pulse width [in samples]
		"""
		self.signal: np.ndarray = signal
		self.pulse_width: int = pulse_width
		
		# Detect pulses in the signal:
		self.pulse_detection_mod = ConvolutionPulseDetection(signal=self.signal, pulse_width=self.pulse_width)
		
		# Extract the noise out of the signal:
		self.pulse_noise_indexes: pd.DataFrame = self.extract_pulse_noise()
		
		# Trim noise indexes above the 90th percentile and pulse below 10th percentile:
		self.trim_noise_pulse_out_layers()
		
	def extract_pulse_noise(self) -> pd.DataFrame:
		"""
		extracts the pulse and the noise indexes in the input signal
		:return: dataframe with indexes for: [pulse_idx, noise_idx]
		"""
		# TODO: add usecase of single pulse
		# detect pulse edges in the signal:
		pulse_edges = self.pulse_detection_mod.detected_pulses_edge
		# Discard the first and last pulses: (if more than 3 pulses are detected)
		if len(pulse_edges) >= 3:
			pulse_edges = pulse_edges.iloc[1:-1].reset_index(drop=True)
		
		# Extract indexes of pulse and noise:
		pulse_idx = []
		noise_idx = []
		for i in range(len(pulse_edges)):
			# pulse inside indexes:
			pulse_idx.extend(range(pulse_edges['pulse_start'][i], pulse_edges['pulse_end'][i] + 1))
			
			# non pulse indexes:
			if i < len(pulse_edges) - 1:
				noise_idx.extend(range(pulse_edges['pulse_end'][i] + 1, pulse_edges['pulse_start'][i+1]))
		
		# Create a data frame for the indexes:
		return pd.DataFrame({'pulse_idx': pd.Series(pulse_idx), 'noise_idx': pd.Series(noise_idx)})
	
	def trim_noise_pulse_out_layers(self, p_noise: int = 90, p_pulse: int = 10) -> None:
		"""
		trim all the noise values that are above the (p_noise)th percentile,
		also for pulse values below the (p_pulse)th percentile.
		:param p_noise: noise value percentile to trim above
		:param p_pulse: pulse value percentile to trim below
		"""
		# Calculate the 90th percentile for noise samples:
		percentile_noise = np.percentile(self.signal[self.pulse_noise_indexes['noise_idx'].dropna().astype(int)], p_noise)
		# Trim the points in noise_idx where the signal value is above the 90th percentile
		self.pulse_noise_indexes.loc[self.pulse_noise_indexes['noise_idx'].apply(lambda x: pd.notna(x) and self.signal[int(x)] > percentile_noise), 'noise_idx'] = np.nan
		
		# Calculate the 10th percentile for pulse samples:
		percentile_pulse = np.percentile(self.signal[self.pulse_noise_indexes['pulse_idx'].dropna().astype(int)], p_pulse)
		# Trim the points in pulse_idx where the signal value is below the 10th percentile
		self.pulse_noise_indexes.loc[self.pulse_noise_indexes['pulse_idx'].apply(lambda x: pd.notna(x) and self.signal[int(x)] < percentile_pulse), 'pulse_idx'] = np.nan
	
	def compute_noise_mean(self) -> float:
		"""
		compute the mean value of the signal noise, the signal values that correspond to the noise indexes.
		:return: signal noise mean
		"""
		noise_mean = self.signal[self.pulse_noise_indexes['noise_idx'].dropna().astype(int)].mean()
		return noise_mean
	
	def compute_pulse_mean(self) -> float:
		"""
		compute the mean value of the signal pulse amplitude, the signal values that correspond to the pulse indexes.
		:return: signal pulse amplitude mean
		"""
		pulse_mean = self.signal[self.pulse_noise_indexes['pulse_idx'].dropna().astype(int)].mean()
		return pulse_mean
	
	def compute_noise_std(self) -> float:
		"""
		compute the standard deviation of the signal noise, the signal values that correspond to the noise indexes.
		:return: signal noise mean
		"""
		noise_std = self.signal[self.pulse_noise_indexes['noise_idx'].dropna().astype(int)].std()
		return noise_std
	
	def compute_signal_snr(self) -> float:
		"""
		computes the signal SNR in dB
		:return: SNR in dB
		"""
		signal_pulse_mean = self.compute_pulse_mean()
		noise_std = self.compute_noise_std()
		return 20 * np.log10(signal_pulse_mean / noise_std)
	
	def display_noise_mean_estimation_process(self) -> None:
		"""
		display the mean values estimation process
		"""
		plt.figure(figsize=(12, 6))
		num_pulse_samples = len(self.pulse_noise_indexes['pulse_idx'].dropna())
		num_noise_samples = len(self.pulse_noise_indexes['noise_idx'].dropna())
		plt.suptitle(f"Pulse signal with {len(self.signal)} Samples "
		             f"({num_pulse_samples} pulse samples) "
		             f"({num_noise_samples} noise samples)", fontweight='bold')

		# Plot the signal
		plt.plot(self.signal, label='signal')
		
		# Plot the 'pulse' indexes in green
		plt.plot(self.pulse_noise_indexes['pulse_idx'].dropna().astype(int),
		         self.signal[self.pulse_noise_indexes['pulse_idx'].dropna().astype(int)],
		         color='green', label='pulse', marker='.', linestyle='None')
		
		# Plot the 'noise' indexes in orange
		plt.plot(self.pulse_noise_indexes['noise_idx'].dropna().astype(int),
		         self.signal[self.pulse_noise_indexes['noise_idx'].dropna().astype(int)],
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
	import os
	
	RAW_DATA_DIR_PATH = r'C:\Work\dym\2025-01-20 A2 SN1 stability\raw'
	DATA_FILE_NAME = 'raw_112'
	SIGNAL_PULSE_WIDTH = 49
	
	# Read a data file:
	file_path = os.path.join(RAW_DATA_DIR_PATH, f"{DATA_FILE_NAME}.csv")
	data_file = RawDataFile(file_path=file_path)
	
	# Extract the main_pd channel:
	# main_current_signal = data_file.df['main_current'][:4096].values
	# main_current_signal = data_file.df['main_current'][:2048].values
	# main_current_signal = data_file.df['main_current'][:1024].values
	
	main_current_signal = data_file.df['main_current'][:512].values  # 5 pulses, all pulses OK
	# main_current_signal = data_file.df['main_current'][:300].values  # 3 pulses, all pulses OK
	# main_current_signal = data_file.df['main_current'][:250].values  # 3 pulses, right side false
	# main_current_signal = data_file.df['main_current'][60:300].values  # 3 pulses, left side false
	# main_current_signal = data_file.df['main_current'][60:250].values  # 3 pulses, both sides false
	
	# Apply detection:
	noise_estimation_mod = PulseSignalNoiseEstimation(signal=main_current_signal, pulse_width=SIGNAL_PULSE_WIDTH)
	noise_estimation_mod.display_noise_mean_estimation_process()
	exit(0)