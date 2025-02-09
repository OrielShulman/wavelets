from convolution_pulse_detection import ConvolutionPulseDetection
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# matplotlib.use("TkAgg")
plt.style.use("ggplot")

# TODO: approach the signal intensity differences in signal mean.


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
		self.pulse_noise_indices: pd.DataFrame = self.extract_pulse_noise_indices()
		
		# Trim noise indexes above the 90th percentile and pulse below 10th percentile:
		self.trim_noise_pulse_out_layers()
		
	def extract_pulse_noise_indices(self) -> pd.DataFrame:
		"""
		extracts the pulse and the noise indexes in the input signal
		pulses from both edges of the signal may be corrupted and result in false pulse detection,
		thous, if the signal was detected with more than 3 pulses - first and last pulses are trimmed.
		:return: dataframe with indexes for: [pulse_idx, noise_idx]
		"""
		# detect pulse edges in the signal:
		pulse_edges = self.pulse_detection_mod.detected_pulses_edge
		
		# Extract indexes of pulse and noise:
		pulse_idx = []
		noise_idx = []
		
		# If only one pulse is detected (for the case of a centered pulse)
		if len(pulse_edges) == 1:
			
			# pulse inside indexes:
			pulse_idx.extend(range(pulse_edges['pulse_start'][0], pulse_edges['pulse_end'][0] + 1))
			# non pulse indexes:
			noise_idx.extend(range(0, pulse_edges['pulse_start'][0]))
			noise_idx.extend(range(pulse_edges['pulse_end'][0] + 1, len(self.signal)))
		
		else:
			# Discard the first and last pulses: (if more than 3 pulses are detected)
			if len(pulse_edges) >= 3:
				pulse_edges = pulse_edges.iloc[1:-1].reset_index(drop=True)
			
			for i in range(len(pulse_edges)):
				# pulse inside indexes:
				pulse_idx.extend(range(pulse_edges['pulse_start'][i], pulse_edges['pulse_end'][i] + 1))
				# non pulse indexes:
				if i < len(pulse_edges) - 1:
					noise_idx.extend(range(pulse_edges['pulse_end'][i] + 1, pulse_edges['pulse_start'][i + 1]))
		
		# Create a data frame for the indexes:
		return pd.DataFrame({'pulse_idx': pd.Series(pulse_idx), 'noise_idx': pd.Series(noise_idx)})
	
	def trim_noise_pulse_out_layers(self, p_noise: int = 90, p_pulse: int = 10) -> None:
		"""
		trim all the noise values that are above the (p_noise)th percentile,
		also for pulse values below the (p_pulse)th percentile.
		:param p_noise: noise value percentile to trim above
		:param p_pulse: pulse value percentile to trim below
		"""
		if self.pulse_noise_indices['noise_idx'].notna().any():
			# Calculate the 90th percentile for noise samples:
			percentile_noise = np.percentile(self.signal[self.pulse_noise_indices['noise_idx'].dropna().astype(int)],
			                                 p_noise)
			# Trim the points in noise_idx where the signal value is above the 90th percentile
			self.pulse_noise_indices.loc[self.pulse_noise_indices['noise_idx'].apply(
				lambda x: pd.notna(x) and self.signal[int(x)] > percentile_noise), 'noise_idx'] = np.nan
		
		if self.pulse_noise_indices['pulse_idx'].notna().any():
			# Calculate the 10th percentile for pulse samples:
			percentile_pulse = np.percentile(self.signal[self.pulse_noise_indices['pulse_idx'].dropna().astype(int)],
			                                 p_pulse)
			# Trim the points in pulse_idx where the signal value is below the 10th percentile
			self.pulse_noise_indices.loc[self.pulse_noise_indices['pulse_idx'].apply(
				lambda x: pd.notna(x) and self.signal[int(x)] < percentile_pulse), 'pulse_idx'] = np.nan
	
	def compute_noise_mean(self) -> float:
		"""
		compute the mean value of the signal noise, the signal values that correspond to the noise indexes.
		:return: signal noise mean
		"""
		noise_mean = self.signal[self.pulse_noise_indices['noise_idx'].dropna().astype(int)].mean()
		return noise_mean
	
	def compute_pulse_mean(self) -> float:
		"""
		compute the mean value of the signal pulse amplitude, the signal values that correspond to the pulse indexes.
		:return: signal pulse amplitude mean
		"""
		pulse_mean = self.signal[self.pulse_noise_indices['pulse_idx'].dropna().astype(int)].mean()
		return pulse_mean
	
	def compute_noise_std(self) -> float:
		"""
		compute the standard deviation of the signal noise, the signal values that correspond to the noise indexes.
		:return: signal noise mean
		"""
		noise_std = self.signal[self.pulse_noise_indices['noise_idx'].dropna().astype(int)].std()
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
		plt.figure(figsize=(15, 8))
		num_pulse_samples = len(self.pulse_noise_indices['pulse_idx'].dropna())
		num_noise_samples = len(self.pulse_noise_indices['noise_idx'].dropna())
		plt.suptitle(f"Pulse signal with {len(self.signal)} samples", fontweight='bold')
		
		# Plot the signal
		plt.plot(self.signal, label='signal')
		
		# Plot the 'pulse' indexes in green
		plt.plot(self.pulse_noise_indices['pulse_idx'].dropna().astype(int),
		         self.signal[self.pulse_noise_indices['pulse_idx'].dropna().astype(int)],
		         color='tab:green', label=f"({num_pulse_samples} pulse samples)", marker='.', linestyle='None')
		
		# Plot the 'noise' indexes in orange
		plt.plot(self.pulse_noise_indices['noise_idx'].dropna().astype(int),
		         self.signal[self.pulse_noise_indices['noise_idx'].dropna().astype(int)],
		         color='tab:blue', label=f"({num_noise_samples} noise samples)", marker='.', linestyle='None')
		
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
	SIGNAL_PULSE_WIDTH = 50
	
	# Read a data file:
	file_path = os.path.join(RAW_DATA_DIR_PATH, f"{DATA_FILE_NAME}.csv")
	data_file = RawDataFile(file_path=file_path)
	
	# Extract the main_pd channel:
	main_pd_signal = data_file.df['main_pd'].values
	signa_partial = main_pd_signal
	# signa_partial = main_pd_signal[:4096]
	# signa_partial = main_pd_signal[:2048]
	# signa_partial = main_pd_signal[:1024]
	
	# signa_partial = main_pd_signal[:512]  # 5 pulses, all pulses OK
	# signa_partial = main_pd_signal[:300]  # 3 pulses, all pulses OK
	# signa_partial = main_pd_signal[:170]  # 2 pulses, right side false
	# signa_partial = main_pd_signal[60:225]  # 2 pulses, left side false
	# signa_partial = main_pd_signal[60:250]  # 3 pulses, both sides false
	# signa_partial = main_pd_signal[15:115]  # 1 centered pulses, successful detection
	
	# Apply detection:
	noise_estimation_mod = PulseSignalNoiseEstimation(signal=signa_partial, pulse_width=SIGNAL_PULSE_WIDTH)
	noise_estimation_mod.display_noise_mean_estimation_process()
	exit(0)
