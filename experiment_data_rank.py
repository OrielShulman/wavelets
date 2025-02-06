from convolution_pulse_detection import ConvolutionPulseDetection
from pulse_signal_noise_estimation import PulseSignalNoiseEstimation
import os
import numpy as np
import pandas as pd

# TODO: Implement:
#  - pulse intensity over time.


class ExperimentDataRank:
	"""
	rank an experiment data for each channel:
	"""
	
	def __init__(self, file_path: str, sample_rate: int, prf: int, duty_cycle: int):
		"""
		initiate data file class, reads the data from the file into a pandas dataframe
		:param file_path: absolute path to the data file
		:param sample_rate: [khz]
		:param prf: [khz]
		:param duty_cycle: [number in range (0, 100)]
		"""
		# Open the datafile:
		self.filename: str = os.path.splitext(os.path.basename(file_path))[0]
		self.df: pd.DataFrame = pd.read_csv(file_path)
		
		# Experiment metadata:
		self.sample_rate: int = sample_rate     # [khz]
		self.prf: int = prf                     # [khz]
		self.duty_cycle: int = duty_cycle       # [1, 100]
		self.pulse_width: int = int((self.sample_rate / self.prf) * (self.duty_cycle / 100))  # [samples]
		
		# Empty dataframe for ranking:
		self.rank_df: pd.DataFrame = pd.DataFrame(columns=self.df.columns)
		
		# Assess the quality of the data in the file:
		self.expected_number_of_pulses()  # Number of pulses expected:
		self.detected_number_of_pulses()  # Number of pulses detected:
		self.expected_pri()  # Expected PRI [pixels]
		self.detected_pri()  # Detected pulses PRI [pixels]
		self.expected_pulse_width()  # Expected pulse width [pixels]
		self.estimate_snr()  # Estimate the signal SNR.
		self.pulse_intensity_variation()  # Compute pulse intensity mean and std
		
		pass
	
	def expected_number_of_pulses(self) -> None:
		"""
		Assess the number of pulses that should be in the sample from the attached metadata.
		"""
		def signal_n_pulses_expected(signal: np.ndarray) -> int:
			"""
			for a single signal - count the number of peaks that was detected.
			:param signal: input pulse signal
			:return: number of pulses in the signal.
			"""
			n_pulses = (len(signal) * self.prf / self.sample_rate)
			
			return int(n_pulses)
		
		self.rank_df.loc['expected_n_pulses'] = [signal_n_pulses_expected(self.df[col].values) for col in self.df.columns]
	
	def detected_number_of_pulses(self) -> None:
		"""
		compute the number of pulses that was detected
		"""
		
		def count_signal_pulses(signal: np.ndarray) -> int:
			"""
			for a single signal - count the number of peaks that was detected.
			:param signal: input pulse signal
			:return: number of pulses in the signal.
			"""
			# detect pulses in the signal:
			pulse_detection_mod = ConvolutionPulseDetection(signal=signal, pulse_width=self.pulse_width)
			# count peaks (pulses center) without the pulses at the edge of the signal.
			n_pulses = len(pulse_detection_mod.convolution_peaks[1: -1])
			return n_pulses
		
		self.rank_df.loc['detected_n_pulses'] = [count_signal_pulses(self.df[col].values) for col in self.df.columns]
		
	def expected_pri(self) -> None:
		"""
		Assess the PRI from the attached file metadata.
		"""
		# Calculate the Pulse Repetition Interval (PRI) in samples (pixels)
		pri_pixels = self.sample_rate / self.prf
		
		self.rank_df.loc['expected_pri'] = [pri_pixels for col in self.df.columns]
	
	def detected_pri(self) -> None:
		"""
		extract the average distance between the detected pulses.
		"""
		def signal_average_pri(signal: np.ndarray) -> float:
			"""
			for a single signal - compute the average PRI of the detected pulses.
			:param signal: input pulse signal
			:return: Average PRI.
			"""
			# detect pulses in the signal:
			pulse_detection_mod = ConvolutionPulseDetection(signal=signal, pulse_width=self.pulse_width)
			# count peaks (pulses center)
			pulse_gaps_width = np.diff(pulse_detection_mod.convolution_peaks[1:-1])
			return float(np.mean(pulse_gaps_width))
		
		self.rank_df.loc['detected_pri'] = [signal_average_pri(self.df[col].values) for col in self.df.columns]
	
	def expected_pulse_width(self) -> None:
		"""
		assign the pulse width from the attached file metadata.
		"""
		
		self.rank_df.loc['expected_pulse_width'] = [self.pulse_width] * len(self.rank_df.columns)

	def estimate_snr(self) -> None:
		"""
		estimate each channels SNR
		"""
		def estimate_signal_snr(signal: np.ndarray) -> float:
			"""
			Estimate the signal SNR
			:param signal: input pulse signal
			:return: the signal estimated SNR.
			"""
			# noise estimation module:
			noise_est_mod = PulseSignalNoiseEstimation(signal=signal, pulse_width=self.pulse_width)

			return noise_est_mod.compute_signal_snr()
		
		self.rank_df.loc['signal_snr'] = [estimate_signal_snr(self.df[col].values) for col in self.df.columns]
	
	def pulse_intensity_variation(self) -> None:
		"""
		for each channel, check the uniformity of the pulse intensities.
		- split to pulses.
		- check each pulse mean intensity.
		- calculate the pulse intensities deviation.
		:return: pulse intensity mean and std
		"""
		
		def signal_pulse_intensity_variation(signal: np.ndarray) -> tuple[float, float]:
			"""
			compute the pulse mean intensities mean and std
			:param signal: input signal
			:return: the mean and std of the signal pulses intensities.
			"""
			# detect pulses in the signal:
			pulse_detection_mod = ConvolutionPulseDetection(signal=signal, pulse_width=self.pulse_width)
			
			# Extract the pulses intensity values:
			pulses_values = []
			for center in pulse_detection_mod.convolution_peaks[1:-1]:
				start = center - self.pulse_width // 2
				end = start + self.pulse_width
				if start < 0 or end >= len(signal):
					continue
				pulses_values.append(signal[start:end])
			pulses_values = np.array(pulses_values)
			
			# compute each pulse intensity mean value:
			pulses_mean = np.mean(pulses_values, axis=0)
			
			# compute the intensity values its mean and std
			return float(np.mean(pulses_mean)), float(np.std(pulses_mean) + 1)
		
		pulse_mean_std = [signal_pulse_intensity_variation(self.df[col].values) for col in self.df.columns]
		self.rank_df.loc['pulse_intensity_mean'] = [res[0] for res in pulse_mean_std]
		self.rank_df.loc['pulse_intensity_std'] = [res[1] for res in pulse_mean_std]
		self.rank_df.loc['coefficient_of_variation'] = [res[1] / res[0] for res in pulse_mean_std]
		
		
		pass


if __name__ == "__main__":
	
	RAW_DATA_DIR_PATH = r'C:\Work\dym\2025-01-20 A2 SN1 stability\raw'
	DATA_FILE_NAME = 'raw_112'

	# file metadata:
	SAMPLE_RATE = 100  # [khz]
	PRF = 1  # [khz]
	DUTY_CYCLE = 50  # [1, 100]
	
	# Read a data file:
	exp_data_rank = ExperimentDataRank(file_path=os.path.join(RAW_DATA_DIR_PATH, f"{DATA_FILE_NAME}.csv"),
	                                   sample_rate=SAMPLE_RATE,
	                                   prf=PRF,
	                                   duty_cycle=DUTY_CYCLE)
	exp_data_assessment = exp_data_rank.rank_df
	exit(0)
	