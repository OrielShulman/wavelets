from convolution_pulse_detection import ConvolutionPulseDetection
from pulse_signal_noise_estimation import PulseSignalNoiseEstimation
import os
import numpy as np
import pandas as pd


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
		self.pulse_amplitude_variation()  # Compute pulse amplitude mean and std
		
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
		self.rank_df.loc['expected_pri'] = [pri_pixels] * len(self.rank_df.columns)
	
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
	
	def pulse_amplitude_variation(self) -> None:
		"""
		for each channel, check the uniformity of the pulse intensities.
		- split to pulses.
		- check each pulse mean amplitude.
		- calculate the pulse intensities deviation.
		:return: pulse amplitude mean and std
		"""
		
		def signal_pulse_amplitude_variation(signal: np.ndarray) -> tuple[float, float]:
			"""
			compute the pulse mean intensities mean and std
			:param signal: input signal
			:return: the mean and std of the signal pulses intensities.
			"""
			# detect pulses in the signal:
			pulse_detection_mod = ConvolutionPulseDetection(signal=signal, pulse_width=self.pulse_width)
			
			# Extract the pulses amplitude values:
			pulses_values = []
			for center in pulse_detection_mod.convolution_peaks[1:-1]:
				start = center - self.pulse_width // 2
				end = start + self.pulse_width
				if start < 0 or end >= len(signal):
					continue
				pulses_values.append(signal[start:end])
			pulses_values = np.array(pulses_values)
			
			# compute each pulse amplitude mean value:
			pulses_mean = np.mean(pulses_values, axis=0)
			
			# compute the amplitude values mean and std
			return float(np.mean(pulses_mean)), float(np.std(pulses_mean) + 1)
		
		pulse_mean_std = [signal_pulse_amplitude_variation(self.df[col].values) for col in self.df.columns]
		self.rank_df.loc['pulse_amp_mean'] = [res[0] for res in pulse_mean_std]
		self.rank_df.loc['pulse_amp_std'] = [res[1] for res in pulse_mean_std]
		self.rank_df.loc['pulse_amp_coefficient_of_variation'] = [res[1] / res[0] for res in pulse_mean_std]


if __name__ == "__main__":
	from datetime import datetime
	import matplotlib.pyplot as plt
	
	RAW_DATA_DIR_PATH = r'C:\Work\dym\2025-01-20 A2 SN1 stability\raw'
	RESULTS_DIR_PATH = r'C:\Work\dym\data_quality_assessments'
	DATA_FILE_NAME = 'raw_112'
	EXP_METADATA = {
		'SAMPLE_RATE': 100,     # [khz]
		'PRF': 1,               # [khz]
		'DUTY_CYCLE': 50,       # (0, 100)
	}
	
	def assess_experiment_directory(
			dir_path: str, sample_rate: int, prf: int, duty_cycle: int, save_dir_path: str) -> None:
		"""
		assess the experiment files in the experiments directory and save the assessment results to a new file.
		:param dir_path: path to the directory that contains the experiments raw data files.
		:param sample_rate: [khz]
		:param prf: [khz]
		:param duty_cycle: percentage in scale (0, 100)
		:param save_dir_path: path to save the results to
		"""
		experiments_assessment_results = []
		
		# Iterate over files in the target directory:
		for file_name in os.listdir(dir_path):
			t_start = datetime.now()
			print(f"Assessing file: {file_name} [{len(experiments_assessment_results) + 1} out of {len(os.listdir(dir_path))}]")
			file_path = os.path.join(dir_path, file_name)
			# Check if the file is a CSV file:
			if os.path.isfile(file_path) and file_path.endswith('.csv'):
				# Apply Experiment file data quality assessment:
				exp_data_res = ExperimentDataRank(file_path=file_path,
				                                  sample_rate=sample_rate,
				                                  prf=prf,
				                                  duty_cycle=duty_cycle).rank_df
				# Rearrange result as a one line:
				transformed_df = pd.DataFrame()  # Create a new dataframe with the transformed columns
				transformed_df['file_name'] = [f'{os.path.splitext(file_name)[0]}']  # Add a 'file_name' column
				for col_index in exp_data_res.columns:
					for row_index in exp_data_res.index:
						new_col_name = f'{col_index}_{row_index}'
						transformed_df[new_col_name] = [exp_data_res.at[row_index, col_index]]
				# Append the transformed dataframe to the list
				experiments_assessment_results.append(transformed_df)
				print(f"\tFinished in {(datetime.now() - t_start).total_seconds():.1f} seconds")
				
		# Concatenate all the transformed dataframes into a single dataframe
		experiments_assessment_results_df = pd.concat(experiments_assessment_results, ignore_index=True)
		
		# Save the new result df to a csv file:
		timestamp = datetime.now().strftime("%Y_%m_%d_%H%M")
		files_dir_name = os.path.basename(os.path.normpath(dir_path))
		new_filename = f"{files_dir_name}_files_assessment_results_{timestamp}.csv"
		experiments_assessment_results_df.to_csv(os.path.join(save_dir_path, new_filename), index=False)
	
	# assess_experiment_directory(dir_path=RAW_DATA_DIR_PATH,
	#                             sample_rate=EXP_METADATA['SAMPLE_RATE'],
	#                             prf=EXP_METADATA['PRF'],
	#                             duty_cycle=EXP_METADATA['DUTY_CYCLE'],
	#                             save_dir_path=RESULTS_DIR_PATH)
	
	def plot_df(df: pd.DataFrame) -> None:
		"""
		plot an example of a pandas df
		:param df: experiment rank df
		"""
		df = df.map(lambda x: f'{x:.2f}' if isinstance(x, float) else f'{int(x)}')
		
		# Plot the table
		fig, ax = plt.subplots(figsize=(15, 4))  # Set the figure size
		ax.axis('tight')
		ax.axis('off')
		table = ax.table(cellText=df.values, colLabels=df.columns, rowLabels=df.index, cellLoc='center', loc='center')
		
		# Adjust the font size
		table.auto_set_font_size(False)
		table.set_fontsize(12)
		table.scale(1, 1.7)
		table.auto_set_column_width(col=list(range(len(df.columns))))
		
		# Adjust the position of the table to fill the figure
		for key, cell in table.get_celld().items():
			cell.set_height(1 / (len(df.index) + 1))
			cell.set_width(1 / (len(df.columns) + 1))
			cell.set_edgecolor('white')  # Set the color of the lines
			cell.set_facecolor('gainsboro')  # Set the background color
		
		plt.tight_layout()
		plt.show()
		
	# Read a data file:
	exp_data_rank = ExperimentDataRank(file_path=os.path.join(RAW_DATA_DIR_PATH, f"{DATA_FILE_NAME}.csv"),
	                                   sample_rate=EXP_METADATA['SAMPLE_RATE'],
	                                   prf=EXP_METADATA['PRF'],
	                                   duty_cycle=EXP_METADATA['DUTY_CYCLE'])
	exp_data_assessment = exp_data_rank.rank_df
	plot_df(df=exp_data_assessment)
	exit(0)
	