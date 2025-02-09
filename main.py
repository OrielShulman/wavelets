import signal_display
import signal_display as sd
from signal import Signal
from signal_wavelet_analysis import SignalWaveletAnalysis
from raw_data_file import RawDataFile
from signal_swt_pulse_detection import SignalSWTPulseDetection

import numpy as np
import matplotlib.pyplot as plt
import pywt
import os

# TODO: Present:
#  - Tools (detection, noise estimation)
#  - Results.
#  - SWT map (cone of influence).
#  - Incoherent integration.

# TODO: talk to Ran about:
#  - What is the attached metadata.
#  - What experiment ranking would be useful for.
#  - Plan Sunday:
#       - Who would i work with on integration.
#       - What can i do preciously.
#       - How does their system integrate with code (services, IO, cloud, Data management.. ).

# TODO: Implement pulse intensity over time.
# TODO: branch the code into 'dev' branch (consult with Micha).
# TODO: Read about 'haar' Vs. 'Daubechies'.
# TODO: Data - find good signals.
# TODO: (Q) Should i trim the edges of the detected pulses in the detection module?


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
	# sd.display_2_signals(signals=[signal, np.abs(signal - reconstructed_signal)],
	# 					 titles=["Original Signal", "| Original Signal - IDWT |"])
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
	# sd.display_2_signals(signals=[signal, np.abs(signal - reconstructed_signal)],
	# 					 titles=["Original Signal", "| Original Signal - ISWT |"])
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
	plt.subplot(n + 1, 1, 1)
	plt.plot(signal)
	plt.title('Original Signal')
	for i in range(n):
		plt.subplot(n + 1, 1, i + 2)
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
	plt.figure(figsize=(12, n * 2))
	plt.suptitle('DWT\n')
	plt.subplot(n, 1, 1)
	plt.plot(signal)
	plt.title('Original Signal')
	for i in range(1, n):
		plt.subplot(n, 1, i + 1)
		plt.plot(coeffs[i], marker='x', markerfacecolor='r', markeredgecolor='r')
		plt.xticks([])
		plt.title(f'Approximation Coefficients - Level {n - i}')
	plt.tight_layout()
	plt.show()


def iterate_raw_data_files(data_dir_path: str) -> iter:
	"""
	iterates over all the data raw files as was given in jan 2025
	:param data_dir_path: target directory of the raw data files
	:return: an iterator of the data files
	"""
	# Iterate over files in the target directory without changing the working directory
	for file_name in os.listdir(data_dir_path):
		file_path = os.path.join(data_dir_path, file_name)
		
		if os.path.isfile(file_path) and file_name.endswith('.csv'):
			data = RawDataFile(file_path=file_path)
			yield data


def read_raw_data_file(data_dir_path: str, file_name: str) -> RawDataFile:
	"""
	reads a specified .csv datafile
	:param data_dir_path: target directory of the raw data files
	:param file_name: target file to open
	:return: the opened file
	"""
	file_path = os.path.join(data_dir_path, f"{file_name}.csv")
	
	# Check if the file exists and is a file
	if os.path.isfile(file_path):
		return RawDataFile(file_path=file_path)
	
	else:
		print(f"File '{file_name}' not found or is not a .csv file.")
		return None


def example_swt_dwt_raw_data(raw_data: RawDataFile) -> None:
	"""
	apply on the sample SWT and DWT and display their scalogram.
	:param raw_data: a raw data file module
	:return:
	"""
	# Extract the main pd sample and slice it:
	df = raw_data.df[['main_current']][:512]
	# Display the signal raw:
	# signal_display.plod_pandas_df(df=df, plot_title=f"data file: {raw_data.filename}")
	# Apply SWT:
	# example_multy_level_haar_swt(signal=df.iloc[:, 0].values, signal_info=f"{raw_data.filename}: main_current")
	# Apply DWT:
	example_multy_level_haar_dwt(signal=df.iloc[:, 0].values, signal_info=f"{raw_data.filename}: main_current")
	return


def example_pulse_detection(signal: np.ndarray) -> None:
	"""
	apply pulse detection using SWT
	:param signal: pulse signal
	"""
	SignalSWTPulseDetection(signal=signal, wavelet='haar')
	SignalSWTPulseDetection(signal=signal, wavelet='haar', level=1)


def zoom_in_data_signal(raw_data: RawDataFile) -> None:
	"""
	read out of the data a segment of the signals and count how many samples are per pulse
	"""
	signals_df = raw_data.df
	s_pd, s_current = raw_data.split_pd_current_columns()
	s = np.power(2, 8)
	w = np.power(2, 8)
	
	short_signal_df = s_current[s:s+w]
	
	# main_pd_df = data_example.df[['main_pd']][:np.power(2, 8)].iloc[:, 0].values
	
	# Plot each column in the DataFrame
	num_signals = short_signal_df.shape[1]
	fig, axes = plt.subplots(num_signals, 1, figsize=(16, 3 * num_signals))
	if num_signals == 1:
		axes = [axes]
	
	for i, column in enumerate(short_signal_df.columns):
		axes[i].plot(short_signal_df.index, short_signal_df[column],
		             marker='x',
		             markeredgecolor='r',
		             markerfacecolor='r',
		             label=column)
		axes[i].set_title(f'Signal {i + 1}: {column}')
		axes[i].grid(True)
		axes[i].legend(loc='center right')
		# axes[i].set_xlabel('Sample')
		# axes[i].set_ylabel('Value')
	
	# Add legend, title, and labels
	fig.suptitle(f"pulse zoom in for file: {data_example.filename}", fontweight='bold')
	plt.legend()
	plt.tight_layout()
	plt.show()


if __name__ == "__main__":
	RAW_DATA_DIR_PATH = r'C:\Work\dym\2025-01-20 A2 SN1 stability\raw'
	
	# Generate Pulse signal:
	pulse_signal_params = {'n_samples': np.power(2, 7),
	                       'pulse_start': np.power(2, 5) + 1,
	                       'pulse_width': np.power(2, 5),
	                       'pulse_amplitude': 10}
	base_pulse_signal = Signal.pulse_1d(n_samples=pulse_signal_params['n_samples'],
	                                    pulse_start=pulse_signal_params['pulse_start'],
	                                    pulse_duration=pulse_signal_params['pulse_width'],
	                                    amplitude=pulse_signal_params['pulse_amplitude'])
	un_signal = Signal.add_uniform_noise_1d(signal=base_pulse_signal)
	gn_signal = Signal.add_gaussian_noise_1d(signal=base_pulse_signal)
	example_signal_s = np.sin(2 * np.pi * 7 * np.linspace(0, 1, 400)) + np.sin(2 * np.pi * 13 * np.linspace(0, 1, 400))
	
	# Execute examples:
	# example_pulse_detection(signal=un_signal)
	
	# example_single_level_haar_dwt(signal=base_pulse_signal)
	#
	# example_multy_level_haar_dwt(signal=base_pulse_signal,
	#                              signal_info=f"pulse signal [signal size: {pulse_signal_params['n_samples']}, "
	#                                          f"pulse width: {pulse_signal_params['pulse_width']}, "
	#                                          f"pulse start: {pulse_signal_params['pulse_start']}]")
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
	
	# for i, raw_data in enumerate(iterate_raw_data_files(data_dir_path=RAW_DATA_DIR_PATH)):
	# 	if i >= 5:
	# 		break
	# 	example_swt_dwt_raw_data(raw_data=raw_data)
	# 	# print("CSV File:", raw_data.filename)
	# 	# print("data shape:", raw_data.df.shape)
	# 	# # content = csv_file.read()
	
	data_example = read_raw_data_file(data_dir_path=RAW_DATA_DIR_PATH, file_name='raw_112')
	example_swt_dwt_raw_data(raw_data=data_example)
	# zoom_in_data_signal(raw_data=data_example)
	# example_pulse_detection(signal=data_example.df[['main_current']][0:1024].iloc[:, 0].values)
	
	# data_example.plot_data(seg_end=2**8)
	# df_main_pd = data_example.df[['main_pd']]
	# df_main_current = data_example.df[['main_current']]
	# pd_df, cur_df = data_example.split_pd_current_columns()
	# signal_display.plod_pandas_df(df=pd_df[:2**8], plot_title=f"data file: {data_example.filename} - pd")
	# signal_display.plod_pandas_df(df=cur_df[:2**8], plot_title=f"data file: {data_example.filename} - curent")
	# s1 = data_example.df[['main_pd', 'main_current']]
	# signal_display.plod_pandas_df(df=s1[:30], plot_title=f"data file: {data_example.filename} - curent")
	exit(0)
