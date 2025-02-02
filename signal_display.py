import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def display_signals(signals: list[np.ndarray], titles: list[str]) -> None:
	"""
	Displays multiple 1D signals in the same figure with different colors and names in the legend.
	:param signals: list of 1D signals to display.
	:param titles: list of titles for the signals.
	"""
	if len(signals) != len(titles):
		raise ValueError("The number of signals and titles must be the same.")
	
	plt.figure(figsize=(10, 6))
	
	for signal, title in zip(signals, titles):
		plt.plot(signal, label=title)
	
	plt.xlabel('Sample Index')
	plt.ylabel('Amplitude')
	plt.legend()
	plt.grid(True)
	plt.title('1D Signals')
	plt.show()


def display_2_signals(signals: list[np.ndarray], titles: list[str]) -> None:
	"""
	display 2 signals in two diferent axis on the same figure.
	:return:
	"""
	if len(signals) != len(titles) != 2:
		raise ValueError("The number of signals and titles must be the 2.")
	
	fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex='all')
	
	ax1.plot(signals[0])
	ax1.set_title(titles[0])
	ax1.set_ylabel('Amplitude')
	ax1.grid(True)
	
	ax2.plot(signals[1])
	ax2.set_title(titles[1])
	ax2.set_xlabel('Sample Index')
	ax2.set_ylabel('Amplitude')
	ax2.grid(True)
	
	plt.tight_layout()
	plt.show()


def plot_dwt_scalogram(signal: np.ndarray,
                       coeffs: np.ndarray,
                       signal_info: str,
                       wavelet_method: str = 'haar') -> None:
	"""
	plots a scaleogram of a signal DWT along the signal
	:param signal: original signal
	:param coeffs: DWT coefficients (both details and approximation coefficients)
	:param signal_info: signal information to attach with the title.
	:param wavelet_method: wavelet method used for DWT
	"""
	# Create a figure with 2 subplots
	fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex='all', gridspec_kw={'height_ratios': [2, 5, 5]})
	
	# Plot the original signal
	ax1.plot(signal)
	ax1.set_title(f"Original Signal ({signal.size} samples)")
	ax1.set_xlabel('Sample')
	ax1.set_ylabel('Amplitude')
	
	ax1.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)
	
	# Extract only the details coefficients:
	detail_coeffs = coeffs[1:]
	
	# Plot the scalogram:
	scalogram = np.zeros((len(detail_coeffs), len(signal)))
	for i, coeff in enumerate(detail_coeffs):
		scale = len(detail_coeffs) - i
		stretched_coeff = np.repeat(coeff, len(signal) // len(coeff))  # Stretch coefficients to signal size
		scalogram[scale - 1, :len(stretched_coeff)] = stretched_coeff
	cax = ax2.imshow(scalogram[::-1],
	                 extent=[0, len(signal), 0.5, len(detail_coeffs) + 0.5],
	                 aspect='auto',
	                 cmap='coolwarm',
	                 vmin=np.min(scalogram),
	                 vmax=np.max(scalogram))
	# set scalogram ticks:
	ax2.set_yticks(range(1, len(detail_coeffs) + 1))
	ax2.set_yticklabels([f'{i}' for i in range(1, len(detail_coeffs) + 1)])
	ax2.set_xlabel('Sample')
	ax2.set_ylabel('Level')
	ax2.set_title(f"Scalogram of DWT Detail Coefficients ({wavelet_method})")
	fig.colorbar(cax, ax=ax2, orientation='horizontal', label='Coefficient Value')
	
	# Plot the absolute value scalogram:
	abs_scalogram = np.zeros((len(detail_coeffs), len(signal)))
	for i, coeff in enumerate(detail_coeffs):
		scale = len(detail_coeffs) - i
		abs_stretched_coeff = np.repeat(np.abs(coeff), len(signal) // len(coeff))  # Stretch coefficients to signal size
		abs_scalogram[scale - 1, :len(abs_stretched_coeff)] = abs_stretched_coeff
	abs_cax = ax3.imshow(abs_scalogram[::-1],
	                     extent=[0, len(signal), 0.5, len(detail_coeffs) + 0.5],
	                     aspect='auto',
	                     cmap='Spectral',
	                     vmin=np.min(abs_scalogram),
	                     vmax=np.max(abs_scalogram))
	# set abs_scalogram ticks:
	ax3.set_yticks(range(1, len(detail_coeffs) + 1))
	ax3.set_yticklabels([f'{i}' for i in range(1, len(detail_coeffs) + 1)])
	ax3.set_xlabel('Sample')
	ax3.set_ylabel('Level')
	ax3.set_title(f"Absolute value Scalogram of DWT Detail Coefficients ({wavelet_method})")
	fig.colorbar(abs_cax, ax=ax3, orientation='horizontal', label='Coefficient Value')
	
	# Plot:
	fig.suptitle(f"DWT scalogram for:\n\n{signal_info}\n", fontweight='bold')
	plt.tight_layout()
	plt.show()


def plot_swt_scalogram(signal: np.ndarray,
                       coeffs: list,
                       signal_info: str,
                       wavelet_method: str = 'haar') -> None:
	"""
	plots a scaleogram of a signal SWT with the signal
	:param signal: original signal
	:param coeffs: SWT coefficients (both details and approximation coefficients)
	:param signal_info: signal information to attach with the title.
	:param wavelet_method: wavelet method used for SWT
	"""
	# Create a figure with subplots
	fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, ncols=1, figsize=(10, 16), sharex='all',
	                                         gridspec_kw={'height_ratios': [2, 5, 5, 5]})
	
	# Set the main title
	fig.suptitle(f"SWT scalogram for:\n\n{signal_info}\n", fontweight='bold')
	
	# ax1: Plot the original signal
	ax1.plot(signal)
	ax1.set_title(f"Original Signal ({signal.size} samples)")
	ax1.set_xlabel('Sample')
	ax1.set_ylabel('Amplitude')
	ax1.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)
	
	# ax2: Plot the scalogram:
	combined_coeffs = np.vstack([c[1] for c in coeffs])  # Combine all detail coefficients for visualization
	cax2 = ax2.imshow(combined_coeffs,
	                  extent=[0, len(signal), 0.5, len(coeffs) + 0.5],
	                  aspect='auto',
	                  cmap='coolwarm',
	                  vmin=np.min(combined_coeffs),
	                  vmax=np.max(combined_coeffs))
	ax2.set_yticks(range(1, len(combined_coeffs) + 1))
	ax2.set_yticklabels([f'{i}' for i in range(1, len(combined_coeffs) + 1)])
	ax2.set_xlabel('Sample')
	ax2.set_ylabel('Level')
	ax2.set_title(f"Scalogram of SWT Detail Coefficients ({wavelet_method})")
	fig.colorbar(cax2, ax=ax2, orientation='horizontal', label='Coefficient Value')
	
	# ax3: Plot the abs value scalogram:
	abs_combined_coeffs = np.abs(combined_coeffs)
	cax3 = ax3.imshow(abs_combined_coeffs,
	                  extent=[0, len(signal), 0.5, len(coeffs) + 0.5],
	                  aspect='auto',
	                  cmap='Spectral',
	                  vmin=np.min(abs_combined_coeffs),
	                  vmax=np.max(abs_combined_coeffs))
	ax3.set_yticks(range(1, len(abs_combined_coeffs) + 1))
	ax3.set_yticklabels([f'{i}' for i in range(1, len(abs_combined_coeffs) + 1)])
	ax3.set_xlabel('Sample')
	ax3.set_ylabel('Level')
	ax3.set_title(f"Absolute value Scalogram of SWT Detail Coefficients ({wavelet_method})")
	fig.colorbar(cax3, ax=ax3, orientation='horizontal', label='Coefficient Value')
	
	# ax4: Plot the scalogram in dB:
	db_combined_coeffs = 20 * np.log10(abs_combined_coeffs + np.finfo(float).eps)
	cax4 = ax4.imshow(db_combined_coeffs,
	                  extent=[0, len(signal), 0.5, len(coeffs) + 0.5],
	                  aspect='auto',
	                  cmap='magma',
	                  vmin=np.min(db_combined_coeffs),
	                  vmax=np.max(db_combined_coeffs))
	ax4.set_yticks(range(1, len(db_combined_coeffs) + 1))
	ax4.set_yticklabels([f'{i}' for i in range(1, len(db_combined_coeffs) + 1)])
	ax4.set_xlabel('Sample')
	ax4.set_ylabel('Level')
	ax4.set_title(f"logarithmic Scalogram of SWT Detail Coefficients [dB] ({wavelet_method})")
	fig.colorbar(cax4, ax=ax4, orientation='horizontal', label='Coefficient Value')
	
	plt.tight_layout()
	plt.show()


def plod_pandas_df(df: pd.DataFrame, plot_title: str) -> None:
	"""
	plots a given pandas df
	:param df: signal dataframe
	:param plot_title: title
	"""
	# Plot each column in the DataFrame
	plt.figure(figsize=(12, 6))
	plt.suptitle(plot_title, fontweight='bold')
	for column in df.columns:
		plt.plot(df.index, df[column], label=column)
	
	# Add legend, title, and labels
	plt.legend()
	plt.title('Signals')
	plt.xlabel('Sample Index')
	# plt.ylabel('Signal Value')
	plt.grid(True)
	plt.tight_layout()
	plt.show()
