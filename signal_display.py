import matplotlib.pyplot as plt
import numpy as np


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