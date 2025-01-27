# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np


def display_signals_together(signals: list[np.ndarray], titles: list[str]) -> None:
	"""
	Displays multiple 1D signals in the same figure with different colors and names in the legend.

	Parameters:
	-----------
	signals : List[npt.NDArray[np.float64]]
		A list of 1D signals to display.
	titles : List[str]
		A list of titles for the signals.
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
	