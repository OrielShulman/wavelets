import matplotlib.pyplot as plt
import numpy as np
import pywt

# TODO: after finding local extrema, filter the coefficients by their value (strongest coefficients for detection)


class SignalSWTPulseDetection:
	"""
	detect pulses start and end in given signal by inspecting its SWT details coefficients
	"""
	
	def __init__(self, signal: np.ndarray, wavelet: str = 'haar', level: int = None):
		"""
		initiate class instance
		:param signal: input signal
		:param wavelet: wavelet method
		:param level: SWT depth level to perform analysis on
		"""
		self.signal:np.ndarray = signal
		self.wavelet: str = wavelet
		self.level: int = level
		
		self.pulse_start_idx: np.ndarray = np.array([])
		self.pulse_end_idx: np.ndarray = np.array([])
		
		self.detect_pulse_edges()
		
		self.display_pulse_detection()
		
	def detect_pulse_edges(self) -> None:
		"""
		computes SWT on the signal and returns the detail coefficients
		:return:
		"""
		# Perform SWT decomposition:
		coefficients = pywt.swt(data=self.signal, wavelet=self.wavelet, level=self.level)
		cd = coefficients[0][1]  # details coefficients at the last SWT computed level
		cd_abs = np.abs(cd)
		
		# Find extrema in the detail coefficients
		extrema = np.where((cd_abs[1:-1] > cd_abs[:-2]) & (cd_abs[1:-1] > cd_abs[2:]))[0] + 1
		
		for ext_index in extrema:
			if cd[ext_index] < 0:
				self.pulse_start_idx = np.append(self.pulse_start_idx, ext_index)
			else:
				self.pulse_end_idx = np.append(self.pulse_end_idx, ext_index)
		
	def display_pulse_detection(self) -> None:
		"""
		display the detection result:
		"""
		plt.figure(figsize=(10, 6))
		plt.suptitle(f"SWT {self.wavelet} pulse detection with SWT level: {self.level}", fontweight='bold')
		
		plt.plot(self.signal, label='Signal')
		
		for start in self.pulse_start_idx:
			plt.axvline(x=start, color='green', linestyle='--')
		plt.axvline(x=self.pulse_start_idx[0], color='green', linestyle='--', label='Pulse Start')
		
		for end in self.pulse_end_idx:
			plt.axvline(x=end, color='red', linestyle='--')
		plt.axvline(x=self.pulse_end_idx[0], color='red', linestyle='--', label='Pulse End')
		
		plt.legend()
		plt.title('Signal with Pulse Start and End')
		plt.xlabel('Sample Index')
		plt.ylabel('Amplitude')
		plt.tight_layout()
		plt.show()
		