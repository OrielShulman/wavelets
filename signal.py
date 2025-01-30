import numpy as np


class Signal:
	"""
	a class for generating 1D signals.
	"""
	
	@staticmethod
	def pulse_1d(n_samples: int, pulse_start: int, pulse_duration: int, amplitude: float) -> np.ndarray:
		"""
		generates a 1D pulse signal with base amplitude of 0.
		:param n_samples: total number of samples in the signal.
		:param pulse_start: starting point of the pulse.
		:param pulse_duration: duration of the pulse.
		:param amplitude: pulse amplitude.
		:return: a 1D pulse signal
		"""
		signal = np.zeros(n_samples)
		signal[pulse_start:pulse_start+pulse_duration] = amplitude
		return signal
	
	@staticmethod
	def add_gaussian_noise_1d(signal: np.ndarray, mean: float = 0, std: float = 1) -> np.ndarray:
		"""
		adds a gaussian noise to an existing 1D signal
		:param signal: input signal to add noise to
		:param mean: noise mean
		:param std: noise std
		:return: the signal with added noise
		"""
		noise = np.random.normal(loc=mean, scale=std, size=signal.shape)
		return signal + noise
	
	@staticmethod
	def add_uniform_noise_1d(signal: np.ndarray, low: float = -1, high: float = 1):
		"""
		adds a uniform noise to an existing 1D signal
		:param signal: input signal to add noise to
		:param low: The lower bound of the uniform noise
		:param high: The upper bound of the uniform noise
		:return: the signal with added noise
		"""
		noise = np.random.uniform(low=low, high=high, size=signal.shape)
		return signal + noise
	
	@staticmethod
	def compute_signal_energy(signal: np.ndarray) -> float:
		"""
		computes the total energy of a signal.
		:param signal: signal input
		:return: signal energy (sum of squares of the signal)
		"""
		e = float(np.sum(np.square(signal)))
		return e
