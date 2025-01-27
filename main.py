from signal import Signal
import signal_display

if __name__ == "__main__":
	signal = Signal.pulse_1d(n_samples=256, pulse_start=64, pulse_duration=32, amplitude=10)
	un_signal = Signal.add_uniform_noise_1d(signal=signal)
	gn_signal = Signal.add_gaussian_noise_1d(signal=signal)
	
	signal_display.display_signals_together(signals=[signal,
	                                                 un_signal,
	                                                 gn_signal],
	                                        titles=["original pulse signal",
	                                                "signal with uniform noise",
	                                                "signal with gaussian noise"])
