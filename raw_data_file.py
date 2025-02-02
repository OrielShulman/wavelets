import signal_display
import pandas as pd
import numpy as np
import os


class RawDataFile:
	"""
	a class for an experiment raw data file.
	"""
	
	def __init__(self, file_path: str):
		"""
		initiate data file class, reads the data from the file into a pandas dataframe
		:param file_path: path of the data file
		"""
		self.filename: str = os.path.splitext(os.path.basename(file_path))[0]
		self.df: pd.DataFrame = pd.read_csv(file_path)
	
	def plot_data(self, seg_start: int = None, seg_end: int = None) -> None:
		"""
		plot the data with legends.
		:param seg_start: Optional, foe signal segmentation for plotting - start of segment
		:param seg_end: Optional, foe signal segmentation for plotting - end of segment
		:return: None
		"""
		seg_start = seg_start if seg_start and 0 < seg_start < len(self.df) else 0
		
		seg_end = seg_end if seg_end and seg_start < seg_end < len(self.df) else len(self.df)
		
		# Slice the DataFrame to get the desired segment
		data_to_plot = self.df.iloc[seg_start:seg_end]
		
		signal_display.plod_pandas_df(df=data_to_plot, plot_title=f"data file: {self.filename}")
		
