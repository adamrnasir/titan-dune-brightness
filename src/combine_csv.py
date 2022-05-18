import os
import glob
import pandas as pd

os.chdir('runs/run_05182022_022828')

extension = 'csv'
all_filenames = [i for i in glob.glob('*.{}'.format(extension))]

inselberg_filenames = [i for i in all_filenames if "inselberg" in i]
mountain_filenames = [i for i in all_filenames if "mountain" in i]

inselberg_combined = pd.concat([pd.read_csv(f) for f in inselberg_filenames])
# Remove lines with NaN values
inselberg_combined.dropna(inplace=True)
inselberg_combined.to_csv("inselberg_combined.csv", index=False)

mountain_combined = pd.concat([pd.read_csv(f) for f in mountain_filenames])
# Remove lines with NaN values
mountain_combined.dropna(inplace=True)
mountain_combined.to_csv("mountain_combined.csv", index=False)