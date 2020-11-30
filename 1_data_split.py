# In this module annotated data is split into training and test data sets in k-fold cross-validation style

import pandas as pd
import numpy as np
from pandas import ExcelWriter

# Import shuffled annotated data into a data frame
sample_data = 'data/1_Big_annotated_shuffled.xlsx'  # Location of the annotated texts
df = pd.read_excel(sample_data, sheet_name='Sheet1', header=0, index_col=0, keep_default_na=True)

# Taking k as 5, split data into 5 parts of 1000 rows and prepare 5 train/test pairs
listOfDfs = [df.loc[idx] for idx in np.split(df.index,5)]

# First train/test pair
frames = [listOfDfs[0], listOfDfs[1], listOfDfs[2], listOfDfs[3]]
df_train_1 = pd.concat(frames)
df_test_1 = listOfDfs[4]

writer_sample = ExcelWriter('data/2_train_sample_1.xlsx')
df_train_1.to_excel(writer_sample, 'Sheet1')
writer_sample.save()

writer_sample = ExcelWriter('data/2_test_sample_1.xlsx')
df_test_1.to_excel(writer_sample, 'Sheet1')
writer_sample.save()

# Second train/test pair
frames = [listOfDfs[0], listOfDfs[1], listOfDfs[2], listOfDfs[4]]
df_train_2 = pd.concat(frames)
df_test_2 = listOfDfs[3]

writer_sample = ExcelWriter('data/2_train_sample_2.xlsx')
df_train_2.to_excel(writer_sample, 'Sheet1')
writer_sample.save()

writer_sample = ExcelWriter('data/2_test_sample_2.xlsx')
df_test_2.to_excel(writer_sample, 'Sheet1')
writer_sample.save()

# Third train/test pair
frames = [listOfDfs[0], listOfDfs[1], listOfDfs[3], listOfDfs[4]]
df_train_3 = pd.concat(frames)
df_test_3 = listOfDfs[2]

writer_sample = ExcelWriter('data/2_train_sample_3.xlsx')
df_train_3.to_excel(writer_sample, 'Sheet1')
writer_sample.save()

writer_sample = ExcelWriter('data/2_test_sample_3.xlsx')
df_test_3.to_excel(writer_sample, 'Sheet1')
writer_sample.save()

# Forth train/test pair
frames = [listOfDfs[0], listOfDfs[2], listOfDfs[3], listOfDfs[4]]
df_train_4 = pd.concat(frames)
df_test_4 = listOfDfs[1]

writer_sample = ExcelWriter('data/2_train_sample_4.xlsx')
df_train_4.to_excel(writer_sample, 'Sheet1')
writer_sample.save()

writer_sample = ExcelWriter('data/2_test_sample_4.xlsx')
df_test_4.to_excel(writer_sample, 'Sheet1')
writer_sample.save()

# Fifth train/test pair
frames = [listOfDfs[1], listOfDfs[2], listOfDfs[3], listOfDfs[4]]
df_train_5 = pd.concat(frames)
df_test_5 = listOfDfs[0]

writer_sample = ExcelWriter('data/2_train_sample_5.xlsx')
df_train_5.to_excel(writer_sample, 'Sheet1')
writer_sample.save()

writer_sample = ExcelWriter('data/2_test_sample_5.xlsx')
df_test_5.to_excel(writer_sample, 'Sheet1')
writer_sample.save()