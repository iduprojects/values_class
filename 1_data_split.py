# Данный модуль предназначен для разделения аннотированных данных на обучающие и тестовые выборки для тестирования
# модели методом перекрестной проверки (k-fold cross-validation method), k = 5

import pandas as pd
import numpy as np
from pandas import ExcelWriter

# Загрузка предобработанных и рандомизированных аннотированнх данных
sample_data = 'data/1_annotated_shuffled.xlsx'  # Путь к данным
df = pd.read_excel(sample_data, sheet_name='Sheet1', header=0, index_col=0, keep_default_na=True)

# Принимая k = 5, разделить данные на 5 частей и подготовить 5 пар обучающих/текстовых выборок
listOfDfs = [df.loc[idx] for idx in np.split(df.index, 5)]

# Первая пара обучающих/тестовых выборок
frames = [listOfDfs[0], listOfDfs[1], listOfDfs[2], listOfDfs[3]]
df_train_1 = pd.concat(frames)
df_test_1 = listOfDfs[4]

writer_sample = ExcelWriter('data/2_train_sample_1.xlsx')
df_train_1.to_excel(writer_sample, 'Sheet1')
writer_sample.save()

writer_sample = ExcelWriter('data/2_test_sample_1.xlsx')
df_test_1.to_excel(writer_sample, 'Sheet1')
writer_sample.save()

# Вторая пара обучающих/тестовых выборок
frames = [listOfDfs[0], listOfDfs[1], listOfDfs[2], listOfDfs[4]]
df_train_2 = pd.concat(frames)
df_test_2 = listOfDfs[3]

writer_sample = ExcelWriter('data/2_train_sample_2.xlsx')
df_train_2.to_excel(writer_sample, 'Sheet1')
writer_sample.save()

writer_sample = ExcelWriter('data/2_test_sample_2.xlsx')
df_test_2.to_excel(writer_sample, 'Sheet1')
writer_sample.save()

# Третья пара обучающих/тестовых выборок
frames = [listOfDfs[0], listOfDfs[1], listOfDfs[3], listOfDfs[4]]
df_train_3 = pd.concat(frames)
df_test_3 = listOfDfs[2]

writer_sample = ExcelWriter('data/2_train_sample_3.xlsx')
df_train_3.to_excel(writer_sample, 'Sheet1')
writer_sample.save()

writer_sample = ExcelWriter('data/2_test_sample_3.xlsx')
df_test_3.to_excel(writer_sample, 'Sheet1')
writer_sample.save()

# Четвертая пара обучающих/тестовых выборок
frames = [listOfDfs[0], listOfDfs[2], listOfDfs[3], listOfDfs[4]]
df_train_4 = pd.concat(frames)
df_test_4 = listOfDfs[1]

writer_sample = ExcelWriter('data/2_train_sample_4.xlsx')
df_train_4.to_excel(writer_sample, 'Sheet1')
writer_sample.save()

writer_sample = ExcelWriter('data/2_test_sample_4.xlsx')
df_test_4.to_excel(writer_sample, 'Sheet1')
writer_sample.save()

# Пятая пара обучающих/тестовых выборок
frames = [listOfDfs[1], listOfDfs[2], listOfDfs[3], listOfDfs[4]]
df_train_5 = pd.concat(frames)
df_test_5 = listOfDfs[0]

writer_sample = ExcelWriter('data/2_train_sample_5.xlsx')
df_train_5.to_excel(writer_sample, 'Sheet1')
writer_sample.save()

writer_sample = ExcelWriter('data/2_test_sample_5.xlsx')
df_test_5.to_excel(writer_sample, 'Sheet1')
writer_sample.save()