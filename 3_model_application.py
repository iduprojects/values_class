# Данный модуль предназначен для классификации предварительно обработанных текстов из социальной сети и интерпретации
# результатов  для предположения структуры ценностей группы людей, участвовавших в создании исхожных текстов

import spacy
import pandas as pd
from pandas import ExcelWriter
import matplotlib
import matplotlib.pyplot as plt

# Загрузить предварительно обработанные тексты для анализа
sample_data = 'data/VK_data_preprocessed.xlsx'
df_sample = pd.read_excel(sample_data, sheet_name='Sheet1', header=0, index_col=False, keep_default_na=True)
df_sample = df_sample.dropna()

# Загрузить обученную модель
nlp = spacy.load('models/model_13_fin')

# Удалить тексты, в которых менее 4х слов
df_sample = df_sample[df_sample['Text_NORM'].str.split().apply(len) > 3]

# Сократить длинные тексты до 150 слов
df_sample['Text_NORM'] = df_sample['Text_NORM'].str.split().str[:150]
df_sample['Text_NORM'] = df_sample['Text_NORM'].str.join(' ')

count_row = df_sample.shape[0]
print('Number of rows:' + str(count_row))

# Классифицировать каждый текст и сохранить результаты классификации по каждой категории в список

# Задать список категорий
categories = ['Housing', 'Education', 'Health', 'Religion', 'Public_transportation', 'Selfcare', 'Groceries', 'Finance',
              'Domestic_services', 'Pets', 'Sports', 'Entertainment_and_culture', 'Work']

# Задать пустой список для каждой категории
cat_lists = {key:[] for key in categories}

# Классифицировать тексты
texts = df_sample['Text_NORM'].tolist()
for text in texts:
    # Применить модель к тексту
    doc = nlp(text)
    # Интерпретировать результаты классификации и сохранить в соответствующий категории список
    for label, score in doc.cats.items():
        if score < 0.6:
            cat_lists[label].append(0)
        else:
            cat_lists[label].append(1)

# Сохранить результаты классификации
df_results = pd.DataFrame.from_dict(cat_lists, orient='index').transpose()
df_results['Text_NORM'] = texts # Добавляется только для сохранения и последующей визуальной проверки

writer_sample = ExcelWriter('data/VK_data_results_0_6.xlsx')
df_results.to_excel(writer_sample, 'Sheet1')
writer_sample.save()

#sample_data = 'data/VK_data_results_0_6.xlsx'
#df = pd.read_excel(sample_data, sheet_name='Sheet1', header=0, index_col=False, keep_default_na=True)

df_results.drop('Text_NORM', inplace=True, axis=1)

# Определить количество текстов, отнесенных к каждой из категорий
sums = df_results.sum()
df_sums = pd.DataFrame({'Cat': sums.index, 'Number': sums.values})

# Рассчитать пропорцию каждой категории относительно максимально представленной категории
max_value = df_sums['Number'].max()
df_sums['Props'] = (df_sums['Number']/max_value) * 100
df_sums.Props = df_sums.Props.round()
print(df_sums)

# Вывести на диаграмму значения для каждой категории пропорционально максимально представленной категории

# Столбчатая диаграмма
ax_pop = df_sums.plot(kind='bar', x='Cat', y='Props')
ax_pop.set_title('Ценности группы...\n')
mylabels = [' ']
ax_pop.legend(labels=mylabels)
ax_pop.set_ylabel('')
ax_pop.set_xlabel('')

plt.show()






