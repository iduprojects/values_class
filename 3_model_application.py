# Данный модуль предназначен для классификации предварительно обработанных текстов из социальной сети и интерпретации
# результатов  в виде структуры значимостей городских функций для группы людей, участвовавших в создании
# исходных текстов

import spacy
import pandas as pd
from pandas import ExcelWriter
import matplotlib.pyplot as plt


# Функция получения структуры значимостей городских функций из исходных текстов
def structure(df_texts, nlp): # df_texts - исходные тексты, nlp - языковая модель с обученным классификатором

    # Удалить тексты, в которых менее 4х слов
    df_texts = df_texts[df_texts['Text_NORM'].str.split().apply(len) > 3]

    # Сократить длинные тексты до 150 слов
    df_texts['Text_NORM'] = df_texts['Text_NORM'].str.split().str[:150]
    df_texts['Text_NORM'] = df_texts['Text_NORM'].str.join(' ')

    # Классифицировать каждый текст и сохранить результаты классификации по каждой категории в список

    # Задать список категорий
    categories = ['Housing', 'Education', 'Health', 'Religion', 'Public_transportation', 'Selfcare', 'Groceries',
                  'Finance', 'Domestic_services', 'Pets', 'Sports', 'Entertainment_and_culture', 'Work']

    # Задать пустой список для каждой категории
    cat_lists = {key: [] for key in categories}

    # Классифицировать тексты
    texts = df_texts['Text_NORM'].tolist()
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

    # Определение структуры значимостей городских функций на основании полученных результатов

    # Определить количество текстов, отнесенных к каждой из категорий
    sums = df_results.sum()
    df_structure = pd.DataFrame({'Cat': sums.index, 'Number': sums.values})

    # Рассчитать пропорцию категорий относительно максимально представленной для получения структуры значимостей
    max_value = df_structure['Number'].max()
    df_structure['Proportion'] = (df_structure['Number'] / max_value) * 100
    df_structure.Proportion = df_structure.Proportion.round()

    return df_structure


# Загрузить предварительно обработанные тексты для анализа
sample_data = 'data/VK_data_preprocessed.xlsx'
the_texts = pd.read_excel(sample_data, sheet_name='Sheet1', header=0, index_col=False, keep_default_na=True)
the_texts = the_texts.dropna()

# Загрузить обученную модель
the_nlp = spacy.load('models/model_13_fin')

# Получить структуру значимостей городских функций из текстов
results = structure(the_texts, the_nlp)

# Сохранить структуру значимостей в табличном виде
writer = ExcelWriter('data/VK_value_structure.xlsx')
results.to_excel(writer, 'Sheet1')
writer.save()

# Визуализировать структуру значимостей в виде столбчатой диаграммы
ax_pop = results.plot(kind='bar', x='Cat', y='Proportion')
ax_pop.set_title('Структура значимостей группы...\n')
mylabels = [' ']
ax_pop.legend(labels=mylabels)
ax_pop.set_ylabel('')
ax_pop.set_xlabel('')

plt.show()






