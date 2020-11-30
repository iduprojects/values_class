# Данный модуль предназначен для предварительной обработки исходных данных, используемых для обучения и тестирования
# текстового классификатора нейросетевой языковой модели, а также для данных, на которых применяется обученная модель

import spacy
import pandas as pd
import re
from nltk.corpus import stopwords
from pandas import ExcelWriter

# Загрузка исходных данных
sample_data = 'data/VK_data.xlsx'  # Путь к исходным данным
df_sample = pd.read_excel(sample_data, sheet_name='Sheet1', header=0, index_col=False, keep_default_na=True)
df_sample = df_sample.dropna()

# Загрузка исходной русскоязычной нейросетевой языковой модели
nlp = spacy.load('spacy-ru/ru2')  # Путь к исходной модели
nlp.add_pipe(nlp.create_pipe('sentencizer'), first=True)

# Функция подготовки текстов к предобработке средствами NLP
def preprocess(text):
    # Добавить пробелы после «!»,«?»,«.»,«,»,«;»,«/» и скобок для более качественного разбиения текстов на предложения
    text = re.sub('([.!?(),;/])', r'\1 ', text)
    # Удалить хештеги в начале и середине предложений
    text = re.sub("[\#].*?[\ ]", " ", text)
    # Удалить хештеги в конце предложений
    text = text.partition("#")[0]
    # Проблема: в некоторых текстах русские символы заменены на идентично выглядящие английские символы (в целях защиты от плагиата?)
    # Например, так может выглядеть строка после удаления из нее всех англоязычных символов:
    # «Змечательный мльтфильм, ктый в увлекательнй фме пзнакмит детей интментми имфничекг ркета»
    # Решение: заменить все англоязычные символы, имеющие "аналоги" в русском языке, на русскоязычные символы, а прочие
    # английские символы удалить позже, после токенизации
    text = re.sub('a', 'а', text)
    text = re.sub('c', 'с', text)
    text = re.sub('e', 'е', text)
    text = re.sub('o', 'о', text)
    text = re.sub('p', 'р', text)
    text = re.sub('y', 'у', text)
    text = re.sub('x', 'х', text)
    text = re.sub('A', 'А', text)
    text = re.sub('B', 'В', text)
    text = re.sub('C', 'С', text)
    text = re.sub('E', 'Е', text)
    text = re.sub('H', 'Н', text)
    text = re.sub('K', 'К', text)
    text = re.sub('M', 'М', text)
    text = re.sub('O', 'О', text)
    text = re.sub('P', 'Р', text)
    text = re.sub('T', 'Т', text)
    text = re.sub('X', 'Х', text)
    # Удалить цифры
    text = re.sub(r'\d+', '', text)
    # Удалить специальные символы
    text = re.sub(r'["=_+*&^@»«„“•/\']', ' ', text)
    # Удалить множественные пробелы
    text = re.sub('\s{2,}', ' ', text)
    return(text)


# Функция предобработки текстов средствами NLP

# Проверка наличия русскоязычных символов в строке
cyr_check = re.compile(r'[а-яА-Я]')
# Список русскоязычных стоп-слов из библиотеки NLTK
stop_words = stopwords.words("russian")
# Допустимые для анализа части речи
POS = ['NOUN', 'VERB', 'PROPN', 'ADJ', 'ADV']

def normalizer(texts):
    results = []  # список для хранения обработанных текстов
    for t in texts:
        doc = nlp(t)
        text = []  # список для хранения слов в нормализованном тексте
        for token in doc:
            if token.pos_ in POS:  # удаление неинтересующих частей речи
                # Лемматизация и приведение токенов к нижнему регистру
                token = token.lemma_.lower()
                # Удаление стоп-слов, токенов без русскоязычных символов, токенов с английскими буквами
                # (в рамках решения вышеобозначенной проблемы) и токенов, состоящих из одного символа
                if token not in stop_words and re.search('[a-zA-Z]', token) is None and len(token)>1 and cyr_check.match(token):
                    text.append(token)
        # Удаление пунктуации и служебных символов
        results.append(re.sub(r'[A-Za-z.,;:?!"()-=_+*&^@/\'’]', ' ', ' '.join(text)))
    return results

# Подготовить исходные тексты к предобработке средствами NLP
df_sample['Text_cleaned'] = df_sample['Text'].apply(preprocess)
texts_list = df_sample['Text_cleaned'].to_list()

# Предобработать тексты средствами NLP
texts_norm = normalizer(texts_list)
df_sample['Text_NORM'] = texts_norm

df_sample.drop('Text', inplace=True, axis=1)
df_sample.drop('Text_cleaned', inplace=True, axis=1)

# Перемешать/рандомизировать массив обработанных текстов
df_sample = df_sample.sample(frac=1)
df_sample = df_sample.sample(frac=1)
df_sample = df_sample.sample(frac=1)

# Сохранить обработанные данные
writer_sample = ExcelWriter('data/VK_data_preprocessed.xlsx')
df_sample.to_excel(writer_sample, 'Sheet1')
writer_sample.save()

