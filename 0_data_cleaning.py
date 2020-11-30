# In this module annotated texts are cleaned and preprocessed for better quality of further model training and testing

import spacy
import pandas as pd
import re
from nltk.corpus import stopwords
from pandas import ExcelWriter

# Import data to be cleaned into a data frame
sample_data = 'data/VK_data.xlsx'  # Location of the annotated texts
df_sample = pd.read_excel(sample_data, sheet_name='Sheet1', header=0, index_col=False, keep_default_na=True)
df_sample = df_sample.dropna()

# Load pretrained Russian model
nlp = spacy.load('C:/Practice2020/values_nlp/spacy-ru/ru2')  # Location of the initial model
nlp.add_pipe(nlp.create_pipe('sentencizer'), first=True)

# Function to prepare texts for NLP preprocessing
def preprocess(text):
    # Adding spaces after «!», «?», «.», «,», «;», «/» and brackets for better performance of the sentencizer
    text = re.sub('([.!?(),;/])', r'\1 ', text)
    # Deleting hashtags in the beginning and the middle of the sentences
    text = re.sub("[\#].*?[\ ]", " ", text)
    # Deleting hashtags in the ending of the sentences
    text = text.partition("#")[0]
    # Problem: in some texts russian symbols are substituted by similar-looking english symbols (for antiplagiarism purposes?)
    # Example after deleting all english symbols in a "russian-looking" string:
    # «Змечательный мльтфильм, ктый в увлекательнй фме пзнакмит детей интментми имфничекг ркета»
    # Solution: change all russian-looking english symbols to russian symbols and delete actual english words later,
    # after tockenization
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
    # Deleting digits
    text = re.sub(r'\d+', '', text)
    # Deleting special symbols
    text = re.sub(r'["=_+*&^@»«„“•/\']', ' ', text)
    # Collapsing multiple spaces
    text = re.sub('\s{2,}', ' ', text)
    return(text)


# Function to normalize texts with NLP

# Search if string has russian symbols
cyr_check = re.compile(r'[а-яА-Я]')
# List of russian stop-words from the NLTK library
stop_words = stopwords.words("russian")
# Allowed parts of speech
POS = ['NOUN', 'VERB', 'PROPN', 'ADJ', 'ADV']

def normalizer(texts):
    results = []  # list of normalized texts (will be returned from the function as a result of normalization)
    for t in texts: # iterating through each text
        doc = nlp(t)
        text = []  # list of words in normalized text
        for token in doc:  # for each word in text
            if token.pos_ in POS:  # deleting insignificant parts of speech
                # Lemmatization and lowercasing of the token
                token = token.lemma_.lower()
                # Deleting stop-words, tokens without russian symbols, tokens with Eng letters
                # (part of the solution of the above problem) and tokens containing only 1 character
                if token not in stop_words and re.search('[a-zA-Z]', token) is None and len(token)>1 and cyr_check.match(token):
                    text.append(token)
        # Deleting punctuation
        results.append(re.sub(r'[A-Za-z.,;:?!"()-=_+*&^@/\'’]', ' ', ' '.join(text)))
    return results

# Prepare texts for NLP
df_sample['Text_cleaned'] = df_sample['Text'].apply(preprocess)
texts_list = df_sample['Text_cleaned'].to_list()

# Normalize texts
texts_norm = normalizer(texts_list)
df_sample['Text_NORM'] = texts_norm

df_sample.drop('Text', inplace=True, axis=1)
df_sample.drop('Text_cleaned', inplace=True, axis=1)

# Shuffle normalized data
df_sample = df_sample.sample(frac=1)
df_sample = df_sample.sample(frac=1)
df_sample = df_sample.sample(frac=1)

# Save normalized and shuffled data sample
writer_sample = ExcelWriter('data/VK_data_cleaned.xlsx')
df_sample.to_excel(writer_sample, 'Sheet1')
writer_sample.save()

