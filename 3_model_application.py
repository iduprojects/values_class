# In this module the pretrained model is applied to the cleaned data from SN, the results are further interpreted to
# and the assumptions are made about the values of the group of people who created the analyzed texts

import spacy
import pandas as pd
from pandas import ExcelWriter
import matplotlib
import matplotlib.pyplot as plt

# Import cleaned data to be analysed into a data frame
sample_data = 'data/VK_data_cleaned.xlsx'  # Location of the cleaned texts
df_sample = pd.read_excel(sample_data, sheet_name='Sheet1', header=0, index_col=False, keep_default_na=True)
df_sample = df_sample.dropna()

# Load trained
nlp = spacy.load('C:/Practice2020/values_nlp/models/model_13_fin')

# Delete text with less than 4 words
df_sample = df_sample[df_sample['Text_NORM'].str.split().apply(len) > 3]

# Reduce long texts to 150 words
df_sample['Text_NORM'] = df_sample['Text_NORM'].str.split().str[:150]
df_sample['Text_NORM'] = df_sample['Text_NORM'].str.join(' ')

count_row = df_sample.shape[0]
print('Number of rows:' + str(count_row))

# Classify each text and save classification results in the lists corresponding to each of the categories

# Create a list of categories
categories = ['Housing', 'Education', 'Health', 'Religion', 'Public_transportation', 'Selfcare', 'Groceries', 'Finance',
              'Domestic_services', 'Pets', 'Sports', 'Entertainment_and_culture', 'Work']

# Create a list of values for each category
cat_lists = {key:[] for key in categories}

# Perform classification
texts = df_sample['Text_NORM'].tolist()

for text in texts:
    # Apply the model to the text
    doc = nlp(text)
    # Interpret and save the classification results to the corresponding lists
    for label, score in doc.cats.items():
        if score < 0.6:
            cat_lists[label].append(0)
        else:
            cat_lists[label].append(1)

# Save the classification results
df_results = pd.DataFrame.from_dict(cat_lists, orient='index').transpose()
df_results['Text_NORM'] = texts

writer_sample = ExcelWriter('data/VK_data_results_0_6.xlsx')
df_results.to_excel(writer_sample, 'Sheet1')
writer_sample.save()

'''sample_data = 'data/VK_data_results_0_6.xlsx'  # Location of the cleaned texts
df = pd.read_excel(sample_data, sheet_name='Sheet1', header=0, index_col=False, keep_default_na=True)'''

df_results.drop('Text_NORM', inplace=True, axis=1)

# Calculate number of assumed texts under each category
sums = df_results.sum()
df_sums = pd.DataFrame({'Cat': sums.index, 'Number': sums.values})

# Find proportion of each category relatively to the max category
max_value = df_sums['Number'].max()
df_sums['Props'] = (df_sums['Number']/max_value) * 100
df_sums.Props = df_sums.Props.round()
print(df_sums)

# Plot the values as proportions of the max value

# Bar chart
ax_pop = df_sums.plot(kind='bar', x='Cat', y='Props')
ax_pop.set_title('Ценности группы МЦ Квадрат\n')
mylabels = [' ']
ax_pop.legend(labels=mylabels)
ax_pop.set_ylabel('')
ax_pop.set_xlabel('')

plt.show()






