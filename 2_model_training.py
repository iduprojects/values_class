# In this module the model is trained and tested

import spacy
import pandas as pd
import random
from ast import literal_eval
from pandas import ExcelWriter

# Evaluation function for a particular category (label_name)
def evaluate(df_test_gold, label_name):
    predictions = []
    for index, row in df_test_gold.iterrows():
        test_text = row['Text_NORM']
        doc = nlp_my(test_text)
        for label, score in doc.cats.items():
            if label == label_name:
                if score >= 0.5:
                    predictions.append('1')
                elif score < 0.5:
                    predictions.append('0')
    df_test['predictions'] = predictions

    # Evaluate the classification results
    tp = 1e-8  # True positives
    fp = 1e-8  # False positives
    fn = 1e-8  # False negatives
    tn = 1e-8  # True negatives

    for index, row in df_test_gold.iterrows():
        gold = int(row[label_name])
        guess = int(row['predictions'])
        if guess == 1 and gold == 1:
            tp += 1.
        elif guess == 1 and gold == 0:
            fp += 1.
        elif guess == 0 and gold == 0:
            tn += 1
        elif guess == 0 and gold == 1:
            fn += 1

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if (precision + recall) == 0:
        f_score = 0.0
    else:
        f_score = 2 * (precision * recall) / (precision + recall)

    precision = round(precision, 3) * 100
    recall = round(recall, 3) * 100
    f_score = round(f_score, 3) * 100

    return {"Category": label_name, "Precision": precision, "Recall": recall, "F-score": f_score, "tp": tp, "fp": fp, "tn": tn, "fn": fn}

# Create a list of categories
categories = ['Housing', 'Education', 'Health', 'Religion', 'Public_transportation', 'Selfcare', 'Groceries', 'Finance',
              'Domestic_services', 'Pets', 'Sports', 'Entertainment_and_culture', 'Work']

# Initialize nlp object

# Load initial Russian model
nlp_my = spacy.load('C:/Practice2020/values_nlp/spacy-ru/ru2')

# Load previously made training sample
train_texts = 'data/1_Big_annotated_shuffled.xlsx'
df_training = pd.read_excel(train_texts, sheet_name='Sheet1', header=0, index_col=False, keep_default_na=True)
df_training.round({'Housing': 0, 'Education': 0, 'Health': 0, 'Religion': 0, 'Public_transportation': 0, 'Domestic_services': 0,
                    'Selfcare': 0, 'Groceries': 0, 'Finance': 0, 'Pets': 0, 'Sports': 0, 'Entertainment_and_culture': 0, 'Work': 0})

# Load previously made test sample
test_texts = 'data/2_test_sample_5.xlsx'
df_test = pd.read_excel(test_texts, sheet_name='Sheet1', header=0, index_col=False, keep_default_na=True)
df_test.round({'Housing': 0, 'Education': 0, 'Health': 0, 'Religion': 0, 'Public_transportation': 0, 'Domestic_services': 0,
                    'Selfcare': 0, 'Groceries': 0, 'Finance': 0, 'Pets': 0, 'Sports': 0, 'Entertainment_and_culture': 0, 'Work': 0})

# Formating the df_training for the spacy training data format:
# Example of the format: TRAIN_DATA = [('Text1', {'cats': {'Housing': 0, 'Pedestrian_space': 1}}),(Text2, {'cats': {'Housing': 0, 'Pedestrian_space': 1}})]

training_list = []
for index, row in df_training.iterrows():
    training_text = "('" + row['Text_NORM'] + "',{'cats': {'Housing': " + str(row['Housing']) + \
                        ", 'Education': " + str(row['Education']) + ", 'Health': " + str(row['Health']) + ", 'Religion': " + str(row['Religion']) + \
                        ", 'Public_transportation': " + str(row['Public_transportation']) + ", 'Selfcare': " + str(row['Selfcare']) + \
                        ", 'Groceries': " + str(row['Groceries']) + ", 'Finance': " + str(row['Finance']) + \
                        ", 'Domestic_services': " + str(row['Domestic_services']) + ", 'Pets': " + str(row['Pets']) + \
                        ", 'Sports': " + str(row['Sports']) + ", 'Entertainment_and_culture': " + str(row['Entertainment_and_culture']) + \
                        ", 'Work': " + str(row['Work']) + "}})"

    training_tuple = literal_eval(training_text)
    training_list.append(training_tuple)
df_training['train_format'] = training_list

# 2 Training the classifier
# Based on examples from: www.kaggle.com/poonaml/text-classification-using-spacy
# and from: datascience.stackexchange.com/questions/55896/how-to-train-a-spacy-model-for-text-classification

# Load training dataset
examples_train = df_training['train_format'].tolist()

print('Training data is loaded...\n')

# Initialize a textcat pipe in a spacy pipeline object (nlp), and add the label variable in it
if 'textcat' not in nlp_my.pipe_names:
  textcat = nlp_my.create_pipe("textcat", config={"exclusive_classes": False})
  nlp_my.add_pipe(textcat, last=True)
else:
  textcat = nlp_my.get_pipe("textcat")

for cat in categories:
    textcat.add_label(cat)

# Training the model
other_pipes = [pipe for pipe in nlp_my.pipe_names if pipe != 'textcat']

print('Model training started...\n')

with nlp_my.disable_pipes(*other_pipes):  # Only train the textcat pipe
    optimizer = nlp_my.begin_training()
    batch_sizes = spacy.util.compounding(1.0, 32.0, 1.001)
    for i in range(10):
        # Shuffle the training data
        random.shuffle(examples_train)

        # Create batches and iterate over them
        batches = spacy.util.minibatch(examples_train, size=batch_sizes)
        for batch in batches:
            # Split the batch in texts and annotations
            texts = [text for text, annotation in batch]
            annotations = [annotation for text, annotation in batch]
            # Update the model
            nlp_my.update(texts, annotations, sgd=optimizer, drop=0.2)

# Save the model
nlp_my.to_disk('models/model_13_fin')

print('Model training completed...\n')

print('Model testing...')
print('')

#nlp_tr = spacy.load('C:/Practice2020/values/models/model_13_1')
#test_texts = 'data/annotated_test.xlsx'
#df_test = pd.read_excel(test_texts, sheet_name='Sheet1', header=0, index_col=False, keep_default_na=True)

results = []
for cat in categories:
    tempo_dict = evaluate(df_test, cat)
    results.append(tempo_dict)
    print('Evaluation finished for ' + cat)

df_results = pd.DataFrame(results)
print(df_results)
print('')

# Save the results
#writer_sample = ExcelWriter('data/3_testres_5.xlsx')
#df_results.to_excel(writer_sample, 'Sheet1')
#writer_sample.save()

print('Overall evaluation:')

# Calculate overall quantity of tp, fp, tn and fn
tp_total = df_results['tp'].sum()
fp_total = df_results['fp'].sum()
tn_total = df_results['tn'].sum()
fn_total = df_results['fn'].sum()

precision_total = tp_total / (tp_total + fp_total)
recall_total = tp_total / (tp_total + fn_total)
if (precision_total + recall_total) == 0:
    f_score_total = 0.0
else:
    f_score_total = 2 * (precision_total * recall_total) / (precision_total + recall_total)

precision_total = round(precision_total, 3) * 100
recall_total = round(recall_total, 3) * 100
f_score_total = round(f_score_total, 3) * 100

print('Precision:' + str(precision_total))
print('Recall:' + str(recall_total))
print('F-score:' + str(f_score_total))



