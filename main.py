import string
import os
import time

import axis as axis
import numpy as np
import spacy
import numpy

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
import nltk
import pymorphy2
from sklearn import metrics
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

stop_words = set(stopwords.words('russian'))
punctuation = set(string.punctuation)
morph = pymorphy2.MorphAnalyzer()

nlp = spacy.load("ru_core_news_sm")


def process_text(text):
    # Токенизация
    tokens = word_tokenize(text, language='russian')

    # Приводим слова к нижнему регистру, удаляем стоп-слова и знаки пунктуации
    filtered_tokens = [word.lower() for word in tokens
                       if word.isalpha() and word.lower() not in stop_words and word not in punctuation]

    # Приводим слова к начальной форме
    lemmas = [morph.parse(token)[0].normal_form for token in filtered_tokens]

    # Обрабатываем текст с помощью nlp модели
    doc = nlp(' '.join(lemmas))

    # Получаем список слов, частей речи и дерево зависимостей для каждого токена
    processed_text, token_pos, token_dep = ([token.text for token in doc],
                                            [token.pos_ for token in doc],
                                            [token.dep_ for token in doc])

    return processed_text, token_pos, token_dep


def count_word_occurrences(lemmas, token_pos, token_dep):
    # Порог для минимального количества встречаемости слова в тексте
    word_count_threshold = 0

    # Считаем количество вхождений каждого слова
    word_counts = Counter(lemmas)

    filtered_lemmas = []
    filtered_token_pos = []
    filtered_token_dep = []

    # Удаляем слова, которые встречаются редко
    for i in range(0, len(lemmas)):
        if word_counts[lemmas[i]] > word_count_threshold:
            filtered_lemmas.append(lemmas[i])
            filtered_token_pos.append(token_pos[i])
            filtered_token_dep.append(token_dep[i])

    return filtered_lemmas, filtered_token_pos, filtered_token_dep


def dict_text(directory):
    file_data = {}
    file_token_pos = []
    file_token_dep = []

    for file_name in os.listdir(directory):
        with open(directory + '/' + file_name, 'r', encoding='UTF-8-SIG') as file:
            lines = file.readlines()

            author = lines[0].strip()
            text = ''.join(lines[2:])

            lemmas, token_pos, token_dep = process_text(text)

            filtered_lemmas, filtered_token_pos, filtered_token_dep = count_word_occurrences(lemmas, token_pos, token_dep)

            file_token_pos += [' '.join(text for text in filtered_token_pos)]
            file_token_dep += [' '.join(text for text in filtered_token_dep)]

            if author not in file_data:
                file_data[author] = []

            file_data[author].append(filtered_lemmas)

    return file_data, file_token_pos, file_token_dep


training_data, training_token_pos, training_token_dep = dict_text("data/training")
authors = [index for index, author in enumerate(training_data) for _ in training_data[author]]  # Разделение текстов по авторам
authors_index = {author: index for index, author in enumerate(training_data)}  # Индексы авторов
final_training_data = [' '.join(text) for author_texts in training_data.values() for text in author_texts]  # Тексты


test_data, test_token_pos, test_token_dep = dict_text('data/testData')
final_test_data = [' '.join(text) for author_texts in test_data.values() for text in author_texts]

tfidf = TfidfVectorizer(min_df=3, max_df=0.80, max_features=5000, analyzer='word')
text_features = tfidf.fit_transform(final_training_data).toarray()

all_pos_tags = list(set([tag for tags_row in training_token_pos + test_token_pos for tag in tags_row.split()]))
all_dep_tags = list(set([tag for tags_row in training_token_dep + test_token_dep for tag in tags_row.split()]))

pos_tags_indices = []
dep_tags_indices = []

for pos_tags_row, dep_tags_row in zip(training_token_pos, training_token_dep):
    pos_tags_indexed = [all_pos_tags.index(tag) for tag in pos_tags_row.split(" ")]
    dep_tags_indexed = [all_dep_tags.index(tag) for tag in dep_tags_row.split(" ")]

    pos_tags_indices.append(pos_tags_indexed)
    dep_tags_indices.append(dep_tags_indexed)

max_length = max(len(row) for row in pos_tags_indices)
fill_value = -1
pos_tags_indices_padded = [row + [fill_value]*(max_length-len(row)) for row in pos_tags_indices]
dep_tags_indices_padded = [row + [fill_value]*(max_length-len(row)) for row in dep_tags_indices]

encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
encoder.fit_transform(np.array(pos_tags_indices_padded + dep_tags_indices_padded))

pos_tags_encoded = encoder.transform(np.array(pos_tags_indices_padded))
dep_tags_encoded = encoder.transform(np.array(dep_tags_indices_padded))

training_all_features = np.concatenate((text_features, pos_tags_encoded, dep_tags_encoded), axis=1)

classifier = MLPClassifier(max_iter=1000, alpha=0.003, hidden_layer_sizes=(50, ),  solver='adam')
classifier.fit(training_all_features, authors)


test_text_features = tfidf.transform(final_test_data).toarray()

test_pos_tags_indices = []
test_dep_tags_indices = []

for test_pos_tags_row, test_dep_tags_row in zip(test_token_pos, test_token_dep):
    test_pos_tags_indexed = [all_pos_tags.index(tag) for tag in test_pos_tags_row.split(" ")]
    test_dep_tags_indexed = [all_dep_tags.index(tag) for tag in test_dep_tags_row.split(" ")]

    test_pos_tags_indices.append(test_pos_tags_indexed)
    test_dep_tags_indices.append(test_dep_tags_indexed)

max_length_test = max(len(row) for row in test_pos_tags_indices)
test_pos_tags_indices_padded = [row + [fill_value]*(max_length_test-len(row)) for row in test_pos_tags_indices]
test_dep_tags_indices_padded = [row + [fill_value]*(max_length_test-len(row)) for row in test_dep_tags_indices]

test_pos_tags_indices_padded = np.array(test_pos_tags_indices_padded)
test_dep_tags_indices_padded = np.array(test_dep_tags_indices_padded)

test_pos_tags_encoded = encoder.transform(test_pos_tags_indices_padded[:, :max_length])
test_dep_tags_encoded = encoder.transform(test_dep_tags_indices_padded[:, :max_length])

test_all_features = np.concatenate((test_text_features, test_pos_tags_encoded, test_dep_tags_encoded), axis=1)

predictions = classifier.predict(test_all_features)

print(predictions)

true_labels = [3, 10, 0, 12, 5, 11, 7, 9, 6, 1, 4, 8, 2]

print(metrics.classification_report(true_labels, predictions, zero_division=0))
