import random
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
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

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

    return lemmas


def count_word_occurrences(lemmas):
    # Порог для минимального количества встречаемости слова в тексте
    word_count_threshold = 0

    # Считаем количество вхождений каждого слова
    word_counts = Counter(lemmas)

    filtered_lemmas = []

    # Удаляем слова, которые встречаются редко
    for i in range(0, len(lemmas)):
        if word_counts[lemmas[i]] > word_count_threshold:
            filtered_lemmas.append(lemmas[i])

    return filtered_lemmas


def dict_text(directory):
    file_data = {}

    for file_name in os.listdir(directory):
        with open(directory + '/' + file_name, 'r', encoding='UTF-8-SIG') as file:
            lines = file.readlines()

            author = lines[0].strip()
            text = ''.join(lines[2:])

            lemmas = process_text(text)

            filtered_lemmas = count_word_occurrences(lemmas)

            if author not in file_data:
                file_data[author] = []

            file_data[author].append(filtered_lemmas)

    return file_data


def split_data(data):
    train_data_split = []
    test_data_split = []
    true_authors_indices = []
    authors_indices = []

    authors_index = {author: index for index, author in enumerate(data)}  # Индексы авторов

    for author in data.keys():
        author_index = authors_index[author]
        texts = [' '.join(text) for text in data[author]]
        test_data_split.append(texts.pop(random.randint(0, len(texts) - 1)))
        true_authors_indices.append(author_index)
        for text in texts:
            train_data_split.append(text)
            authors_indices.append(author_index)

    return train_data_split, test_data_split, true_authors_indices, authors_indices


all_data = dict_text("data/training")
# Обучающие данные, тестовые данные, метки тестовых данных, индексы авторов
training_data, test_data, true_labels, authors = split_data(all_data)

print(true_labels)

tfidf = TfidfVectorizer(min_df=3, max_df=0.80, max_features=5000, analyzer='word')
text_features = tfidf.fit_transform(training_data).toarray()

classifier = MLPClassifier(max_iter=1000, alpha=0.003, hidden_layer_sizes=(50, ),  solver='lbfgs', activation='logistic')
classifier.fit(text_features, authors)

test_text_features = tfidf.transform(test_data).toarray()

predictions = classifier.predict(test_text_features)

print(metrics.classification_report(true_labels, predictions, zero_division=0))