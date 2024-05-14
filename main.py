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
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression

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


def count_word_occurrences(lemmas):

    # Удаляет леммы, но не удаляет части речи и уровень зависимости
    # Удаляет леммы, но не удаляет части речи и уровень зависимости
    # Удаляет леммы, но не удаляет части речи и уровень зависимости
    # Удаляет леммы, но не удаляет части речи и уровень зависимости

    # Порог для минимального количества встречаемости слова в тексте
    word_count_threshold = 0

    # Считаем количество вхождений каждого слова
    word_counts = Counter(lemmas)

    # Удаляем слова, которые встречаются редко
    filtered_lemmas = [word for word in lemmas if word_counts[word] >= word_count_threshold]

    return filtered_lemmas


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

            # filtered_lemmas = count_word_occurrences(lemmas)

            filtered_lemmas = lemmas

            file_token_pos += [' '.join(text for text in token_pos)]
            file_token_dep += [' '.join(text for text in token_dep)]

            if author not in file_data:
                file_data[author] = []

            file_data[author].append(filtered_lemmas)

    return file_data, file_token_pos, file_token_dep


training_data, training_token_pos, training_token_dep = dict_text("data/training")
authors = [index for index, author in enumerate(training_data) for _ in training_data[author]]  # Разделение текстов по авторам
authors_index = {author: index for index, author in enumerate(training_data)}  # Индексы авторов
final_training_data = [' '.join(text) for author_texts in training_data.values() for text in author_texts]  # Тексты

print(authors_index)

test_data, test_token_pos, test_token_dep = dict_text('data/testData')
final_test_data = [' '.join(text) for author_texts in test_data.values() for text in author_texts]

tfidf = TfidfVectorizer()
text_features = tfidf.fit_transform(final_training_data).toarray()
training_pos_features = tfidf.transform(training_token_pos).toarray()
training_dep_features = tfidf.transform(training_token_dep).toarray()

training_all_features = np.concatenate((text_features, training_pos_features, training_dep_features), axis=1)
classifier = LogisticRegression()
classifier.fit(training_all_features, authors)

test_text_features = tfidf.transform(final_test_data).toarray()
test_pos_features = tfidf.transform(test_token_pos).toarray()
test_dep_features = tfidf.transform(test_token_dep).toarray()

test_all_features = np.concatenate((test_text_features, test_pos_features, test_dep_features), axis=1)
new_count_vectorizer = tfidf.transform(final_test_data)

predictions = classifier.predict(test_all_features)

print(predictions)
