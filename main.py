import string
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import pymorphy2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


stop_words = set(stopwords.words('russian'))
punctuation = set(string.punctuation)


def list_text(directory):
    file_data = []
    for root, directories, filenames in os.walk(directory):
        for filename in filenames:
            file_path = os.path.join(root, filename)
            with open(file_path, 'r', encoding='utf-8') as file:  # Считываем даннные из файла
                text = file.read()

                # Токенизация
                tokens = word_tokenize(text, language='russian')
                # Приводим слова к нижнему регистру, удаляем стоп-слова и знаки пунктуации
                filtered_tokens = [word.lower() for word in tokens if
                                   word.isalpha() and word.lower() not in stop_words and word not in punctuation]

                morph = pymorphy2.MorphAnalyzer()
                # Приводим слова к начальной форме
                lemmas = [morph.parse(token)[0].normal_form for token in filtered_tokens]

                file_data.append(lemmas)

    return file_data


training_data = list_text('data/training/Pushkin')
final_training_data = [' '.join(text) for text in training_data]

test_data = list_text('data/test/Pushkin')
final_test_data = [' '.join(text) for text in test_data]

authors = [3, 2, 3, 1, 1, 3, 2, 3]

tfidf = TfidfVectorizer()
X = tfidf.fit_transform(final_training_data)

classifier = LogisticRegression()
classifier.fit(X, authors)

new_X = tfidf.transform(final_test_data)
predictions = classifier.predict(new_X)

print(predictions)
