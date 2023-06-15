import string

import numpy as np
import pandas as pd
import nltk
import re

from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
from textblob import TextBlob

nltk.download('stopwords')

stopwords = set(stopwords.words('english'))

# https://data.world/alexfilatov/2016-usa-presidential-election-tweets


data = pd.read_csv('tweetsUSElection.csv')


# Process text
def formatText(text):
    if pd.isnull(text):
        return ""
    if isinstance(text, str):
        text = "".join([char for char in text if char not in string.punctuation])
        text = re.sub(r'http\S+', '', text)
        tokens = re.split('\W+', text)
        text = " ".join([word for word in tokens if word.lower() not in stopwords])
    else:
        text = str(text) 
    return text


data['clean_text'] = data['tweet_text'].apply(formatText)


# Wykorzystaj bibliotekę TextBlob do analizy nastrojów każdego tweetu
def get_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity


data['sentiment'] = data['clean_text'].apply(get_sentiment)

# Przeanalizuj i podsumuj wyniki
positive_tweets = data[data['sentiment'] > 0]
negative_tweets = data[data['sentiment'] < 0]
neutral_tweets = data[data['sentiment'] == 0]

positive_percentage = len(positive_tweets) / len(data) * 100
negative_percentage = len(negative_tweets) / len(data) * 100
neutral_percentage = len(neutral_tweets) / len(data) * 100

print(f"Positive tweets: {positive_percentage}%")
print(f"Negative tweets: {negative_percentage}%")
print(f"Neutral tweets: {neutral_percentage}%")

wordcloud = WordCloud(width=800, height=400).generate(' '.join(data['clean_text']))

# Wyświetlanie chmury słów
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

positive_tweets = len(data[data['polarity'] > 0])
negative_tweets = len(data[data['polarity'] < 0])
neutral_tweets = len(data[data['polarity'] == 0])

# Tworzenie wykresu słupkowego
labels = ['Pozytywne', 'Negatywne', 'Neutralne']
values = [positive_tweets, negative_tweets, neutral_tweets]

plt.figure(figsize=(10, 5))
plt.bar(labels, values)
plt.xlabel('Nastroje')
plt.ylabel('Liczba tweetów')
plt.title('Rozkład nastrojów')
plt.show()

positive_tweets = len(data[data['polarity'] > 0])
negative_tweets = len(data[data['polarity'] < 0])
neutral_tweets = len(data[data['polarity'] == 0])

# Tworzenie wykresu liniowego na podstawie wyników nastrojów w czasie
plt.figure(figsize=(10, 5))
plt.plot(data['created_at'], data['polarity'])
plt.xlabel('Czas')
plt.ylabel('Nastroj')
plt.title('Zmiany nastrojów w czasie')
plt.xticks(rotation=45)
plt.show()

# Tworzenie listy słów
word_list = [word for tweet in data['clean_text'] for word in tweet]

# Obliczanie wystąpień słów
word_count = Counter(word_list)

# # Wyodrębnienie 50 najczęściej występujących słów
# top_words = word_count.most_common(50)
# top_words = dict(top_words)
#
# # Tworzenie wykresu słupkowego
# plt.figure(figsize=(12, 6))
# plt.bar(top_words.keys(), top_words.values())
# plt.xticks(rotation=90)
# plt.xlabel('Słowo')
# plt.ylabel('Liczba wystąpień')
# plt.title('50 najczęściej występujących słów')
# plt.tight_layout()
# plt.show()

# Wykres zawierający liczbę słów
top_words = word_count.most_common(50)
words, counts = zip(*top_words)

plt.figure(figsize=(12, 6))  # Zwiększenie rozmiaru wykresu
plt.bar(words, counts)
plt.xlabel('Słowa')
plt.ylabel('Liczba wystąpień')
plt.title('Najczęściej występujące słowa')
plt.xticks(rotation=45)  # Obrót etykiet osi x o 45 stopni
plt.tight_layout()  # Dopasowanie wykresu
plt.show()

# Wykres zawierający liczbę słów
top_words = word_count.most_common(50)
words, counts = zip(*top_words)

plt.figure(figsize=(12, 6))  # Zwiększenie rozmiaru wykresu
plt.bar(words, counts)
plt.xlabel('Słowa')
plt.ylabel('Liczba wystąpień')
plt.title('Najczęściej występujące słowa')
plt.xticks(rotation=45)  # Obrót etykiet osi x o 45 stopni
plt.tight_layout()  # Dopasowanie wykresu
plt.show()

sorted_words = sorted(top_words, key=lambda x: x[1], reverse=True)

print("Posortowane najczęściej występujące słowa:")
for word, count in sorted_words:
    print(f"Słowo: {word}, Liczba wystąpień: {count}")
