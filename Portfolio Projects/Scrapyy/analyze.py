import os
import re
import nltk
import csv
from nltk.tokenize import word_tokenize, sent_tokenize

# Stopwords
stop_words = set(open(r'path_to_stop_words.txt').read().split())

# Positive and negative words lists
positive_words = set(open(r'path_to_positive_words.txt').read().split())
negative_words = set(open(r'path_to_negative_words.txt').read().split())

# Function to count syllables in a word
def count_syllables(word):
    word = word.lower()
    syllables = 0
    vowels = "aeiouy"
    if word[0] in vowels:
        syllables += 1
    for index in range(1, len(word)):
        if word[index] in vowels and word[index-1] not in vowels:
            syllables += 1
    if word.endswith("e"):
        syllables -= 1
    if word.endswith("le") and len(word) > 2 and word[-3] not in vowels:
        syllables += 1
    if syllables == 0:
        syllables += 1
    return syllables

# Directory of text files
directory = r'C:\Users\abdul\OneDrive - Kagune\Desktop\Scrapyy\newdata'

# CSV file for results
csv_file_path = 'results.csv'
with open(csv_file_path, mode='w', newline='', encoding='utf-8') as csv_file:
    writer = csv.writer(csv_file)
    # Write the header row
    writer.writerow(['Filename', 'POSITIVE SCORE', 'NEGATIVE SCORE', 'POLARITY SCORE', 'SUBJECTIVITY SCORE', 
                     'AVG SENTENCE LENGTH', 'PERCENTAGE OF COMPLEX WORDS', 'FOG INDEX', 'AVG NUMBER OF WORDS PER SENTENCE',
                     'COMPLEX WORD COUNT', 'WORD COUNT', 'SYLLABLE PER WORD', 'PERSONAL PRONOUNS', 'AVG WORD LENGTH'])

    # Process each file
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read().lower()
                
                # Tokenization and Cleaning
                words = word_tokenize(text)
                sentences = sent_tokenize(text)
                cleaned_words = [word for word in words if word not in stop_words and word.isalpha()]
                
                # Counts
                positive_count = sum(word in positive_words for word in cleaned_words)
                negative_count = sum(word in negative_words for word in cleaned_words)
                complex_words_count = sum(count_syllables(word) > 2 for word in cleaned_words)
                total_syllables = sum(count_syllables(word) for word in cleaned_words)
                word_count = len(cleaned_words)
                sentence_count = len(sentences)
                personal_pronouns_count = len(re.findall(r'\b(i|we|my|ours|us)\b', text, re.IGNORECASE))
                total_characters = sum(len(word) for word in cleaned_words)
                
                # Calculations
                polarity_score = (positive_count - negative_count) / ((positive_count + negative_count) + 0.000001)
                subjectivity_score = (positive_count + negative_count) / (word_count + 0.000001)
                avg_sentence_length = word_count / sentence_count
                percent_complex_words = (complex_words_count / word_count) * 100
                fog_index = 0.4 * (avg_sentence_length + percent_complex_words)
                avg_words_per_sentence = word_count / sentence_count
                avg_word_length = total_characters / word_count
                syllables_per_word = total_syllables / word_count
                
                # Write data row for each file
                writer.writerow([filename, positive_count, negative_count, polarity_score, subjectivity_score, 
                                 avg_sentence_length, percent_complex_words, fog_index, avg_words_per_sentence,
                                 complex_words_count, word_count, syllables_per_word, personal_pronouns_count, avg_word_length])