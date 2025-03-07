import datetime
import os

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS, cross_origin

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize

from pprint import pprint


nltk.download('stopwords')
nltk.download('punkt')


app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/search_api", methods=["GET"])
@cross_origin()
def search_api():
    text_str = request.form['text']
    result = run_summarization(text_str)
        
    return result

@app.route("/search", methods=["POST"])
@cross_origin()
def search():
    text_str = request.form['text']
    result = run_summarization(text_str)
    return result



def _create_frequency_table(text_string) -> dict:
    """
    we create a dictionary for the word frequency table.
    For this, we should only use the words that are not part of the stopWords array.
    Removing stop words and making frequency table
    Stemmer - an algorithm to bring words to its root word.
    :rtype: dict
    """
    stopWords = set(stopwords.words("spanish"))
    words = word_tokenize(text_string)
    ps = PorterStemmer()

    freqTable = dict()
    for word in words:
        word = ps.stem(word)
        if word in stopWords:
            continue
        if word in freqTable:
            freqTable[word] += 1
        else:
            freqTable[word] = 1

    return freqTable


def _score_sentences(sentences, freqTable) -> dict:
    """
    score a sentence by its words
    Basic algorithm: adding the frequency of every non-stop word in a sentence divided by total no of words in a sentence.
    :rtype: dict
    """

    sentenceValue = dict()

    for sentence in sentences:
        word_count_in_sentence = (len(word_tokenize(sentence)))
        word_count_in_sentence_except_stop_words = 0
        for wordValue in freqTable:
            if wordValue in sentence.lower():
                word_count_in_sentence_except_stop_words += 1
                if sentence[:20] in sentenceValue:
                    sentenceValue[sentence[:20]] += freqTable[wordValue]
                else:
                    sentenceValue[sentence[:20]] = freqTable[wordValue]

        if sentence[:20] in sentenceValue:
            sentenceValue[sentence[:20]] = sentenceValue[sentence[:20]] / word_count_in_sentence_except_stop_words

        '''
        Notice that a potential issue with our score algorithm is that long sentences will have an advantage over short sentences. 
        To solve this, we're dividing every sentence score by the number of words in the sentence.
        
        Note that here sentence[:10] is the first 10 character of any sentence, this is to save memory while saving keys of
        the dictionary.
        '''

    return sentenceValue

def _find_average_score(sentenceValue) -> int:
    """
    Find the average score from the sentence value dictionary
    :rtype: int
    """
    sumValues = 0
    for entry in sentenceValue:
        sumValues += sentenceValue[entry]

    # Average value of a sentence from original text
    average = (sumValues / len(sentenceValue))

    return average

def _generate_summary(sentences, sentenceValue, threshold):
    sentence_count = 0
    summary = ''

    for sentence in sentences:
        if sentence[:20] in sentenceValue and sentenceValue[sentence[:20]] >= (threshold):
            summary += " " + sentence
            sentence_count += 1

    return summary



def run_summarization(text):
    # 1 Create the word frequency table
    freq_table = _create_frequency_table(text)

    print(freq_table)
    print('---------------------------------')

    '''
    We already have a sentence tokenizer, so we just need 
    to run the sent_tokenize() method to create the array of sentences.
    '''

    # 2 Tokenize the sentences
    sentences = sent_tokenize(text)

    pprint(sentences)
    print('---------------------------------')

    # 3 Important Algorithm: score the sentences
    sentence_scores = _score_sentences(sentences, freq_table)

    pprint(sentence_scores)
    print('---------------------------------')

    # 4 Find the threshold
    threshold = _find_average_score(sentence_scores)

    pprint(threshold)
    print('---------------------------------')


    # 5 Important Algorithm: Generate the summary
    summary = _generate_summary(sentences, sentence_scores, threshold)

    pprint(summary)
    print('---------------------------------')


    return summary