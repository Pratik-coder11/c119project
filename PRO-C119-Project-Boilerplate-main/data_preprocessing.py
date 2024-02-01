# Text Data Preprocessing Lib
import nltk
nltk.download('punkt')
nltk.download('wordnet')

# to stem words
from nltk.stem import PorterStemmer

# create an instance of class PorterStemmer
stemmer = PorterStemmer()

# importing json lib
import json
import pickle
import numpy as np

words=[] #list of unique roots words in the data
classes = [] #list of unique tags in the data
pattern_word_tags_list = [] #list of the pair of (['words', 'of', 'the', 'sentence'], 'tags')

# words to be ignored while creating Dataset
ignore_words = ['?', '!',',','.', "'s", "'m"]

# open the JSON file, load data from it.
train_data_file = open('intents.json')
data = json.load(train_data_file)
train_data_file.close()

# creating function to stem words
def get_stem_words(words, ignore_words):
    stem_words = []
    for word in words:
        if word.lower() not in ignore_words:
            stemmed_word = stemmer.stem(word.lower())
            stem_words.append(stemmed_word)
    return stem_words

# creating a function to make corpus
def create_bot_corpus(words, classes, pattern_word_tags_list, ignore_words):

    for intent in data['intents']:

        # Add all patterns and tags to a list
        for pattern in intent['patterns']:  

            # tokenize the pattern          
            pattern_words = nltk.word_tokenize(pattern)

            # add the tokenized words to the words list        
            words.extend(pattern_words)

            # add the 'tokenized word list' along with the 'tag' to pattern_word_tags_list
            pattern_word_tags_list.append((pattern_words, intent['tag']))

        # Add all tags to the classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

    stem_words = get_stem_words(words, ignore_words) 

    # Remove duplicate words from stem_words
    stem_words = list(set(stem_words))

    # sort the stem_words list and classes list
    stem_words.sort()
    classes.sort()

    # print stem_words
    print('stem_words list : ' , stem_words)

    return stem_words, classes, pattern_word_tags_list

# Training Dataset: 
# Input Text----> as Bag of Words 
# Tags-----------> as Label

def bag_of_words_encoding(stem_words, pattern_word_tags_list):
    
    bag = []
    for word_tags in pattern_word_tags_list:
        # example: word_tags = (['hi', 'there'], 'greetings']

        pattern_words = word_tags[0] # ['Hi' , 'There']
        bag_of_words = []

        # stemming pattern words before creating Bag of words
        stemmed_pattern_word = get_stem_words(pattern_words, ignore_words)

        # Input data encoding 
        for w in stem_words:
            bag_of_words.append(1) if w in stemmed_pattern_word else bag_of_words.append(0)

        bag.append(bag_of_words)
    
    return np.array(bag)

def class_label_encoding(classes, pattern_word_tags_list):
    
    labels = []

    for word_tags in pattern_word_tags_list:

        # Start with list of 0s 
        labels_encoding = list([0]*len(classes))  

        # example: word_tags = (['hi', 'there'], 'greetings')

        tag = word_tags[1]   # 'greetings'

        tag_index = classes.index(tag)

        # Labels Encoding
        labels_encoding[tag_index] = 1

        labels.append(labels_encoding)
        
    return np.array(labels)

def preprocess_train_data():
  
    stem_words, tag_classes, word_tags_list = create_bot_corpus(words, classes, pattern_word_tags_list, ignore_words)
    
    # Convert Stem words and Classes to Python pickel file format
    with open('stem_words.pkl', 'wb') as f:
        pickle.dump(stem_words, f)
    with open('classes.pkl', 'wb') as f:
        pickle.dump(classes, f)

    train_x = bag_of_words_encoding(stem_words, word_tags_list)
    train_y = class_label_encoding(tag_classes, word_tags_list)
    
    return train_x, train_y

bow_data  , label_data = preprocess_train_data()

# after completing the code, remove comment from print statements
print("first BOW encoding: " , bow_data[0])
print("first Label encoding: " , label_data[0])
