import json
from nltk.stem import PorterStemmer
import tensorflow as tf 

from tensorflow.keras.models import Sequential
import numpy as np

import random

#open json.file, read it READEADAW


def tokenize(sentence):
    split = sentence.split()
    #punctuation remove
    for i in range(len(split)):    
        #?!.,'
        if "?" in split[i] or "!" in split[i] or "." in split[i] or "," in split[i]:
            exclude_punct = split[i][:-1]
            split[i] = exclude_punct
    return split

with open("C:\\Users\\Chris\\Desktop\\intents.json") as f:
    intents_data = json.load(f)

words = []
tags = []
documentx = []
documenty = []

for intent in intents_data['intents']:
    tag = intent['tag']
    patterns = intent['patterns']
    for pattern in patterns:
        #WORDS
        #tokenize
        tokens = tokenize(pattern)
        #stemming
        stemmer = PorterStemmer()
        stemmed_tokens = [stemmer.stem(token.lower()) for token in tokens]
        #appending
        for token in stemmed_tokens:
            words.append(token)

        #TAGS
        tags.append(tag)

        #DOCUMENTS
        #documentx = [[hello,how,are,you], [need,assistant, with,this,problem],[wee,daw,jjj]]
        #words = [hello,how,are,you,need,assistant,with,this,problem,wee,daw,jjj]
        #documenty = [greeting, help]
        documentx.append(stemmed_tokens)
        documenty.append(tag)
words = list(set(words))   
tags = list(set(tags))

words = sorted(words)
tags = sorted(tags)

#bagofwords
x = []
y = []

#[stemmer.stem(w) for w in pattern]
def bagofwords(doc1):
    ex = [0]*len(words)
    lst = [stemmer.stem(w) for w in doc1]
    for i, w in enumerate(words):
        if w in lst:
            ex[i] = 1
    return ex

for doc1 in documentx:
    docx = bagofwords(doc1)
    x.append(docx)

#location = listset_tags.index(tag)
def bagofwords2(tg):
    ey = [0]*len(tags)
    loc = tags.index(tg)
    ey[loc] = 1
    return ey

for tg in documenty:
    docy = bagofwords2(tg)
    y.append(docy)


#training model

#tf.keras.layers.Dense * 3
#tf.keras.Sequential

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(len(x[0]))),
    tf.keras.layers.Dense(12, activation='relu'),
    tf.keras.layers.Dense(12, activation='relu'),
    tf.keras.layers.Dense(len(y[0]), activation='softmax')
])

#training
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(np.array(x), np.array(y), batch_size=8, epochs=300, verbose=1)
#batch_size, epochs, verbose, callbacks, validation_split, validation_data, shuffle, class_weight, etcetcetc

#1.user input
#2. model.predict()
#3. y prediction -> randomly chosen response

while True:
    user_input = input("[Type exit to quit Chat Bot]\nType your sentence: ")
    if user_input == "exit":
        break
    print("")
    #print("Your Input = " + user_input)
    
    #tokenize & stem
    tokenized_input = tokenize(user_input)
    stemmed_input = [stemmer.stem(yes) for yes in tokenized_input]

    #print("stemmed_input = ", stemmed_input)

    listofinput = []
    for i in stemmed_input:
        listofinput.append(i)

    #print("listofinput = ", listofinput)

    #bag of words
    def bow(listofinput):
        lst = [0] * len(words)
        for i, wor in enumerate(words):
            if wor in listofinput:
                lst[i] = 1
        return lst
    
    #print(words)
    new_listofinput = bow(listofinput)

    newreshape = np.reshape(np.array(new_listofinput),(1, len(new_listofinput)))

    #print("newlistofinput = ", new_listofinput)

    #model prediction
    response = model.predict(np.array(newreshape))
    #print(response)

    max_number_index = np.argmax(response)

    #print(max_number_index)

    tag = tags[max_number_index]
    for res in intents_data['intents']:
        if res['tag'] == tag:
            final_response = random.choice(res['responses'])
            print("Chat Bot: ", final_response)


 




#use while loop v
#user input --> tokenize, stemm
#bagofwords use
#model prediction
#select response
#output response