from sklearn.datasets import fetch_20newsgroups
import spacy
#Funcion para quitar stopwords
def stopwords (doc):
    #doc = nlp(doc)
    stop_words = [
    "the", "a", "an", "some", "to", "of", "the", "it", "this", "that", "those", "these", "these", "those", "this", "this", "these", "that", "that", "those", "that", "some", "a", "an", "some", "some", "a", "before", "under", "about", "with", "against", "from", "since", "in", "between", "toward", "until", "for", "by", "according to", "without", "so", "on", "after", "during", "by means of", "except", "through", "according to", "above", "below", "in front of", "inside", 
    "and", "or", "but", "nor", "that", "if", "as", "because", "although", "while", "whenever", "since", "because", "even though", "furthermore", "however", "so", "therefore", "so", "as", "as soon as", "as", "both", "either", "neither", "not only... but also", "or", "either... or",
    "I", "you", "he", "she", "we", "you", "they", "they", "you", "we", "me", "you", "him", "us", "you", "them", "themselves", "myself", "yourself", "himself", "mine", "yours", "his", "ours", "yours", "theirs", "mine", "yours", "his", "ours", "yours", "theirs" , "i"
    ]
    texto_sin_stopwords = [token for token in doc if token.text.lower() not in stop_words]
    #union de tokens
    texto_sin_stopwords = ' '.join(token.text for token in texto_sin_stopwords)

    return texto_sin_stopwords

#Funcion para lematizar
def lematizar (doc):
    lemmatized_tokens = []
    doc = nlp(doc)
    for token in doc: 
        lemmatized_tokens.append(token.lemma_)
    #union de tokens
    lemmatized_tokens = ' '.join(lemmatized_tokens)
    return lemmatized_tokens

nlp = spacy.load("en_core_web_sm")

#Carga y separación del corpus
newsgroups_train = fetch_20newsgroups(subset='train')
print (newsgroups_train.filenames.shape)
newsgroups_test = fetch_20newsgroups(subset='test')

#Preprocesamiento del corpus
#Quitar stopwords
newsgroups_trainDataSinSW = [stopwords(nlp(doc)) for doc in newsgroups_train.data]
newsgroups_testDataSinSW = [stopwords(nlp(doc)) for doc in newsgroups_test.data]

#guardar corpus en un archivo

with open('newsgroups_trainDataSinStopwords_en.txt', 'w', encoding='utf-8') as f:
    for item in newsgroups_trainDataSinSW:
        f.write("%s\n&&&&&&&&\n" % item)
with open('newsgroups_testDataSinStopwords_en.txt', 'w', encoding='utf-8') as f:
    for item in newsgroups_testDataSinSW:
        f.write("%s\n&&&&&&&&\n" % item)


#X_train = newsgroups_train.data
#X_test = newsgroups_test.data

X_train = newsgroups_trainDataSinSW
X_test = newsgroups_testDataSinSW

y_train = newsgroups_train.target
y_test = newsgroups_test.target

#guardar corpus en un archivo
"""
with open('newsgroups_trainData.txt', 'w', encoding='utf-8') as f:
    for item in newsgroups_train.data:
        f.write("%s\n" % item)

with open('newsgroups_testData.txt', 'w', encoding='utf-8') as f:
    for item in newsgroups_test.data:
        f.write("%s\n" % item)

with open('newsgroups_trainTarget.txt', 'w', encoding='utf-8') as f:
    for item in newsgroups_train.target:
        f.write("%s\n" % item)

with open('newsgroups_testTarget.txt', 'w', encoding='utf-8') as f:
    for item in newsgroups_test.target:
        f.write("%s\n" % item)

"""


#representación del texto TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
#vectorizer = TfidfVectorizer()
#vectorizer = CountVectorizer()
#vectorizer = CountVectorizer(binary=True)

#representación del texto frecuencia

vectorizer = TfidfVectorizer()
vectors_train = vectorizer.fit_transform(X_train)
vectors_test = vectorizer.transform(X_test)

#Guarda los vectores en un archivo

"""
with open('vectors_trainTFIDF.txt', 'w', encoding='utf-8') as f:
    for item in vectors_train:
        f.write("%s\n" % item)

with open('vectors_testTFIDF.txt', 'w', encoding='utf-8') as f:
    for item in vectors_test:
        f.write("%s\n" % item)
"""


#Clasificadores
from sklearn.metrics import classification_report

#logistic regression
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
clf.fit(vectors_train, y_train)
y_pred = clf.predict(vectors_test)
print ('*****************Logistic Regression****************')
print (classification_report(y_test, y_pred))



#multinomial naive bayes
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(vectors_train, y_train)
y_pred = clf.predict(vectors_test)
print ('*****************MultinomialNB****************')
print (classification_report(y_test, y_pred))


#kneighbors classifier
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier()
clf.fit(vectors_train, y_train)
y_pred = clf.predict(vectors_test)
print ('*****************KNeighborsClassifier****************')
print (classification_report(y_test, y_pred))


#random forest classifier
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(vectors_train, y_train)
y_pred = clf.predict(vectors_test)
print ('*****************RandomForestClassifier****************')
print (classification_report(y_test, y_pred))

"""#multilayer perceptron
from sklearn.neural_network import MLPClassifier
clf = MLPClassifier()
clf.fit(vectors_train, y_train)
y_pred = clf.predict(vectors_test)
print ('*****************MLPClassifier****************')
print (classification_report(y_test, y_pred))

"""