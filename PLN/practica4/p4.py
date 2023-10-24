from sklearn.datasets import fetch_20newsgroups
import spacy
#Funcion para quitar stopwords
def stopwords (doc):
    stop_words = [
        "el", "la", "los", "las", "un", "una", "unos", "unas", "al", "del", "lo", "este", "ese", "aquel", "estos", "esos", "aquellos", "este", "esta", "estas", "eso", "esa", "esas", "aquello", "alguno", "alguna", "algunos", "algunas",
        "a", "ante", "bajo", "cabe", "con", "contra", "de", "desde", "en", "entre", "hacia", "hasta", "para", "por", "según", "sin", "so", "sobre", "tras", "durante", "mediante", "excepto", "a través de", "conforme a", "encima de", "debajo de", "frente a", "dentro de",
        "y", "o", "pero", "ni", "que", "si", "como", "porque", "aunque", "mientras", "siempre que", "ya que", "pues", "a pesar de que", "además", "sin embargo", "así que", "por lo tanto", "por lo que", "tan pronto como", "a medida que", "tanto como", "no solo... sino también", "o bien", "bien... bien",
        "yo", "tú", "él", "ella", "nosotros", "vosotros", "ellos", "ellas", "usted", "nosotras", "me", "te", "le", "nos", "os", "les", "se", "mí", "ti", "sí", "conmigo", "contigo", "consigo", "mi", "tu", "su", "nuestro", "vuestro", "sus", "mío", "tuyo", "suyo", "nuestro", "vuestro", "suyo"]
    texto_sin_stopwords = [token for token in doc if token.text.lower() not in stop_words]
    #union de tokens
    texto_sin_stopwords = ' '.join(texto_sin_stopwords)
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

nlp = spacy.load("es_core_news_sm")

#Carga y separación del corpus
newsgroups_train = fetch_20newsgroups(subset='train')
print (newsgroups_train.filenames.shape)
newsgroups_test = fetch_20newsgroups(subset='test')

#Preprocesamiento del corpus
#Quitar stopwords
newsgroups_train.data = [stopwords(doc) for doc in newsgroups_train.data]
newsgroups_test.data = [stopwords(doc) for doc in newsgroups_test.data]

#guardar corpus en un archivo

with open('newsgroups_trainDataSinStopwords.txt', 'w', encoding='utf-8') as f:
    for item in newsgroups_train.data:
        f.write("%s\n&&&&&&&&\n" % item)
with open('newsgroups_testDataSinStopwords.txt', 'w', encoding='utf-8') as f:
    for item in newsgroups_test.data:
        f.write("%s\n&&&&&&&&\n" % item)


#X_train = newsgroups_train.data
y_train = newsgroups_train.target
#X_test = newsgroups_test.data
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
#representación del texto frecuencia

vectorizer = CountVectorizer(binary=True)
vectors_train = vectorizer.fit_transform(newsgroups_train.data)
vectors_test = vectorizer.transform(newsgroups_test.data)

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