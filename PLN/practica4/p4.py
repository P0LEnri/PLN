from sklearn.datasets import fetch_20newsgroups
import spacy
import re
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

def cleanText (doc):
    doc = re.sub(r'\S+@\S+', '', doc)
    # Eliminar líneas con información no deseada
    doc = re.sub(r'\b(subject|Organization|Distribution|NNTP - Posting - host|X - newsreader|line)\s*:[^\n]*', '', doc)

    # Eliminar líneas con caracteres repetidos
    doc = re.sub(r'-{8,}', '', doc)

    # Eliminar números de línea
    doc = re.sub(r'line\s*:\s*\d+', '', doc)

    # Eliminar respuestas o citas de otros correos electrónicos
    doc = re.sub(r'>[^\n]*', '', doc)

    doc = re.sub(r'[^\w\s]', '', doc)

    return doc

nlp = spacy.load("en_core_web_sm")

#Carga y separación del corpus
newsgroups_train = fetch_20newsgroups(subset='train')
print (newsgroups_train.filenames.shape)
newsgroups_test = fetch_20newsgroups(subset='test')

#Preprocesamiento del corpus
#Quitar stopwords
"""newsgroups_trainDataSinSW = [stopwords(nlp(doc)) for doc in newsgroups_train.data]
newsgroups_testDataSinSW = [stopwords(nlp(doc)) for doc in newsgroups_test.data]
"""
#guardar corpus en un archivo

"""with open('newsgroups_trainDataSinStopwords_en.txt', 'w', encoding='utf-8') as f:
    for item in newsgroups_trainDataSinSW:
        f.write("%s\n&&&&&&&&\n" % item)
with open('newsgroups_testDataSinStopwords_en.txt', 'w', encoding='utf-8') as f:
    for item in newsgroups_testDataSinSW:
        f.write("%s\n&&&&&&&&\n" % item)
"""

#Recuperar corpus de un archivo separado por &&&&&&&&

"""with open('newsgroups_trainDataSinStopwords_en.txt', 'r', encoding='utf-8') as f:
    newsgroups_trainDataSinSW = f.read().split('\n&&&&&&&&')
    # eliminar el último elemento de la lista que es vacío
    newsgroups_trainDataSinSW.pop()

with open('newsgroups_testDataSinStopwords_en.txt', 'r', encoding='utf-8') as f:
    newsgroups_testDataSinSW = f.read().split('\n&&&&&&&&')
    # eliminar el último elemento de la lista que es vacío
    newsgroups_testDataSinSW.pop()"""

#Lematizar
"""newsgroups_trainDataLem = [lematizar(nlp(doc)) for doc in newsgroups_train.data]
newsgroups_testDataLem = [lematizar(nlp(doc)) for doc in newsgroups_test.data]
"""
#guardar corpus en un archivo
"""with open('newsgroups_trainDataLem_en.txt', 'w', encoding='utf-8') as f:
    for item in newsgroups_trainDataLem:
        f.write("%s\n&&&&&&&&\n" % item)
with open('newsgroups_testDataLem_en.txt', 'w', encoding='utf-8') as f:
    for item in newsgroups_testDataLem:
        f.write("%s\n&&&&&&&&\n" % item)"""
#Recuperar corpus de un archivo separado por &&&&&&&&
"""with open('newsgroups_trainDataLem_en.txt', 'r', encoding='utf-8') as f:
    newsgroups_trainDataLem = f.read().split('\n&&&&&&&&')
    # eliminar el último elemento de la lista que es vacío
    newsgroups_trainDataLem.pop()
with open('newsgroups_testDataLem_en.txt', 'r', encoding='utf-8') as f:
    newsgroups_testDataLem = f.read().split('\n&&&&&&&&')
    # eliminar el último elemento de la lista que es vacío
    newsgroups_testDataLem.pop()"""

#quitamos stopwords al texto lematizado
"""newsgroups_trainDataLemSinSW = [stopwords(nlp(doc)) for doc in newsgroups_trainDataLem]
newsgroups_testDataLemSinSW = [stopwords(nlp(doc)) for doc in newsgroups_testDataLem]
"""
#guardar corpus en un archivo
"""with open('newsgroups_trainDataLemSinSW_en.txt', 'w', encoding='utf-8') as f:
    for item in newsgroups_trainDataLemSinSW:
        f.write("%s\n&&&&&&&&\n" % item)
with open('newsgroups_testDataLemSinSW_en.txt', 'w', encoding='utf-8') as f:
    for item in newsgroups_testDataLemSinSW:
        f.write("%s\n&&&&&&&&\n" % item)"""
#X_train = newsgroups_train.data
#X_test = newsgroups_test.data

with open('newsgroups_trainDataLemSinSW_en.txt', 'r', encoding='utf-8') as f:
    newsgroups_trainDataLemSinSW = f.read().split('\n&&&&&&&&')
    # eliminar el último elemento de la lista que es vacío
    newsgroups_trainDataLemSinSW.pop()
with open('newsgroups_testDataLemSinSW_en.txt', 'r', encoding='utf-8') as f:
    newsgroups_testDataLemSinSW = f.read().split('\n&&&&&&&&')
    # eliminar el último elemento de la lista que es vacío
    newsgroups_testDataLemSinSW.pop()


#clean text 
"""newsgroups_trainDataLemSinSW_CT = [cleanText(doc) for doc in newsgroups_trainDataLemSinSW]
newsgroups_testDataLemSinSW_CT = [cleanText(doc) for doc in newsgroups_testDataLemSinSW]
"""
#guardar corpus en un archivo
"""with open('newsgroups_trainDataLemSinSW_CT_en.txt', 'w', encoding='utf-8') as f:
    for item in newsgroups_trainDataLemSinSW_CT:
        f.write("%s\n&&&&&&&&\n" % item)
with open('newsgroups_testDataLemSinSW_CT_en.txt', 'w', encoding='utf-8') as f:
    for item in newsgroups_testDataLemSinSW_CT:
        f.write("%s\n&&&&&&&&\n" % item)"""


X_train = newsgroups_trainDataLemSinSW
X_test = newsgroups_testDataLemSinSW



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
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
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
"""from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
clf.fit(vectors_train, y_train)
y_pred = clf.predict(vectors_test)
print ('*****************Logistic Regression****************')
print (classification_report(y_test, y_pred))"""



#multinomial naive bayes
from sklearn.naive_bayes import MultinomialNB
"""clf = MultinomialNB()
clf.fit(vectors_train, y_train)
y_pred = clf.predict(vectors_test)
print ('*****************MultinomialNB****************')
print (classification_report(y_test, y_pred))"""


#kneighbors classifier
from sklearn.neighbors import KNeighborsClassifier
"""clf = KNeighborsClassifier()
clf.fit(vectors_train, y_train)
y_pred = clf.predict(vectors_test)
print ('*****************KNeighborsClassifier****************')
print (classification_report(y_test, y_pred))"""


#random forest classifier
from sklearn.ensemble import RandomForestClassifier
"""clf = RandomForestClassifier()
clf.fit(vectors_train, y_train)
y_pred = clf.predict(vectors_test)
print ('*****************RandomForestClassifier****************')
print (classification_report(y_test, y_pred))"""

#multilayer perceptron
"""from sklearn.neural_network import MLPClassifier
clf = MLPClassifier()
clf.fit(vectors_train, y_train)
y_pred = clf.predict(vectors_test)
print ('*****************MLPClassifier****************')
print (classification_report(y_test, y_pred))
"""
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

# Define el modelo MLP
#clf = MLPClassifier()

from sklearn.neural_network import MLPClassifier

"""clf = MLPClassifier(
    hidden_layer_sizes=(100, 100),  # Número y tamaño de las capas ocultas
    activation= 'relu',  # Función de activación ('relu' es común) indentity
    solver='adam',  # Algoritmo de optimización ('adam' es común)
    alpha=0.0001,  # Término de regularización L2 0.001
    learning_rate='adaptive',  # Tasa de aprendizaje adaptativa
    max_iter=200  # Número máximo de iteraciones
)"""
"""clf = MLPClassifier(
    hidden_layer_sizes=(100, 50),  # Tamaño de las capas ocultas
    activation='relu',             # Función de activación ReLU
    solver='adam',                 # Optimizador Adam
    alpha=0.0001,                  # Término de regularización L2
    learning_rate='constant',      # Tasa de aprendizaje constante
    learning_rate_init=0.001,      # Tasa de aprendizaje inicial
    max_iter=200,                  # Número máximo de iteraciones
    random_state=42                # Semilla para reproducibilidad
)

clf.fit(vectors_train, y_train)  # Donde "vectors_train" son los vectores de características de tu conjunto de datos
y_pred = clf.predict(vectors_test)  # "vectors_test" son los vectores de características del conjunto de prueba

print ('*****************MLPClassifier****************')
print (classification_report(y_test, y_pred))

"""


from sklearn.svm import SVC
pipe = Pipeline([('text_representation', TfidfVectorizer()), ('dimensionality_reduction', TruncatedSVD(300)), ('classifier', MLPClassifier(hidden_layer_sizes=(100,),activation="relu",solver="adam",alpha=0.0001,learning_rate="constant",max_iter=200,random_state=42,early_stopping=True,verbose=True))])
pipe = Pipeline([('text_representation', TfidfVectorizer()), ('dimensionality_reduction', TruncatedSVD(300)), ('classifier', SVC(
    C=100,            # Parámetro de regularización (ajustar según sea necesario)
    kernel='rbf',      # Kernel radial basis function (RBF)
    gamma='scale',    # Escala inversa de la distancia entre puntos para el kernel RBF
    random_state=42   # Semilla para reproducibilidad
))])

pipe.set_params(dimensionality_reduction__n_components=1000)
print (pipe)
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
print (classification_report(y_test, y_pred))

"""
clf = MLPClassifier(
    hidden_layer_sizes=(100,),
    activation="relu",
    solver="adam",
    alpha=0.0001,
    learning_rate="constant",
    max_iter=200,
    random_state=42,
    early_stopping=True,
    verbose=True
)
clf.fit(vectors_train, y_train)
y_pred = clf.predict(vectors_test)
print ('*****************MLPClassifier****************')
print (classification_report(y_test, y_pred))"""

from sklearn.svm import SVC

"""# Crear el clasificador SVC
clf = SVC(
    C=100,            # Parámetro de regularización (ajustar según sea necesario)
    kernel='rbf',      # Kernel radial basis function (RBF)
    gamma='scale',    # Escala inversa de la distancia entre puntos para el kernel RBF
    random_state=42   # Semilla para reproducibilidad
)

# Entrenar el modelo y realizar la predicción
clf.fit(vectors_train, y_train)  # Asegúrate de tener tus datos de entrenamiento (X_train, y_train)
y_pred = clf.predict(vectors_test)  # Asegúrate de tener tus datos de prueba (X_test)

# Imprime el informe de clasificación
print('*****************SVC****************')
print(classification_report(y_test, y_pred))  # Donde "y_test" son las etiquetas verdaderas del conjunto de prueba
"""
