#Extract from the news corpus the news section and generate a new corpus with this content
#-Tokenize the extracted content and create a tokenized corpus
#-Lemmatize the corpus
#-Remove stop words that are articles,prepositions,conjunctions and pronouns

import re
import spacy
from nltk.corpus import stopwords
# read corpus from file
# Abre el archivo y lee su contenido
with open('corpus_noticias.txt', 'r', encoding='utf-8') as archivo:
    corpus_original = archivo.read()

# Use regex to extract the news section between "&&&&&" delimiters
expresion  =  r'([^&]+)&&&&&&&&'
news_sections = re.findall(expresion, corpus_original)

news_sections = news_sections[2::3]

"""#save the news sections in a file
with open('corpus_noticias_secciones.txt', 'w', encoding='utf-8') as archivo:
    for news in news_sections:
        archivo.write(news + '\n')"""


# Cargar el modelo de spaCy en español
nlp = spacy.load("es_core_news_sm")

# TOKENIZAR
tokenized_sections = []

for section in news_sections:
    # Procesar el texto con SpaCy
    doc = nlp(section)
    
    # Obtener tokens del documento
    tokens = [token.text for token in doc]
    
    # Agregar la lista de tokens a 'tokenized_sections'
    tokenized_sections.append(tokens)

# Lemmatizar


lemmatized_sections = []

for section_tokens in tokenized_sections:
    section_text = " ".join(section_tokens)
    doc = nlp(section_text)
    lemmatized_tokens = [token.lemma_ for token in doc]
    lemmatized_sections.append(lemmatized_tokens)

# Remove stop words that are articles,prepositions,conjunctions and pronouns en español
# Agregar signos de puntuación a la lista de stopwords
#punctuation = ['.', ',', ';', '!', '?', ':', '-', '"', '(', ')', '[', ']', '{', '}', '...', '¡', '¿', '»', '«', '...', '``', "''", '/', '|', '“', '”','*','él']

stopwords  = [
    "el", "la", "los", "las", "un", "una", "unos", "unas","la","lo","a","al","del"  # Artículos
    "a", "ante", "bajo", "con", "contra", "de", "desde", "en", "entre", "hacia", "para", "por", "según", "sin", "sobre", "tras",  # Preposiciones
    "y", "o", "pero", "ni", "que", "si", "como", "porque","u",  # Conjunciones
    "yo", "tú", "él", "ella", "nosotros", "vosotros", "ellos", "ellas", "mí", "ti", "sí", "nos", "vos", "se", "me", "te", "le", "nos", "os", "les", "se",  # Pronombres personales
]

#stopwords = stopwords.union(punctuation)#spacy.lang.es.stop_words.STOP_WORDS.union(punctuation)

# Eliminar stopwords de cada lista de tokens lematizados
filtered_sections = []

for lemmatized_tokens in lemmatized_sections:
    filtered_tokens = [token for token in lemmatized_tokens if token.lower() not in stopwords]
    filtered_sections.append(filtered_tokens)

# Save the filtered corpus in a file
with open('corpus_final.txt', 'w', encoding='utf-8') as archivo:
    for section in filtered_sections:
        section_text = " ".join(section)
        archivo.write(section_text + '\n')


