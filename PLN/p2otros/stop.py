import spacy

# Cargar el modelo de lenguaje para español
nlp = spacy.load("es_core_news_sm")

# Acceder a las stopwords
stopwords = spacy.lang.es.stop_words.STOP_WORDS

# Mostrar algunas de las stopwords
print(list(stopwords))  # Muestra las primeras 10 stopwords

# Cargar el modelo de lenguaje para español
nlp = spacy.load("es_core_news_sm")

# Definir categorías de stopwords
categorias_stopwords = {
    "ARTICULOS": {"el", "la", "los", "las", "un", "una", "unos", "unas"},
    "PREPOSICIONES": {"a", "ante", "bajo", "con", "contra", "de", "desde", "en", "entre", "hacia", "para", "por", "según", "sin", "sobre", "tras"},
    "CONJUNCIONES": {"y", "e", "ni", "o", "u", "pero", "aunque", "si", "que"},
    "PRONOMBRES": {"yo", "tú", "él", "ella", "nosotros", "vosotros", "ellos", "ellas", "mí", "ti", "sí", "nos", "vos", "se", "me", "te", "le", "nos", "os", "les", "se"},
}

# Acceder a las stopwords y clasificarlas
stopwords = spacy.lang.es.stop_words.STOP_WORDS
stopwords_clasificadas = {
    categoria: sorted([palabra for palabra in stopwords if palabra in categorias_stopwords[categoria]])
    for categoria in categorias_stopwords
}

# Mostrar las stopwords clasificadas
for categoria, palabras in stopwords_clasificadas.items():
    print(f"{categoria}: {', '.join(palabras)}")
#En este código, he agrupado las stopwords en las categorías de ARTICULOS, PREPOSICIONES, CONJUNCIONES y PRONOMBRES, y luego he mostrado ejemplos de palabras que pertenecen a cada una de estas categorías. Ten en cuenta que esta lista de ejemplos no es exhaustiva, pero te da una idea de las palabras comunes en cada categoría. Puedes ampliar la lista de palabras en cada categoría según tus necesidades específicas.





