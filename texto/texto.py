# Imports
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
from pandas import DataFrame, Series
#from nltk import twitter

# Definições
class Texto:

    def __init__(self, entrada, arquivo=True, stop_words=None):
        '''
        Parametros:
          entrada = 
            str: nome de um arquivo
            list: lista de strings
          arquivo = só importa se a entrada for uma lista
            True = a lista será tratada como uma lista de nomes de arquivos
            False = a lista será tratada como uma lista de documentos
        Retorno: Objeto Texto
        '''
        entrada = entrada if type(entrada) is list else open(entrada)
        vector = CountVectorizer(input = 'filename' if arquivo and type(entrada) is list else 'content',
                                 stop_words = stop_words)

        # Matriz Documentos X Termos
        # Documentos -> Linhas
        # Termos -> Colunas
        self.ocorrencias = vector.fit_transform(entrada)

        # Lista com as palavras
        self.palavras_distintas = vector.get_feature_names()
        
        self.palavras_ignoradas = vector.get_stop_words()

    def para_dataframe(self):
        return DataFrame(self.ocorrencias, columns=self.palavras_distintas)

    def para_frequencias(self):
        return Series(self.ocorrencias.sum(axis=0).A1, index=self.palavras_distintas)

    def barplot(self, max=10):
        '''
        Parametros:
          max = número máximo de barras
        Retorno: Plot de barras
        '''
        return self.para_frequencias().sort_values()[-max:].plot.barh(fontsize=13)

    def wordcloud(self, min_ocorrencias=-1, **kwargs):
        '''
        Parametros:
          gt = Numero minimo de ocorrencias da palavra para aparecer
        Retorna: Imagem com um WordCloud do texto
        '''
        f = self.para_frequencias()
        return WordCloud(**kwargs) \
          .generate_from_frequencies(frequencies=f[f > min_ocorrencias]).to_image()