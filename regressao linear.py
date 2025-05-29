import random
import matplotlib.pyplot as plt
import numpy as np


class Perceptron: # cria uma classe pra representar um perceptron simples
    def __init__(self):
        self.peso = random.uniform(-1,1) # peso
        self.vies = random.uniform(-1,1) # vies

    def hipotese (self, x): # calcula a hipotese dada x entrada
        return self.peso * x + self.vies

    def atualizar_pesos(self, entrada,valor_real, taxa_aprendizagem): # atualiza o peso e vies dada uma entrada e saida esperada
        somatorio_gradiente_peso = 0
        somatorio_gradiente_vies = 0

        for entrada_rede, saida_correta in zip(entrada, valor_real):
            h = self.hipotese(entrada_rede)
            erro = saida_correta - h
            somatorio_gradiente_peso += erro * entrada_rede
            somatorio_gradiente_vies += erro

        self.peso += 2 * taxa_aprendizagem * somatorio_gradiente_peso # atualiza o peso
        self.vies += 2 * taxa_aprendizagem * somatorio_gradiente_vies # atualiza o vies

    def treinamento(self, conjunto_entradas, conjunto_saidas, epocas, taxa_aprendizagem): # treina o perceptron por x epocas

        for epoca in range(epocas):
            self.atualizar_pesos(conjunto_entradas, conjunto_saidas, taxa_aprendizagem)


entrada_x = [1,2,3,4,5,6] # conjunto de entradas x
saida_y = [20,10,7,11,5,2] # conjunto de entradas y


neuronio = Perceptron() # cria o perceptron
neuronio.treinamento(entrada_x, saida_y, 100, 0.01) # treina ele 1000 vezes (pode alterar para ver o precesso de aprendizagem)

for entrada in entrada_x:
    print( round(neuronio.hipotese(entrada),2) ) # printa a hipotese do perceptron dado o conjunto de entradas




x_vals = np.linspace(min(entrada_x), max(entrada_x), 100) # cria um vetor x para plotar a reta

y_vals = [neuronio.hipotese(x) for x in x_vals] # calcula as previsões da reta para esses x

plt.scatter(entrada_x, saida_y, color='red', label='Dados Reais') # plot dos pontos reais

plt.plot(x_vals, y_vals, color='blue', label='Reta Ajustada (hipótese)') # plot da reta ajustada

# informações do gráfico
plt.xlabel('Entrada (x)')
plt.ylabel('Saída (y)')
plt.title('Regressão Linear com Perceptron')
plt.legend()
plt.grid(True)

plt.show() # exibe o gráfico