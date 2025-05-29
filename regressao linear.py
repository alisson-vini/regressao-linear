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
x_vals = np.linspace(min(entrada_x), max(entrada_x), 100) # cria um vetor x para plotar a reta
fig, ( (graf1,graf2), (graf3,graf4) ) = plt.subplots(nrows=2, ncols=2, figsize=(13,8))


neuronio = Perceptron() # cria o perceptron

y_0_treinos = [neuronio.hipotese(x) for x in x_vals]

neuronio.treinamento(entrada_x, saida_y, 10, 0.01) # treina ele 1000 vezes (pode alterar para ver o precesso de aprendizagem)
y_10_treinos = [neuronio.hipotese(x) for x in x_vals]

neuronio.treinamento(entrada_x, saida_y, 40, 0.01) # treina ele 1000 vezes (pode alterar para ver o precesso de aprendizagem)
y_50_treinos = [neuronio.hipotese(x) for x in x_vals]

neuronio.treinamento(entrada_x, saida_y, 50, 0.01) # treina ele 1000 vezes (pode alterar para ver o precesso de aprendizagem)
y_100_treinos = [neuronio.hipotese(x) for x in x_vals]

for entrada in entrada_x:
    print( round(neuronio.hipotese(entrada),2) ) # printa a hipotese do perceptron dado o conjunto de entradas

graf1.set_title("sem treino", fontsize=10)
graf1.scatter(entrada_x, saida_y, color='orange', label='ponto real')
graf1.grid()
graf1.legend()
graf1.plot(x_vals, y_0_treinos, linewidth=2)

graf2.set_title("10 iterações de treino", fontsize=10)
graf2.scatter(entrada_x, saida_y, color='orange', label='ponto real')
graf2.grid()
graf2.legend()
graf2.plot(x_vals, y_10_treinos, linewidth=2)
plt.legend()

graf3.set_title("50 iterações de treino", fontsize=10)
graf3.scatter(entrada_x, saida_y, color='orange', label='ponto real')
graf3.grid()
graf3.legend()
graf3.plot(x_vals, y_50_treinos, linewidth=2)
plt.legend()

graf4.set_title("100 iterações de treino", fontsize=10)
graf4.scatter(entrada_x, saida_y, color='orange', label='ponto real')
graf4.legend()
graf4.grid()
graf4.plot(x_vals, y_100_treinos, linewidth=2)

plt.show() # exibe o gráfico

