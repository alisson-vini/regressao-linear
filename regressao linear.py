import random
import matplotlib.pyplot as plt
import numpy as np

# Construção da classe Perceptron (treino, atualização e inicialização dos pesos, calculo de hipotese)
class Perceptron: # cria uma classe pra representar um perceptron simples
    def __init__(self):
        self.peso = random.uniform(-1,1) # peso
        self.vies = random.uniform(-1,1) # vies


    # calcula a hipotese dada x entrada
    def hipotese (self, x):
        return self.peso * x + self.vies


    # atualiza o peso e vies dada uma entrada e saida esperada (a função de custo utilizada foi a MSE)
    def atualizar_pesos(self, entrada,valor_real, taxa_aprendizagem):
        somatorio_gradiente_peso = 0
        somatorio_gradiente_vies = 0

        for entrada_rede, saida_correta in zip(entrada, valor_real):
            h = self.hipotese(entrada_rede)
            erro = saida_correta - h
            somatorio_gradiente_peso += erro * entrada_rede
            somatorio_gradiente_vies += erro

        # atualiza o peso e o vies
        self.peso += (taxa_aprendizagem * somatorio_gradiente_peso) / len(entrada)
        self.vies += (taxa_aprendizagem * somatorio_gradiente_vies) / len(entrada)


    # treina o perceptron por x epocas
    def treinamento(self, conjunto_entradas, conjunto_saidas, epocas, taxa_aprendizagem):

        for epoca in range(epocas):
            self.atualizar_pesos(conjunto_entradas, conjunto_saidas, taxa_aprendizagem)



# pode alterar, acrescentar ou retirar valores dos conjuntos de entrada e saída para alterar a reta que vai ser criada desde que as listas de x e y tenham a mesma quantidade de elementos
entrada_x = [0.5, 1.7, 2.4, 4.1, 5.9, 6.3, 7.0, 8.8, 10.2, 12.0] # conjunto de entradas x
saida_y   = [20.1, 18.3, 17.5, 15.6, 13.2, 12.6, 11.4, 9.7, 7.9, 6.2] # conjunto de entradas y
neuronio = Perceptron() # cria o perceptron

x_vals = np.linspace(min(entrada_x), max(entrada_x), 250) # cria um vetor contendo varios valores para plotar a reta
fig, ( (graf1,graf2), (graf3,graf4) ) = plt.subplots(nrows=2, ncols=2, figsize=(13,8)) # inicializa os gráficos



# salva um vetor com cada hipotese gerada pelo perceptron para os valores de x_vals (sem treino)
y_0_treinos = [neuronio.hipotese(x) for x in x_vals]

# treina o perceptron 10 vezes
neuronio.treinamento(entrada_x, saida_y, 10, 0.01)
y_10_treinos = [neuronio.hipotese(x) for x in x_vals] # salva um vetor com cada hipotese gerada pelo perceptron para os valores de x_vals (com 10 treinos ao todo)

# treina o perceptron 500 vezes
neuronio.treinamento(entrada_x, saida_y, 490, 0.01)
y_50_treinos = [neuronio.hipotese(x) for x in x_vals] # salva um vetor com cada hipotese gerada pelo perceptron para os valores de x_vals (com 50 treinos ao todo)

# treina ele 2000 vezes
neuronio.treinamento(entrada_x, saida_y, 1500, 0.01)
y_100_treinos = [neuronio.hipotese(x) for x in x_vals] # salva um vetor com cada hipotese gerada pelo perceptron para os valores de x_vals (com 100 treinos ao todo)

# printa a hipotese final do perceptron dado o conjunto de entradas
for entrada in entrada_x:
    print( round(neuronio.hipotese(entrada),2) )


# configura o grafico (sem treino)
graf1.set_title("sem treino", fontsize=10)
graf1.scatter(entrada_x, saida_y, color='orange', label='ponto real')
graf1.grid()
graf1.legend()
graf1.plot(x_vals, y_0_treinos, linewidth=2)

# configura o grafico (10 iterações de treino)
graf2.set_title("10 iterações de treino", fontsize=10)
graf2.scatter(entrada_x, saida_y, color='orange', label='ponto real')
graf2.grid()
graf2.legend()
graf2.plot(x_vals, y_10_treinos, linewidth=2)

# configura o grafico (500 iterações de treino)
graf3.set_title("500 iterações de treino", fontsize=10)
graf3.scatter(entrada_x, saida_y, color='orange', label='ponto real')
graf3.grid()
graf3.legend()
graf3.plot(x_vals, y_50_treinos, linewidth=2)

# configura o grafico (2000 iterações de treino)
graf4.set_title("2000 iterações de treino", fontsize=10)
graf4.scatter(entrada_x, saida_y, color='orange', label='ponto real')
graf4.legend()
graf4.grid()
graf4.plot(x_vals, y_100_treinos, linewidth=2)

plt.show() # exibe o gráfico