# I have no idea what I'm doing...

# Cетка написана с помощью учебника "Создаем Нейронную сеть" Тарика Рашида

import numpy as np
import scipy.special


# Определение класса нейронной сети

class neuralNetwork:
  # Инициализация нейронной сети
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes


        # матрицы весовых коэфов связей wih и who
        # Весовые коэфы связей между узлом i и j
        # следующего слоя обозначены как w_i_j
        # w11 w21
        # w12 w22
        self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5), 
                                    (self.hnodes, self.inodes))
        self.who = np.random.normal(0.0, pow(self.onodes, -0.5), 
                                    (self.onodes, self.hnodes))

        


    #Коэф обучения
        self.lr = learningrate
      # Лямбда для сигмойды в качестве активации
        self.activation_function = lambda x: scipy.special.expit(x)

      
    def train(self, inputs_list, targets_list):
        # Входные значения в двумерный массив
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        # Расчёт входящих сигналов для скрытого слоя
        hidden_inputs = np.dot(self.wih, inputs)
        # расчёт исходящих сигналов для скрытого слоя
        hidden_outputs = self.activation_function(hidden_inputs)

        # Расчёт входящих сигналов для выходного слоя
        final_inputs = np.dot(self.who, hidden_outputs)
        # Расчёт исходящих сигналов для выходного слоя
        final_outputs = self.activation_function(final_inputs)

        # ошибка = цель - факт значения сети
        output_errors = targets - final_outputs

        hidden_errors = np.dot(self.who.T, output_errors)

        # Код для обновления весовых коэфов связей между скрытым и выходным слоями
        self.who += self.lr * np.dot((output_errors * final_outputs *
                                      (1.0 - final_outputs)), np.transpose(hidden_outputs))
        
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs *
                                      (1.0 - hidden_outputs)), np.transpose(inputs))
        pass
    # опрос нейронной сети
    def query(self, input_list):
        # GПреобразование списка входных значений в двумерный массив
        inputs = np.array(input_list, ndmin = 2).T
        # Расчёт входящих сигналов для скрытого слоя
        hidden_inputs = np.dot(self.wih, inputs)
        # расчёт исходящих сигналов для скрытого слоя
        hidden_outputs = self.activation_function(hidden_inputs)

        # Расчёт входящих сигналов для выходного слоя
        final_inputs = np.dot(self.who, hidden_outputs)
        # Расчёт исходящих сигналов для выходного слоя
        final_outputs = self.activation_function(final_inputs)

        return final_outputs
        
# Количество входных, скрытых и выходных узлов

input_nodes = 3
hidden_nodes = 3
output_nodes = 3

# коэф обучения
learning_rate = 0.3

# создаем экземпляр нейронной сети
n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
