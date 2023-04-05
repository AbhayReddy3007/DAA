import numpy as np
import torch as tr
import pandas as pd
import tensorflow as tf
def function():
    data = pd.read_csv(r"Bank_Personal_Loan_Modelling.csv")
    data.drop(['ID'] , axis = 1 , inplace = True)
    x = data.drop(['Personal Loan'] , axis = 1).values
    y = data['Personal Loan'].values
    x = tr.tensor(x , dtype = tr.float64)
    y = tr.tensor(y , dtype=  tr.float64)
    y = y.to(tr.float64)
    from sklearn.model_selection import train_test_split
    x_train , x_test , y_train , y_test = train_test_split(x , y , random_state = 42 , test_size = 0.25)
    return x_train , x_test , y_train , y_test

class NN(tr.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = tr.nn.Linear(12, 10)
        self.linear2 = tr.nn.Linear(10, 20)
        self.linear3 = tr.nn.Linear(20 , 1)
        self.relu = tr.nn.ReLU()
        self.sigmoid = tr.nn.Sigmoid()

    def forward(self, x):
        x = self.linear1(x.float())
        x = self.relu(x.float())
        x = self.linear2(x.float())
        x = self.linear3(x.float())
        x = self.relu(x.float())
        x = self.sigmoid(x.float())
        return x

model = NN()
loss_function = tr.nn.MSELoss()

class CulturalOptimizer:
    def __init__(self, model, population_size, mutation , decay ,  inputs  , labels):
        self.model = model
        self.population_size = population_size
        self.mutation = mutation
        self.population = self.init_population()
        self.decay = decay
        self.inputs = inputs
        self.labels = labels
        self.culture = None

    def init_population(self):
        population = []
        for i in range(self.population_size):
            weights = []
            for weight in self.model.parameters():
                weights.append(weight.data.numpy())
            population.append(weights)
        return population

    def selection(self, fitness_scores):
        cumulative_scores = np.cumsum(fitness_scores)
        total_score = np.sum(fitness_scores)
        rand = np.random.uniform(0, total_score)
        selected_index = np.searchsorted(cumulative_scores, rand)
        return selected_index

    def crossover(self, male, female):
        random_crossover = np.random.randint(1, len(male))
        child1 = female[:random_crossover] + male[random_crossover:]
        child2 = female[:random_crossover] + male[random_crossover:]
        return child1, child2
    
    def decay_mutation_rate(self):
        self.mutation -= (self.decay*self.mutation)

    def mutate(self, child):
        for i in range(len(child)):
            if np.random.uniform(0, 1) < self.mutation:
                child[i] += np.random.normal(0, 0.1, child[i].shape)
        return child

    def generate_offspring(self, fitness_scores):
        new_population = []
        for _ in range(self.population_size):
            parent1_index = self.selection(fitness_scores)
            parent2_index = self.selection(fitness_scores)
            parent1 = self.population[parent1_index]
            parent2 = self.population[parent2_index]
            child1, child2 = self.crossover(parent1, parent2)
            child1 = self.mutate(child1)
            child2 = self.mutate(child2)
            new_population.append(child1)
            new_population.append(child2)
        self.population = new_population

    def update_weight(self):
        fitness_scores = [self.fitness(weights) for weights in self.population]
        best_index = np.argmax(fitness_scores)
        best_weights = self.population[best_index]
        for i, param in enumerate(self.model.parameters()):
            param.data = torch.Tensor(best_weights[i])

    def fitness(self, weights):
        if (self.culture == None):
            for i, param in enumerate(self.model.parameters()):
                param.data = torch.Tensor(weights[i])
            outputs = self.model(self.inputs)
            loss = loss_function(outputs.float(), self.labels.reshape([len(self.inputs) , 1]).float())
            return 1 / (loss.item() + 1e-6)

x_train , x_test , y_train , y_test = function()
culturalOptimizer = CulturalOptimizer(model, population_size=20, mutation=0.3  , decay = 0.05 , inputs = x_train, labels = y_train)

def train(num_epochs):
    loss_list = []
    with tf.device('/gpu:0'):
        for epoch in range(num_epochs):
            culturalOptimizer.generate_offspring([])
            culturalOptimizer.update_weight()
            outputs = model(x_train)
            loss = loss_function(outputs, y_train.reshape([len(x_train) , 1]).float())
            loss_list.append(loss.item())
            loss.backward()
            culturalOptimizer.generate_offspring([])
            culturalOptimizer.update_weight()
            if (epoch%10 == 0):
                print("Epoch" , epoch , " : " , loss.item());
                culturalOptimizer.decay_mutation_rate()
    return loss_list
