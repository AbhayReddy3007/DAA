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

class AntColonyOptimizer:
    def _init_(self, num_of_ants, epochs,   initial_pheremone, decay_rate , size  , inputs , labels):
        self.num_of_ants = num_of_ants
        self.epochs = epochs
        self.initial_pheremone = initial_pheremone
        self.decay_rate = decay_rate
        self.size = size
        self.pheromone = np.full((2, self.size, 1), self.pheromone_init)
        self.inputs = inputs
        self.labels = labels
        
    def fun(self, self.inputs , self.labels):
        w = np.zeros((self.inputs, self.labels))
        for i in range(inputs):
            for j in range(output):
                prob = self.get_transition_prob(i, j)
                w[i][j] = np.random.normal(loc=prob, scale=0.5)
        return w
    
    def update_weight(self):
        fitness_scores = [self.fitness(weights) for weights in self.population]
        best_index = np.argmax(fitness_scores)
        best_weights = self.population[best_index]
        for i, param in enumerate(self.model.parameters()):
            param.data = torch.Tensor(best_weights[i])

x_train , x_test , y_train , y_test = function()
antcolonyoptimizer = AntColonyOptimizer(model, num_of_ants =20, epochs = 100  , decay_rate = 0.05 , inputs = x_train, labels = y_train)

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
