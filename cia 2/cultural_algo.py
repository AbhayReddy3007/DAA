import numpy as np
import torch as tr
import pandas as pd
import tensorflow as tf

def fx():

    data = pd.read_csv(r"Bank_Personal_Loan_Modelling.csv")

    data.drop(['ID'] , axis = 1 , inplace = True)
    x = data.drop(['Personal Loan'] , axis = 1).values
    x = tr.tensor(x , dtype = tr.float64)

    y = data['Personal Loan'].values
    y = tr.tensor(y , dtype=  tr.float64)
    y = y.to(tr.float64)

    from sklearn.model_selection import chain_test_split
    x_chain , x_test , y_chain , y_test = chain_test_split(x , y , random_state = 42 , test_size = 0.25)

    return x_chain , x_test , y_chain , y_test

class class_n(tr.class_n.Module):

    def init(self):

        super().init()

        self.linear1 = tr.class_n.Linear(13, 9)
        self.linear2 = tr.class_n.Linear(15, 25)
        self.linear3 = tr.class_n.Linear(25 , 1)
        self.relu = tr.class_n.ReLU()
        self.sigmoid = tr.class_n.Sigmoid()

model = class_n()
loss_fx = tr.class_n.MSELoss()

class ACO:
    def init(self, no_ants, epochs,   first_ph, rate_of_decay , size  , inputs , labels):

        self.no_ants = no_ants; self.epochs = epochs
        self.first_ph = first_ph
        self.rate_of_decay = rate_of_decay ; self.size = size
        self.pheromone = np.full((2, self.size, 1), self.pheromone_init)
        self.inputs = inputs ;  self.labels = labels
        
    def fun(self, self.inputs , self.labels):
        w = np.zeros((self.inputs, self.labels))
        for i in range(inputs):
            for j in range(output):
                prob = self.get_transition_prob(i, j)
                w[i][j] = np.random.normal(loc=prob, scale=0.5)
        return w
    
    def update_weight(self):
        best_weights = self.population[np.argmax([self.fitness(weights) for weights in self.population])]
        for i, param in enumerate(self.model.parameters()):
            param.data = torch.Tensor(best_weights[i])

x_chain , x_test , y_chain , y_test = fx()
ACO = ACO(model, no_ants =50, epochs = 125  , rate_of_decay = 0.06 , inputs = x_chain, labels = y_chain)

def chain(num_epochs):
    loss_list = []
    with tf.device('/gpu:0'):
        for epoch in range(num_epochs):
            cO.generate_offspring([])
            cO.update_weight()
            outputs = model(x_chain)
            loss = loss_fx(outputs, y_chain.reshape([len(x_chain) , 1]).float())
            loss_list.append(loss.item())
            loss.backward()
            cO.generate_offspring([])
            cO.update_weight()

            if (!(epoch%15)):
                print("for epoch" , epoch , " , loss item :  " , loss.item());
                cO.decay_mutation_rate()

    return loss_list
