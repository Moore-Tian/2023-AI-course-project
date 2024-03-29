import numpy as np
import sys

np.set_printoptions(threshold=np.inf)


class back_propagation():

    def __init__(self, layers, learning_rate=0.02, batch_size=16, 
                 epochs=100, dropout=0, classifacation=False, lamda=0,
                 activate='leakyrelu', X_valid=[], y_valid=[]):
        self.layers = layers
        self.num_layers = len(layers)
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.dropout = dropout
        self.lamda = lamda
        self.classifacation = classifacation
        if self.classifacation is True:
            self.loss = 'cross_entropy'
        else:
            self.loss = 'mean_squared_error'
        self.activate = activate
        self.Weights = []
        self.Biases = []
        self.X_valid = X_valid
        self.y_valid = y_valid
        self.initialize_weights_and_biases()

    def initialize_weights_and_biases(self):
        for i in range(self.num_layers - 1):
            Weight = np.random.randn(self.layers[i+1], self.layers[i])*(np.sqrt(2/(self.layers[i+1] + self.layers[i])))
            Bias = np.zeros((self.layers[i+1], 1))
            self.Weights.append(Weight)
            self.Biases.append(Bias)

    # 用于让用户载入既有的权重（可以用来载入模型）
    def user_defined_weights(self, user_Weights, user_Biases):
        self.Weights = user_Weights
        self.Biases = user_Biases

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def leakyrelu(self, x):
        return np.where(x > 0, x, x * 0.001)

    def softmax(self, X):
        X_exp = np.exp(X-np.max(X, axis=1, keepdims=True))
        row_sums = np.sum(X_exp, axis=1, keepdims=True)
        softmax = X_exp / row_sums
        return softmax

    # 用于将对某一层的输出向量随机置零
    def Dropout(self, X, iftest=0):
        if iftest == 1:
            return X
        else:
            return X * np.random.binomial(1, 1-self.dropout, size=X.shape)

    def forward_propogation(self, X, iftest=0):
        activations = [X]
        if self.activate == 'sigmoid':
            for i in range(self.num_layers - 2):
                activations.append(self.sigmoid(self.Dropout((np.dot(self.Weights[i], activations[i].T) + self.Biases[i]).T, iftest=iftest)))
        elif self.activate == 'leakyrelu':
            for i in range(self.num_layers - 2):
                activations.append(self.leakyrelu(self.Dropout((np.dot(self.Weights[i], activations[i].T) + self.Biases[i]).T, iftest=iftest)))

        activations.append((np.dot(self.Weights[-1], activations[-1].T) + self.Biases[-1]).T)

        if self.classifacation is True:
            activations[-1] = self.softmax(activations[-1])

        return activations

    def loss_compute(self, y_true, y_pred):
        if self.loss == 'mean_squared_error':
            return self.square_error(y_true, y_pred)
        else:
            return self.cross_entropy(y_true, y_pred)

    def square_error(self, y_true, y_pred):
        return np.sum((y_pred - y_true)**2, axis=1)/2, y_pred - y_true

    def cross_entropy(self, y_true, y_pred):
        return np.sum(np.multiply(-y_true, np.log(y_pred + sys.float_info.epsilon)), axis=1), y_pred - y_true

    # 该函数是为了得到计算梯度时所需要的一个部分
    def get_deltas(self, activations):
        deltas = []
        if self.activate == 'sigmoid':
            for i in range(self.num_layers - 2):
                deltas.append(np.multiply(activations[i+1], 1-activations[i+1]))
        if self.activate == 'leakyrelu':
            for i in range(self.num_layers - 2):
                deltas.append(np.where(activations[i+1] >= 0, 1, 0.001))

        return deltas

    def backward_propagation(self, activations, y):
        _, loss_grad = self.loss_compute(y, activations[-1])
        gradients_W = []
        gradients_b = []

        deltas = self.get_deltas(activations)

        # 第一步的梯度计算（第一步放在循环外是为了防止在循环内部写if语句）
        gradient_b = loss_grad
        gradients_b.insert(0, gradient_b)
        gradients_W.insert(0, gradient_b[:, :, np.newaxis] * activations[-2][:, np.newaxis, :])
        for i in range(2, self.num_layers):
            gradient_b = np.multiply(np.dot(gradient_b, self.Weights[-i+1]), deltas[-i+1])
            gradients_b.insert(0, gradient_b)
            gradients_W.insert(0, gradient_b[:, :, np.newaxis] * activations[-i-1][:, np.newaxis, :])

        return gradients_W, gradients_b

    def update_weights(self, gradients_W, gradients_b):
        for i in range(self.num_layers-1):
            self.Weights[i] -= self.learning_rate * (np.sum(gradients_W[i], axis=0)/self.batch_size)
            self.Biases[i] -= self.learning_rate * (np.sum(gradients_b[i], axis=0)[:, np.newaxis]/self.batch_size)

    def train(self, X, y):
        train_accuracy_scores = []
        valid_accuracy_scores = []
        loss_scores = []
        i = 0
        for i in range(self.epochs):
            mapping = np.arange(X.shape[0])
            np.random.shuffle(mapping)

            for j in range(0, X.shape[0], self.batch_size):
                if j + self.batch_size > X.shape[0]:
                    X_batch = X[mapping[-self.batch_size:], :]
                    y_batch = y[mapping[-self.batch_size:]]
                else:
                    X_batch = X[mapping[j:j+self.batch_size], :]
                    y_batch = y[mapping[j:j+self.batch_size]]
                activations = self.forward_propogation(X_batch)
                gradients_W, gradients_b = self.backward_propagation(activations, y_batch)
                self.update_weights(gradients_W, gradients_b)

            losses, _ = self.loss_compute(y_batch, activations[-1])
            loss = np.sum(losses) / self.batch_size
            train_pred = np.argmax(np.array(self.predict(X)), axis=1)
            y_train = np.argmax(y, axis=1)
            valid_pred = np.argmax(np.array(self.predict(self.X_valid)), axis=1)
            train_accuracy_scores.append(self.accuracy_score(y_train, train_pred))
            valid_accuracy_scores.append(self.accuracy_score(self.y_valid, valid_pred))
            loss_scores.append(loss)
            print(f"Epoch {i}: loss = {loss}, train acc = {train_accuracy_scores[-1]}, valid acc = {valid_accuracy_scores[-1]}, learning_rate = {self.learning_rate}")
            if train_accuracy_scores[-1] > 0.99:
                self.learning_rate = 0.001
            elif train_accuracy_scores[-1] > 0.98:
                self.learning_rate = 0.005

        return train_accuracy_scores, valid_accuracy_scores, loss_scores

    def predict(self, X):
        y_pred = self.forward_propogation(X, iftest=1)
        return y_pred[-1]

    def accuracy_score(self, y_true, y_pred):
        equal = (y_true == y_pred)
        acc = np.sum(equal) / equal.size
        return acc
