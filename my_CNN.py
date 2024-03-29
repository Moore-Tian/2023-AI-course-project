import numpy as np

np.set_printoptions(threshold=np.inf)


class CNN_:

    def __init__(self, input_shape, DNN_layers, num_channels=1, num_Kernels=(3, 3), kernel_size=(3, 3), stride=2, 
                 padding=1, pooling_size=(2, 2), learning_rate=0.05, batch_size=800, epochs=100, 
                  dropout=0, activate='leakyrelu', X_valid=[], y_valid=[]):
        self.input_shape = input_shape
        self.num_conv_layer = len(num_Kernels)
        self.DNN_layers = DNN_layers
        self.num_DNN_layers = len(DNN_layers)
        self.num_channels = num_channels
        self.num_Kernels = num_Kernels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.pooling_size = pooling_size
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.dropout = dropout
        self.epochs = epochs
        self.activate = activate
        self.X_valid = X_valid
        self.y_valid = y_valid
        self.Weights = []
        self.Biases = []
        self.Kernels = []
        self.Conv_Biases = []

        self.initialize_weights_and_kernels()

    def compute_DNN_input_size(self):
        input_rows, input_cols = self.input_shape
        for i in range(self.num_conv_layer):
            input_rows = (input_rows - self.kernel_size[0] + 2*self.padding)//self.stride + 1
            input_cols = (input_cols - self.kernel_size[1] + 2*self.padding)//self.stride + 1
            input_rows = (input_rows - self.pooling_size[0])//self.stride + 1
            input_cols = (input_cols - self.pooling_size[1])//self.stride + 1
        
        return input_rows, input_cols

    def initialize_weights_and_kernels(self):
        # 先计算DNN网络的输入尺寸，初始化第一个权重
        DNN_input_rows, DNN_input_cols = self.compute_DNN_input_size()
        DNN_input_total = self.num_channels*DNN_input_rows * DNN_input_cols *np.prod(np.array(self.num_Kernels))
        self.Weights.append(np.random.randn(self.DNN_layers[0], DNN_input_total)*(np.sqrt(2/(DNN_input_total + self.DNN_layers[0]))))
        self.Biases.append(np.zeros((self.DNN_layers[0], 1)))
        for i in range(1, self.num_DNN_layers):
            Weight = np.random.randn(self.DNN_layers[i], self.DNN_layers[i-1])*(np.sqrt(2/(self.DNN_layers[i-1] + self.DNN_layers[i])))
            Bias = np.zeros((self.DNN_layers[i], 1))
            self.Weights.append(Weight)
            self.Biases.append(Bias)

        for i in range(self.num_conv_layer):
            lay_kernel = []
            lay_bias = []
            for j in range(self.num_Kernels[i]):
                Kernel = np.random.randn(self.kernel_size[0], self.kernel_size[1])
                lay_kernel.append(Kernel)
                lay_bias.append(0)
            self.Kernels.append(lay_kernel)
            self.Conv_Biases.append(lay_bias)

    def f_activate(self, x):
        if self.activate == 'leakyrelu':
            return self.leakyrelu(x)
        elif self.activate == 'sigmoid':
            return self.sigmoid(x)

    def leakyrelu(self, x):
        return np.where(x > 0, x, x * 0.00001)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def softmax(self, X):
        X_exp = np.exp(X-np.max(X, axis=1, keepdims=True))
        row_sums = np.sum(X_exp, axis=1, keepdims=True)
        softmax = X_exp / row_sums
        return softmax

    def DNN_normalize(self, X):
        row_means = np.mean(X, axis=1, keepdims=True)
        row_stds = np.std(X, axis=1, keepdims=True)
        return (X - row_means) / row_stds, np.concatenate((row_means, row_stds), axis=1)

    def conv(self, X, Kernel, stride):
        in_height, in_width = X.shape[0:2]
        kernel_height, kernel_width = Kernel.shape
        out_height = (in_height - kernel_height) // stride + 1
        out_width = (in_width - kernel_width) // stride + 1
        out = np.zeros((out_height, out_width))

        for i in range(out_height):
            for j in range(out_width):
                start_i = i * self.stride
                start_j = j * self.stride
                end_i = start_i + kernel_height
                end_j = start_j + kernel_width
                # 如果越界，则忽略剩余部分
                if end_i > in_height or end_j > in_width:
                    continue
                out[i, j] = np.sum(X[start_i:end_i, start_j:end_j] * Kernel)
        return out

    def f_padding(self, X):
        padded_X = np.pad(X, self.padding, mode='constant', constant_values=0)
        return padded_X

    def pooling(self, X):
        input_height, input_width = X.shape
        pool_height, pool_width = self.pooling_size
        out_height = (input_height - pool_height) // self.stride + 1
        out_width = (input_width - pool_width) // self.stride + 1
        out = np.zeros((out_height, out_width))
        record = np.zeros_like(X)

        for i in range(out_height):
            for j in range(out_width):
                start_i = i * self.stride
                start_j = j * self.stride
                end_i = start_i + pool_height
                end_j = start_j + pool_width

                if end_i > input_height or end_j > input_width:
                    continue

                pool_window = X[start_i:end_i, start_j:end_j]
                out[i, j] = np.max(pool_window)
                max_index = np.argmax(pool_window)
                row_index, col_index = np.unravel_index(max_index, pool_window.shape)
                record[start_i + row_index, start_j + col_index] = 1

        return out, record

    def full_forward_propogation(self, X, iftest=0):
        Conv_out, pool_records, size_records = self.Conv_forward_propogation(X)
        print(np.array(Conv_out[-1]).shape)
        DNN_in = np.array(Conv_out[-1]).reshape(self.batch_size, -1)
        DNN_out = self.DNN_forward_propogation(DNN_in, iftest)
        return Conv_out, DNN_out, pool_records, size_records

    def Conv_forward_propogation(self, X):
        activations = [X]
        pool_records = []
        # 该列表用于记录每次卷积之后、池化之前的尺寸，方便后续求导时的计算
        size_records = []

        batch_results = []
        batch_pool_records = []

        for batch_sample in range(self.batch_size):
            channels_results = []
            channels_pool_records = []

            for channel_num in range(len(activations[-1][batch_sample])):
                kernels_results = []
                kernels_pool_records = []

                for kernrl_num in range(self.num_Kernels[0]):

                    result_after_activate = self.f_activate(self.conv(self.f_padding(activations[-1][batch_sample][channel_num]), self.Kernels[0][kernrl_num], self.stride)+self.Biases[0][kernrl_num])
                    size_record = result_after_activate.shape
                    conv_result, pool_record = self.pooling(result_after_activate)
                    kernels_results.append(conv_result)
                    kernels_pool_records.append(pool_record)

                channels_results.append(kernels_results)
                channels_pool_records.append(kernels_pool_records)

            batch_results.append(channels_results)
            batch_pool_records.append(channels_pool_records)

        activations.append(batch_results)
        print("out_shape", np.array(activations[-1]).shape)
        pool_records.append(batch_pool_records)
        size_records.append(size_record)

        for layer_num in range(1, self.num_conv_layer):
            batch_results = []
            batch_pool_records = []

            for batch_sample in range(self.batch_size):
                channels_results = []
                channels_pool_records = []
                for channel_num in range(len(activations[-1][batch_sample])*self.num_Kernels[layer_num-1]):
                    kernels_results = []
                    kernels_pool_records = []

                    for kernrl_num in range(self.num_Kernels[layer_num]):

                        result_after_activate = self.f_activate(self.conv(self.f_padding(activations[-1][batch_sample][channel_num//self.num_Kernels[layer_num-1]][kernrl_num % self.num_Kernels[layer_num-1]]), self.Kernels[layer_num][kernrl_num], self.stride)+self.Conv_Biases[layer_num][kernrl_num])
                        size_record = result_after_activate.shape
                        conv_result, pool_record = self.pooling(result_after_activate)
                        kernels_results.append(conv_result)
                        kernels_pool_records.append(pool_record)

                    channels_results.append(kernels_results)
                    channels_pool_records.append(kernels_pool_records)

                batch_results.append(channels_results)
                batch_pool_records.append(channels_pool_records)

            activations.append(batch_results)
            pool_records.append(batch_pool_records)
            size_records.append(size_record)

        return activations, pool_records, size_records

    def Dropout(self, X, iftest=0):
        if iftest == 1:
            return X
        else:
            return X * np.random.binomial(1, 1-self.dropout, size=X.shape)

    def DNN_forward_propogation(self, X, iftest=0):
        activations = [X]
        if self.activate == 'sigmoid':
            for i in range(self.num_DNN_layers - 1):
                activations.append(self.sigmoid(self.Dropout((np.dot(self.Weights[i], activations[i].T) + self.Biases[i]).T, iftest=iftest)))
        elif self.activate == 'leakyrelu':
            for i in range(self.num_DNN_layers - 1):
                activations.append(self.leakyrelu(self.Dropout((np.dot(self.Weights[i], activations[i].T) + self.Biases[i]).T, iftest=iftest)))

        activations.append((np.dot(self.Weights[-1], activations[-1].T) + self.Biases[-1]).T)
        activations[-1] = self.softmax(activations[-1])

        return activations

    def cross_entropy(self, y_true, y_pred):
        return np.sum(np.multiply(-y_true, np.log(y_pred)), axis=1), y_pred - y_true

    # 该函数是为了得到计算梯度时所需要的一个部分
    def get_deltas(self, activations):
        deltas = []
        if self.activate == 'sigmoid':
            for i in range(self.num_DNN_layers - 1):
                deltas.append(np.multiply(activations[i+1], 1-activations[i+1]))
        if self.activate == 'leakyrelu':
            for i in range(self.num_DNN_layers - 1):
                deltas.append(np.where(activations[i+1] >= 0, 1, 0.00001))

        return deltas
    
    # 已知对激活结果的梯度和激活结果，求对激活前的输入的梯度
    def conv_deactivate(self, gradient, activation):
        A = np.ones(self.pooling_size)
        if self.activate == 'sigmoid':
            delta = np.multiply(activation, 1-activation)
            gradient_to_beforeact = np.multiply(np.kron(delta, A), gradient)
        elif self.activate == 'leakyrelu':
            delta = np.where(activation >= 0, 1, 0.00001)
            gradient_to_beforeact = np.multiply(np.kron(delta, A), gradient)

        return gradient_to_beforeact

    def de_pooling(self, gradient, record, before_size):
        A = np.ones(self.pooling_size)
        gradient_to_beforepool = np.multiply(np.kron(gradient, A), record)

        # 因为池化中可能存在防止越界而省去的操作，所以这里需要补零来还原
        expanded_rows = before_size[0] - gradient_to_beforepool.shape[0]
        expanded_cols = before_size[1] - gradient_to_beforepool.shape[1]
        gradient_to_beforepool = np.pad(gradient_to_beforepool, ((0, expanded_rows), (0, expanded_cols)), mode='constant')

        return gradient_to_beforepool

    # 用于在矩阵的元素之间插零，方便之后的卷积内核求导运算
    def insert_zeros(self, matrix, i):
        rows, cols = matrix.shape
        new_rows = rows + (rows - 1) * i
        new_cols = cols + (cols - 1) * i
        result = np.zeros((new_rows, new_cols))

        result[::i+1, ::i+1] = matrix

        return result

    # 此处的matrix是进行卷积操作之前的矩阵
    def grad_to_kernel(self, matrix, gradient_base):
        matrix = np.array(matrix)
        row_cut = (matrix.shape[0] - self.kernel_size[0])//self.stride
        col_cut = (matrix.shape[1] - self.kernel_size[1])//self.stride
        
        matrix_cut = matrix[:-row_cut, :-col_cut]
        print('matrix_cut = ', matrix_cut.shape)

        gradient_base = self.insert_zeros(gradient_base, self.stride-1)

        return self.conv(matrix_cut, gradient_base, 1)

    def grad_to_X(self, kernel, gradient_base):
        kernel_rot = np.rot90(kernel, 2)
        gradient_base = self.insert_zeros(gradient_base, self.stride-1)
        pad_row = self.kernel_size[0] - 1
        pad_col = self.kernel_size[1] - 1
        gradient_base = np.pad(gradient_base, ((pad_row, pad_row), (pad_col, pad_col)), mode='constant')

        return self.conv(gradient_base, kernel_rot, 1)[self.padding[0]:-self.padding[0], self.padding[1]:-self.padding[1]]

    def full_backward_propagation(self, DNN_out, Conv_out, pool_records, size_records, y):
        gradients_W, gradients_b = self.DNN_backward_propagation(DNN_out, y)
        input_rows, input_cols = self.compute_DNN_input_size()
        gradient_base = np.dot(gradients_b[0], self.Weights[0]).reshape(self.batch_size, self.num_channels*np.prod(np.array(self.num_Kernels)), input_rows, input_cols)
        print("4486486416516548561111111111111118: ", gradient_base.shape)
        gradient_base = gradient_base.tolist()
        gradients_kernel, gradiens_con_bias = self.Conv_backward_propagation(gradient_base, Conv_out, pool_records, size_records)
        return gradients_W, gradients_b, gradients_kernel, gradiens_con_bias

    def DNN_backward_propagation(self, activations, y):
        _, loss_grad = self.cross_entropy(y, activations[-1])
        gradients_W = []
        gradients_b = []
        deltas = self.get_deltas(activations)

        # 第一步的梯度计算（第一步放在循环外是为了防止在循环内部写if语句）
        gradient_b = loss_grad
        gradients_b.insert(0, gradient_b)
        gradients_W.insert(0, gradient_b[:, :, np.newaxis] * activations[-2][:, np.newaxis, :])
        for i in range(2, self.num_DNN_layers+1):
            gradient_b = np.multiply(np.dot(gradient_b, self.Weights[-i+1]), deltas[-i+1])
            gradients_b.insert(0, gradient_b)
            gradients_W.insert(0, gradient_b[:, :, np.newaxis] * activations[-i-1][:, np.newaxis, :])

        return gradients_W, gradients_b

    def Conv_backward_propagation(self, gradient_base, Conv_out, pool_records, size_records):
        gradients_kernel = []
        gradients_con_bias = []
        for i in range(self.num_conv_layer):
            layer_gradient_kernel = []
            layer_gradient_con_bias = []
            for kernel_num in range(self.num_Kernels[-i-1]):
                kernel_gradient_sum = np.zeros_like(self.Kernels[-i-1][kernel_num])
                bias_gradient_sum = 0
                for batch_sample in range(self.batch_size):
                    for conv_channel in range(len(Conv_out[-i-1][batch_sample])):
                        gradient_to_beforepool = self.de_pooling(gradient_base[batch_sample][conv_channel], pool_records[-i-1][batch_sample][conv_channel][kernel_num], size_records[-i-1])
                        gradient_to_beforeact = self.conv_deactivate(gradient_to_beforepool, Conv_out[-i-1][batch_sample][conv_channel][kernel_num])
                        gradient_to_kernel = self.grad_to_kernel(Conv_out[-i-2], gradient_to_beforeact)
                        gradient_to_bias = gradient_to_beforeact
                        kernel_gradient_sum += gradient_to_kernel
                        bias_gradient_sum += gradient_to_bias
                gradient_kernel = kernel_gradient_sum / (self.batch_size*self.num_channels)
                gradient_bias = bias_gradient_sum / (self.batch_size*self.num_channels)
                layer_gradient_kernel.insert(0, gradient_kernel)
                layer_gradient_con_bias.insert(0, gradient_bias)
            gradients_kernel.insert(0, layer_gradient_kernel)
            gradients_con_bias.insert(0, layer_gradient_con_bias)

            # 求对下一层输出的导数
            new_gradient_base = []
            for batch_sample in range(self.batch_size):
                new_gradients_for_batch = []
                for conv_channel in range(len(Conv_out[-i-2][batch_sample])):
                    new_gradiens_for_channel = []
                    for kernel_num in range(self.num_Kernels):
                        new_gradient_sum = 0
                        # 拆分求导求和
                        for the_kernel_num in range(self.num_Kernels):
                            new_gradient_sum += self.grad_to_X(self.Kernels[-i-1], gradient_base[batch_sample][3*conv_channel+kernel_num][the_kernel_num])
                        new_gradiens_for_channel.append(new_gradient_sum/self.num_Kernels)
                    new_gradients_for_batch.append(new_gradiens_for_channel)
                new_gradient_base.append(new_gradients_for_batch)
            gradient_base = new_gradient_base
                        
        return gradients_kernel, gradients_con_bias

    def update_weights(self, gradients_W, gradients_b):
        for i in range(self.num_layers-1):
            self.Weights[i] -= self.learning_rate * (np.sum(gradients_W[i], axis=0)/self.batch_size)
            self.Biases[i] -= self.learning_rate * (np.sum(gradients_b[i], axis=0)[:, np.newaxis]/self.batch_size)

    def update_kernels(self, gradients_kernel, gradients_con_bias):
        for i in range(self.num_layers-1):
            for j in range(self.num_Kernels[i]):
                self.kernels[i][j] -= self.learning_rate * gradients_kernel[i][j]
                self.con_bias[i][j] -= self.learning_rate * gradients_con_bias[i][j]

    def train(self, X, y):
        train_accuracy_scores = []
        valid_accuracy_scores = []
        i = 0
        X = np.array(X)
        while i < self.epochs:
            mapping = np.arange(X.shape[0])
            np.random.shuffle(mapping)

            for j in range(0, X.shape[0], self.batch_size):
                if j + self.batch_size > X.shape[0]:
                    X_batch = X[mapping[-self.batch_size:], :]
                    y_batch = y[mapping[-self.batch_size:]]
                else:
                    X_batch = X[mapping[j:j+self.batch_size], :]
                    y_batch = y[mapping[j:j+self.batch_size]]
                Conv_out, DNN_out, pool_records, size_records = self.full_forward_propogation(X_batch)
                gradients_W, gradients_b, gradients_kernel, gradients_con_bias = self.full_backward_propagation(DNN_out, Conv_out, pool_records, size_records, y_batch)
                self.update_weights(gradients_W, gradients_b)
                self.updata_kernels(gradients_kernel, gradients_con_bias)

            losses, _ = self.cross_entropy(y_batch, DNN_out[-1])
            loss = np.sum(losses) / self.batch_size
            train_pred = np.argmax(np.array(self.predict(X)), axis=1)
            y_train = np.argmax(y, axis=1)
            valid_pred = np.argmax(np.array(self.predict(self.X_valid)), axis=1)
            train_accuracy_scores.append(self.accuracy_score(y_train, train_pred))
            valid_accuracy_scores.append(self.accuracy_score(self.y_valid, valid_pred))
            print(f"Epoch {i}: loss = {loss}, train acc = {train_accuracy_scores[-1]}, valid acc = {valid_accuracy_scores[-1]}, learning_rate = {self.learning_rate}")

            i += 1

            """ if i == self.epochs - 1:
                extra = int(input("你还想要几个epoch: "))
                self.epochs += extra """

            if train_accuracy_scores[-1] > 0.98:
                self.learning_rate = 0.005

            """ self.learning_rate *= 0.99

            self.learning_rate = max(self.learning_rate, 0.001) """

    def predict(self, X):
        _, y_pred, _, _ = self.full_forward_propogation(X, iftest=1)
        return y_pred[-1]

    def accuracy_score(self, y_true, y_pred):
        equal = (y_true == y_pred)
        acc = np.sum(equal) / equal.size
        return acc
