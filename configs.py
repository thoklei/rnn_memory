class DefaultConfig(object):
    num_epochs = 1000
    batchsize = 128
    layer_dim = 50
    learning_rate = 1e-4
    max_grad_norm = 5.0
    fw_lambda = 0.9
    fw_eta = 0.5
    c_alpha = 12
    c_lambda = 0.065
    init_scale = 0.1
    layer_norm = True
    norm_gain =  1
    norm_shift = 1


class Default_MNIST_Config(DefaultConfig):
    input_length = 784
    input_dim = 1
    embedding_size = 100
    output_dim = 10


class Default_AR_Config(DefaultConfig):
    input_length = 9
    input_dim = 26+10+1
    embedding_size = 100
    output_dim = 26+10+1
