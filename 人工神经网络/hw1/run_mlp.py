import argparse
from network import Network
from utils import LOG_INFO
from layers import Relu, Sigmoid, Linear
from loss import EuclideanLoss, SoftmaxCrossEntropyLoss
from solve_net import train_net, test_net
from load_data import load_mnist_2d

def build_model_from_string(def_str):
    model = Network()
    def_str.strip()
    layer_strs = def_str.split(';')
    for layer_str in layer_strs:
        tokens = layer_str.split(',')
        if(len(tokens) <= 1):
            raise Exception("Invalid token: {} in layer definition".format(layer_str))
        type = tokens[0].strip()
        name = tokens[1].strip()
        if(type == "linear"):
            model.add(Linear(name, int(tokens[2]), int(tokens[3]), float(tokens[4])))
        elif(type == "sigmoid"):
            model.add(Sigmoid(name))
        elif(type == "relu"):
            model.add(Relu(name))
        else:
            raise NotImplementedError("Unsupported layer type {}".format(type))
    print("="*50 + "\nModel Summary:\n{}\n".format(model) + "="*50 + "\n")
    return model

if(__name__ == '__main__'):
    parser = argparse.ArgumentParser(description="MLP Trainer")
    parser.add_argument('-lr', '--learning-rate', default=0.01, type=float)
    parser.add_argument('-wd', '--weight-decay', default=0.0005, type=float)
    parser.add_argument('-mm', '--momentum', default=0.9, type=float)
    parser.add_argument('-b', '--batch-size', default=100, type=int)
    parser.add_argument('-e', '--max-epoch', default=100, type=int)
    parser.add_argument('-d', '--display-freq', default=50, type=int)
    parser.add_argument('-t', '--test-epoch', default=5, type=int)
    parser.add_argument('-layer', help="Layer definition string.\
        For a single layer, format: LAYER_TYPE,NAME,EXTRAS.\
        For multi-layers, separate format strings of single layers with delimiter ';'\
        For linear, EXTRAS should be INPUT_DIM,OUTPUT_DIM,INITIAL_STDEV\
        For relu/sigmoid, EXTRAS should be left empty", default="linear,fc1,784,10,0.01", type=str)
    parser.add_argument('-loss', help="Loss function, should be either mse or cross_entropy", default="mse", type=str)
    parser.add_argument('-o', '--output', help="Output filename")
    args = parser.parse_args()
    train_data, test_data, train_label, test_label = load_mnist_2d('data')
    if(args.loss == "mse"):
        loss = EuclideanLoss(name='loss')
    elif(args.loss == "cross_entropy"):
        print("Using cross entropy loss")
        loss = SoftmaxCrossEntropyLoss(name='loss')
    else:
        raise NotImplementedError("Unsupported loss function {}".format(args.loss))
    model = build_model_from_string(args.layer)
    config = {
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'momentum': args.momentum,
        'batch_size': args.batch_size,
        'max_epoch': args.max_epoch,
        'disp_freq': args.display_freq,
        'test_epoch': args.test_epoch
    }

    loss_list = []
    acc_list = []

    for epoch in range(config['max_epoch']):
        if epoch != 0 and epoch % config['test_epoch'] == 0:
            LOG_INFO('Testing @ %d epoch...' % (epoch))
            test_net(model, loss, test_data, test_label, config['batch_size'])
        LOG_INFO('Training @ %d epoch...' % (epoch))
        loss_, acc_ = train_net(model, loss, config, train_data, train_label, config['batch_size'], config['disp_freq'])
        loss_list.extend(loss_)
        acc_list.extend(acc_)

    LOG_INFO('Final test')
    test_net(model, loss, test_data, test_label, config['batch_size'])

    import numpy as np
    from datetime import datetime
    name = args.output or 'Network_' + datetime.now().strftime("%Y-%m-%d_%H%M%S")
    np.save(name, [loss_list, acc_list])
