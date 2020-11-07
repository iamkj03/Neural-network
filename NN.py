##############################################
# CSE5523 Machine Learning                    #
# HW5                                         #
# error back propagation algorithm            #
# Kyeong Joo Jung                             #
# 500411516                                   #
##############################################

import argparse
import sys
import random
import numpy as np

###Using argparse to set parameters
parser = argparse.ArgumentParser(
    description="Please enter the parameters(train_data.txt/train_target.txt/hiddenlayernumber/unitofhiddenlayer/activation function/loss function/output.txt/learning rate/epoch/batchsize/tol) needed")
parser.add_argument('-A', help="train data text file")
parser.add_argument('-y', help="train target text file")
parser.add_argument('-ln', type=int, help="number of hidden layers")
parser.add_argument('-un',
                    help="number of units of each hidden layer / should be separated by ,s and the total number of hidden layers should be inputted")
parser.add_argument('-a', help="activation function")
parser.add_argument('-ls', help="specify loss function(SSE, CE)")
parser.add_argument('-out', help="output text file")
parser.add_argument('-lr', type=float, help="learning rate")
parser.add_argument('-nepochs', type=int, help="maximum number of epochs")
parser.add_argument('-bs', type=int, help="batch size")
parser.add_argument('-tol', type=float, help="specifies the minimal SSE/CE")
args = parser.parse_args()

### variables set from the parameters

train_data = str(args.A)
train_target = str(args.y)
num_hidden = int(args.ln)
unit_number = str(args.un)
activation = str(args.a)
loss_func = str(args.ls)
output_file = str(args.out)
learning_rate = float(args.lr)
epoch = int(args.nepochs)
batch = int(args.bs)
tol = float(args.tol)

unit = []  # string number of units of each layer

for i in range(len(unit_number.split(','))):
    if unit_number.split(',')[i] == '' or num_hidden != len(
            unit_number.split(',')):  # if there are additional , or numbers does not match, exit the program
        print("input number of un should be same as ln size and there shouldn't be empty element")
        sys.exit()
    unit.append(int(unit_number.split(',')[i]))  # add the numbers into unit list


def shuffle_data(split_list, list, total_size, bs):  # shuffles the element order
    for i in range(total_size):
        i = i + 1
        list.append(i)  # make empty list with element number in order
    random.shuffle(random_element)  # shuffle the list's order
    a = total_size // bs
    b = total_size % bs
    if (b == 0):  # if the sample num / batch = 0,
        for i in range(a):
            new = []
            for j in range(bs):
                new.append(random_element[j + bs * i])
            split_list.append(new)  # all the batch has same number
    if (b != 0):  # if not,
        for i in range(a):
            new = []
            for j in range(bs):
                new.append(random_element[j + (bs - 1) * i])
            split_list.append(new)
        new = []
        for f in range(b):
            new.append(random_element[f + (bs - 1) * a])
        split_list.append(new)  # last layer will have different number

    return split_list


def layer_init(seed, num_layer, num_unit, feature):
    np.random.seed(seed)
    weightval = {}

    for i in range(num_layer):
        li = i + 1
        if (i == num_layer - 1):  # last hidden layer
            layerin = num_unit[i - 1]
            layerout = num_unit[i]
        elif (i == 0):  # first layer
            layerin = feature
            layerout = num_unit[i]
        else:  # other hidden layers
            layerin = num_unit[i - 1]
            layerout = num_unit[i]

        weightval['W' + str(li)] = np.random.randn(layerout, layerin) * 0.1
        weightval['W0_' + str(li)] = np.random.randn(layerout, 1) * 0.1

    weightval['W' + str(num_layer + 1)] = np.random.randn(1, num_unit[num_layer - 1]) * 0.1  # last layer
    weightval['W0_' + str(num_layer + 1)] = np.random.randn(1, 1) * 0.1
    return weightval  # return weights


def sig(z):  # sigmoid function
    return 1 / (1 + np.exp(-z))


def sig_inv(z, da):  # sigmoid inverse for the back propagation
    inv = sig(z)
    return da * inv * (1 - inv)


def tahn(z):  # tahn function
    return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))


def tahn_inv(z, da):  # tahn inverse for the back propagation
    inv = tahn(z)
    return da * (1 - (inv * inv))


def for_pro(data, weight, num_layer, act):  # forward propagation
    zanda = {}  # place to store z and a
    a_i = data  # ith a is data

    for i in range(num_layer):  # going in to the beginning of the layer
        li = i + 1
        a_i_1 = a_i  # ith a would i-1th a because it's going to the next layer

        cur_weight = weight['W' + str(li)]  # current weight and bias
        cur_w0 = weight['W0_' + str(li)]

        cur_z = np.dot(cur_weight, a_i_1) + cur_w0  # current z
        if (act == 'sigmoid'):
            a_i = sig(cur_z)
        elif (act == 'tahn'):  # choosing the activation function
            a_i = tahn(cur_z)
        else:
            print('Only \'sigmoid\' and \'tahn\' is available for the activation function')
            sys.exit()  # program ends if the activation function is not sigmoid or tahn

        zanda['a' + str(i)] = a_i_1
        zanda['z' + str(li)] = cur_z  # keep the values in z and a{}

    a_i_1 = a_i
    cur_z = np.dot(weight['W' + str(num_layer + 1)], a_i) + weight[
        'W0_' + str(num_layer + 1)]  # everything is the same but this is for the last layer not hidden layer
    if (act == 'sigmoid'):
        a_i = sig(cur_z)
    elif (act == 'tahn'):  # choosing the activation function
        a_i = tahn(cur_z)
    else:
        print('Only \'sigmoid\' and \'tahn\' is available for the activation function')
        sys.exit()

    zanda['a' + str(num_layer)] = a_i_1
    zanda['z' + str(num_layer + 1)] = cur_z  # adding last a_i and z to zanda
    return a_i, zanda  # return the current a, and zanda{}


def lossf(loss_func, pred, y):  # calculating loss function 1: cross entropy 2: squared error
    if (loss_func == 'CE'):
        return -np.dot(y, np.log(pred)) + np.dot(1 - y, np.log(1 - pred))  # 1
    elif (loss_func == 'SSE'):
        return (pred - y) ** 2.0  # 2


def lossf_deri(loss_func, pred, y):  # calculating loss function derivation 1: cross entropy 2: squared error
    if (loss_func == 'CE'):
        return np.divide(1 - y, 1 - pred) - np.divide(y, pred)  # 1
    elif (loss_func == 'SSE'):
        return 2 * pred - 2 * y  # 2


def back_pro(pred, y, za, weight_value, num_layer, act, loss_func):  # back propagation function
    gradient = {}  # list to contain gradients
    da_i_1 = lossf_deri(loss_func, pred, y)  # derivation of i-1th a

    for i in range(num_layer):
        j = num_layer - i  # opposite direction since it is back order
        li = j + 1

        da_i = da_i_1  # derivation of a

        a_i_1 = za['a' + str(j)]  # bringing i-1th a
        unit_num = a_i_1.shape[1]  # number of units in the layer
        z_i = za['z' + str(li)]  # bringing ith z
        w_i = weight_value['W' + str(li)]  # bringing ith w
        b_i = weight_value['W0_' + str(li)]  # bringing ith bias
        if (act == 'sigmoid'):
            dz_i = sig_inv(z_i, da_i)
        elif (act == 'tahn'):
            dz_i = tahn_inv(z_i, da_i)  # choosing activation function

        dw_i = np.dot(dz_i, a_i_1.T) / unit_num
        db_i = np.sum(dz_i, keepdims=True, axis=1) / unit_num
        da_i_1 = np.dot(w_i.T, dz_i)  # calculating derivation of ith w, b and i-1th a

        gradient['dw' + str(li)] = dw_i
        gradient['db' + str(li)] = db_i  # storing them in gradient

    a_i_1 = za['a0']  # bringing a1 for first layer

    unit_num = a_i_1.shape[0]  # number of units in the first layer
    z_i = za['z1']  # bringing first z
    w_i = weight_value['W1']  # bringing first w
    b_i = weight_value['W0_1']  # bringing first bias
    if (act == 'sigmoid'):
        dz_i = sig_inv(z_i, da_i_1)
    elif (act == 'tahn'):
        dz_i = tahn_inv(z_i, da_i_1)  # choosing activation function for first layer

    dw_i = np.dot(dz_i, a_i_1.T) / unit_num
    db_i = np.sum(dz_i, keepdims=True, axis=1) / unit_num
    da_i_1 = np.dot(w_i.T, dz_i)  # calculating derivation of ith w, b and i-1th a

    gradient['dw1'] = dw_i
    gradient['db1'] = db_i  # storing them in gradient

    return gradient


if __name__ == '__main__':
    print('train data: ', train_data)
    print('train target: ', train_target)
    print('num hidden: ', num_hidden)
    print('unit number: ', unit_number)
    print('activation func: ', activation)
    print('loss func: ', loss_func)
    print('output file: ', output_file)
    print('learning rate: ', learning_rate)
    print('epoch: ', epoch)
    print('batch: ', batch)
    print('tol: ', tol)  # printing out the parameters

    X_data = np.genfromtxt(train_data, delimiter=" ")
    X_target = np.genfromtxt(train_target)
    X_data = np.array(X_data)
    X_target = X_target.reshape(len(X_data), 1)  # preprocessing the data
    sample_size, feature_size = X_data.shape  # achieving sample size and feature size from X train data.txt

    print('X_data: ', X_data.shape)
    print('X_target: ', X_target.shape)

    weight_value = layer_init(2, num_hidden, unit, feature_size)  # initializing layers

    random_element = []
    split_data = []
    ranspl = shuffle_data(split_data, random_element, sample_size,
                          batch)  # shuffled and splitted elements by batch size

    avg_error = []  # list to store errors
    for i in range(epoch):  # number of epochs

        for j in ranspl:  # iterating data by batch size
            error_sum = 0
            avg_val = 0
            for data in j:  # starting batch training
                pred, za = for_pro(X_data[data - 1].reshape(len(X_data[data - 1]), 1), weight_value, num_hidden,
                                   activation)  # forward propagation
                err = lossf(loss_func, pred, X_target[data - 1])  # error of one data sample
                error_sum = error_sum + err

                gradient_val = back_pro(pred, X_target[data - 1], za, weight_value, num_hidden, activation,
                                        loss_func)  # back propagation

                for index in range(num_hidden):  # updating the weight
                    lindex = index + 1
                    weight_value['W' + str(lindex)] -= learning_rate * gradient_val[
                        'dw' + str(lindex)]  # updating hidden layers
                    weight_value['W0_' + str(lindex)] -= learning_rate * gradient_val['db' + str(lindex)]
                weight_value['W' + str(num_hidden + 1)] -= learning_rate * gradient_val[
                    'dw' + str(num_hidden + 1)]  # updating last layer
                weight_value['W0_' + str(num_hidden + 1)] -= learning_rate * gradient_val['db' + str(num_hidden + 1)]

            avg_val = error_sum / len(j)
            avg_error.append(avg_val)  # sppending the batch avg error to avg_error

            if avg_val < tol:
                file = open(output_file, "w")
                for row in avg_error:
                    np.savetxt(file, row)
                sys.exit()  # comparing the loss value with tol

    print(avg_error)
    print('finished!')
    file = open(output_file, "w")
    for row in avg_error:
        np.savetxt(file, row)  # exporting the txt file with the values from SGD function
