import matplotlib.pyplot as plt
import numpy as np
import cupy as cp
from keras.datasets import mnist
from sklearn.metrics import confusion_matrix


def random_sample(train_size):
    index = np.random.randint(60000, size=train_size)
    sampled_data = X_train[:, index]
    return sampled_data


def kernel_choice(a, b, c=1, d=2, sigmoid_c=0.1, theta=-10, sigma=15, kernel_type="polynomial"):
    kernel = a @ b
    if kernel_type == "polynomial":
        K = (kernel + c) ** d
    elif kernel_type == "gaussian":
        K = np.zeros((len(b.T), len(b.T)))
        for i in range(len(a)):
            for j in range(len(b.T)):
                K[i, j] = np.exp(-np.linalg.norm(b[:, i] - b[:, j]) / (2 * (sigma ** 2)))
    else:
        K = np.tanh(sigmoid_c * kernel + theta)
    np.save(f"K_sigmoid{len(b.T)}.npy", K)
    return K


def distance_kernel(K, j, sum_1, sum_2):
    return sum_1 + K[j, j] - sum_2[:, j]


def W_init():
    w_current = np.zeros((train_size, k))
    for i in range(train_size):
        w_current[i, np.random.randint(k)] = 1
    pj = np.sum(w_current, axis=0)
    w_current = w_current / pj
    return w_current


def W_update(w_current, K):
    iteration = 0
    check = False
    while not check:
        iteration += 1
        print("Current iteration: ", iteration)
        w_update = np.zeros((train_size, k))
        sum_1 = np.diag(w_current.T @ (K @ w_current))
        sum_2 = 2 * (w_current.T @ K)
        for i in range(train_size):
            w_update[i, np.argmin(distance_kernel(K, i, sum_1, sum_2))] = 1
        partition_sum = np.sum(w_update, axis=0)
        w_update = w_update / partition_sum
        if np.array_equal(w_current, w_update):
            check = True
        if iteration == 200:
            check = True
        w_current = w_update
    np.save("W_30k_sigmoid.npy", w_update)
    return iteration


def kernel_dist_test(index, sum1, sum2):
    return sum1 - sum2[:, index]


def test():
    W_test = np.zeros((test_size, k))
    sum1 = np.diag(w_update.T @ (K @ w_update))
    sum2 = 2 * (w_update.T @ K2)
    for i in range(test_size):
        W_test[i, np.argmin(kernel_dist_test(i, sum1, sum2))] = 1
    return W_test


def results():
    arr = []
    y_pred = []
    y_true = []
    for i in range(test_size):
        if (W_test[i, 0] != 0):
            arr.append(i)
            pred_label = 0
        elif (W_test[i, 1] != 0):
            arr.append(i)
            pred_label = 9
        elif (W_test[i, 2] != 0):
            arr.append(i)
            pred_label = 6
        elif (W_test[i, 3] != 0):
            arr.append(i)
            pred_label = 5
        elif (W_test[i, 4] != 0):
            arr.append(i)
            pred_label = 1
        elif (W_test[i, 5] != 0):
            arr.append(i)
            pred_label = 6
        elif (W_test[i, 6] != 0):
            arr.append(i)
            pred_label = 0
        elif (W_test[i, 7] != 0):
            arr.append(i)
            pred_label = 7
        elif (W_test[i, 8] != 0):
            arr.append(i)
            pred_label = 8
        else:
            arr.append(i)
            pred_label = 3
        y_pred.append(pred_label)
        y_true.append(y_test[arr[i]])
    check_all = np.zeros(test_size)
    for i in range(test_size):
        check_all[i] = (y_pred[i] == y_true[i])
    return y_pred, y_true, check_all


def show_clusters(index):
    arr = []
    for i in range(test_size):
        if W_test[i, index] != 0:
            arr.append(i)
    plt.gray()  # B/W Images
    plt.figure(figsize=(10, 9))  # Adjusting figure size
    temp = X_test.reshape(28, 28, -1)
    # print(len(arr))
    for i in range(0, len(arr)):
        plt.subplot(1, len(arr), i + 1)
        plt.imshow(temp[:, :, arr[i]])


def show_centroids():
    temp_keeper = []
    for x in range(k):
        arr = []
        for i in range(len(w_update)):
            if w_update[i, x] != 0:
                arr.append(i)
        temp_add = np.zeros([len(w_update), 784])
        for i in range(len(arr)):
            temp_add[i, :] = sampled_data[:, arr[i]]
        centroid_matrix = np.zeros(784)
        for i in range(784):
            centroid_matrix[i] = np.mean(temp_add[:, i])
        temp_keeper.append(centroid_matrix)
    plt.gray()  # B/W Images
    plt.figure(figsize=(10, 9))  # Adjusting figure size
    temp_keeper = np.array(temp_keeper)
    temp = temp_keeper.reshape(k, 28, 28)
    for i in range(10):
        plt.subplot(3, 4, i + 1)
        plt.imshow(temp[i])


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    # Normalization
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    X_train = x_train.reshape(len(x_train), -1).T
    X_test = x_test.reshape(len(x_test), -1).T

    train_size = 30000
    k = 10
    test_size = 10000
    sigma = 15
    kernel_type = "gaussian"

    X_test = X_test[:, :test_size]

    sampled_data = random_sample(train_size)
    K = kernel_choice(sampled_data.T, sampled_data, kernel_type=kernel_type)
    # K = np.load("K_sigmoid30000.npy")
    iter_no = W_update(W_init(), K)

    if kernel_type == "gaussian":
        K_gaussian = np.zeros((train_size, test_size))
        for i in range(train_size):
            for j in range(test_size):
                K_gaussian[i, j] = np.exp(-np.linalg.norm(sampled_data[:, i] - X_test[:, j]) / (2 * (sigma ** 2)))
        K2 = K_gaussian
    else:
        K2 = kernel_choice(sampled_data.T, X_test, sigma=sigma, kernel_type=kernel_type)

    w_update = np.load("W_30k_sigmoid.npy")
    W_test = test()
    show_clusters(0)
    y_pred, y_true, check_all = results()
    print(confusion_matrix(y_true, y_pred))
    print(sum(check_all) / test_size)
