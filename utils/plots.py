import matplotlib.pyplot as plt


def plot_loss(history):
    # loss
    plt.plot(history.history['loss'], label='train loss')
    plt.plot(history.history['val_loss'], label='val loss')
    plt.legend()


def plot_acc(history, name):
    # accuracies
    plt.plot(history.history['accuracy'], label='train accuracy')
    plt.plot(history.history['val_accuracy'], label='val accuracy')
    plt.legend()
    plt.savefig(f'AccVal_acc_LossVal_loss_{name}')
