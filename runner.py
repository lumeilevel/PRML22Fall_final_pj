import tensorflow as tf
from tensorflow import keras
from eda import load_image
from matplotlib import pyplot as plt
import numpy as np

def masked_loss(labels, preds):
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels, preds)

    mask = (labels != 0) & (loss < 1e8)
    mask = tf.cast(mask, loss.dtype)

    loss = loss * mask
    loss = tf.reduce_sum(loss) / tf.reduce_sum(mask)
    return loss

def masked_acc(labels, preds):
    mask = tf.cast(labels != 0, tf.float32)
    preds = tf.argmax(preds, axis=-1)
    labels = tf.cast(labels, tf.int64)
    match = tf.cast(preds == labels, mask.dtype)
    acc = tf.reduce_sum(match * mask) / tf.reduce_sum(mask)
    return acc

def plot_history(history, quota):
    if quota == 'acc':
        plt.plot(history['masked_acc'], label='accuracy', marker='*')
        plt.plot(history['val_masked_acc'], label='val_accuracy', marker='x')
    else:
        plt.plot(history['loss'], label='loss', marker='*')
        plt.plot(history['val_loss'], label='val_loss', marker='x')
    plt.ylim([0, max(plt.ylim())])
    plt.xlabel('Epoch #')
    plt.ylabel('CE/token')
    plt.legend()
    plt.show()

def plot_attention_maps(image, str_tokens, attention_map):
    fig = plt.figure(figsize=(16, 9))
    len_result = len(str_tokens)
    titles = []
    for i in range(len_result):
        map = attention_map[i]
        grid_size = max(int(np.ceil(len_result / 2)), 2)
        ax = fig.add_subplot(3, grid_size, i + 1)
        titles.append(ax.set_title(str_tokens[i]))
        img = ax.imshow(image)
        ax.imshow(map, cmap='gray', alpha=0.75, extent=img.get_extent(), clim=[0.0, np.max(map)])
    plt.tight_layout()

class GenerateText(keras.callbacks.Callback):
    def __init__(self, model, path, root_dir='./data/'):
        super().__init__()
        self.image = load_image(path, root_dir)
        self.model = model

    def on_epoch_end(self, epoch=None, logs=None):
        print('\n')
        for t in (0.0, 0.5, 1.0):
            result = self.model.simple_gen(self.image, temperature=t)
            print(result)
        print()