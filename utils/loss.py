import tensorflow as tf
import numpy as np

def distill_loss_T(y_true, y_pred, alpha=0.5, T=1):
    """
    Objective: construct the distillation loss with T the temperature and alpha to weight the sum between bce and
                mse: alpha = 1 bce only, alpha = 0 mse only
    
    Source: https://stackoverflow.com/questions/45961428/make-a-custom-loss-function-in-keras
    Inputs:
        - y_true, np.array: the true values
        - y_pred, np.array: the preds values
        - alpha, float: shoul be between 0 and 1 to say what is the most important loss
        - T, int: the temperature for the MSE
    Outputs:
        - loss, float: the calculated loss
    """
    
    y_pred = np.array(y_pred) if type(y_pred) == list else y_pred
    y_true = np.array(y_true) if type(y_true) == list else y_true
    
    
    bce = tf.keras.losses.BinaryCrossentropy()
    mse = tf.keras.losses.MeanSquaredError()
    l1 = tf.cast(bce(tf.cast(tf.keras.activations.sigmoid(y_true) > 0.5, dtype='int32'),
                     tf.keras.activations.sigmoid(y_pred)), dtype='float32')
    l2 = mse(tf.keras.activations.sigmoid(tf.cast(y_true / T, dtype='float32')),
             tf.keras.activations.sigmoid(tf.cast(y_pred / T, dtype='float32')))
    
    if alpha == 0:
        loss =  l2
    elif alpha == 1:
        loss = l1
    else:
        loss = alpha * l1 + (1 - alpha) * l2 * T ** 2 # * T ** 2 comes from here https://export.arxiv.org/pdf/1503.02531
        # "Since the magnitudes of the gradients produced by the soft targets scale as 1/T**2
        # it is important to multiply them by T**2 when using both hard and soft targets"

    return loss



def distill_loss(alpha, T):
    def distill(y_true, y_pred):
        return distill_loss_T(y_true, y_pred, alpha, T)
    return distill


def recall_distill(y_true, y_pred, recall_metrics):

    y_pred = np.array(y_pred) if type(y_pred) == list else y_pred
    y_true = np.array(y_true) if type(y_true) == list else y_true

    y_true = tf.cast(tf.keras.activations.sigmoid(y_true) > 0.5, dtype='int8')
    y_pred = tf.cast(tf.keras.activations.sigmoid(y_pred) > 0.5, dtype='int8')

    recall_metrics.update_state(y_true, y_pred)
    
    return recall_metrics.result()


def distill_recall():
    recall_metrics = tf.keras.metrics.Recall()
    @tf.function
    def recall(y_true, y_pred):
        return recall_distill(y_true, y_pred, recall_metrics)
    return recall



def precision_distill(y_true, y_pred, precision_metrics):

    y_pred = np.array(y_pred) if type(y_pred) == list else y_pred
    y_true = np.array(y_true) if type(y_true) == list else y_true

    y_true = tf.cast(tf.keras.activations.sigmoid(y_true) > 0.5, dtype='int8')
    y_pred = tf.cast(tf.keras.activations.sigmoid(y_pred) > 0.5, dtype='int8')

    precision_metrics.update_state(y_true, y_pred)
    
    return precision_metrics.result()

def distill_precision():
    precision_metrics = tf.keras.metrics.Precision()
    @tf.function
    def precision(y_true, y_pred):
        return precision_distill(y_true, y_pred, precision_metrics)
    return precision