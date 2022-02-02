import logging

logging.basicConfig(level=logging.INFO)

from os.path import join
import pandas as pd
import numpy as np
from transformers import TFAutoModel, AutoTokenizer
import tensorflow as tf
from sklearn.utils import shuffle

from loss import distill_loss_T

##
## for training cv
##

def get_X_y(PATH_DATA, name_data, categories):
    """
    Objective get the data (features, targets)

    Inputs:
        - PATH_DATA, str: the path to the data
        - name_data, str: the name of the file
        - categories, list: list of the categories
    Outputs:
        - X, np.array: the features
        - y, np.array: the targets
    """

    df = pd.read_csv(join(PATH_DATA, '{}.csv'.format(name_data)), engine='python')

    X = df.loc[:, 'text_pp'].values
    y = df.loc[:, categories].values

    logging.info('Training data loaded')

    return X, y


def load_transformer_models(bert):
    """
    Objective: load the tokenizer we'll use and also the transfomer model
    
    Inputs:
        - bert, str: the name of models look at https://huggingface.co/models for all models
    Outputs:
        - tokenizer, transformers.tokenization_distilbert.DistilBertTokenizer: the tokenizer of the model
        - transformer_model, transformers.modeling_tf_distilbert.TFDistilBertModel: the transformer model that
                                                                                    we will use as base
                                                                                    (embedding model)
    """
    tokenizer = AutoTokenizer.from_pretrained(bert)

    transformer_model = TFAutoModel.from_pretrained(bert)

    return tokenizer, transformer_model


def train_model(X_train, y_train, tokenizer,
             transformer_model, categories, rate=0.5, lr=2e-05, epochs=1000,
              batch_size=16, max_length=64, 
             gamma=2, alpha=0.25):

    """
    Objective: create architecture and train the model

    Inputs:
        - X_train, np.array: the texts preprocessed
        - y_train, np.array: the targets
        - tokenizer, transformers.tokenization_distilbert.DistilBertTokenizer: the tokenizer of the model
        - transformer_model, transformers.modeling_tf_distilbert.TFDistilBertModel: the transformer model that
                                                                                    we will use as base
                                                                                    (embedding model)
        - categories, list: the list of categories
        - rate, float: the dropout rate for the model training between 0 and 1
        - lr, float: the learning rate for training
        - epochs, int: the epoch number for training
        - batch_size, int: the batch size during training
        - max_length, int: the max lenght of the encoded vector for each text
        - gamma, float: parameter for focall loss
        - alpha, float: parameter for focall loss
    Outputs:
        - model, tf.keras.Model : model trained 
    """

    steps_per_epoch = int(len(X_train) / batch_size)

    batches = get_batches(X_train, y_train, tokenizer, batch_size, max_length)

    logging.info('Data batches generated')

    model = get_model(max_length, transformer_model, len(categories), rate, name_model=False)

    logging.info('Model loaded')

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss=tf.keras.losses.BinaryCrossentropy(), 
                  metrics=[tf.keras.metrics.Recall(), tf.keras.metrics.Precision()])

    model.fit(batches, epochs=epochs, steps_per_epoch=steps_per_epoch)

    return model


def train_distilled_model(X_train, y_train, tokenizer,
             transformer_model, categories, rate=0.5, lr=2e-05, epochs=1000,
              batch_size=16, max_length=64, alpha=0, T=5):

    """
    Objective: create architecture and train the model

    Inputs:
        - X_train, np.array: the texts preprocessed
        - y_train, np.array: the targets
        - tokenizer, transformers.tokenization_distilbert.DistilBertTokenizer: the tokenizer of the model
        - transformer_model, transformers.modeling_tf_distilbert.TFDistilBertModel: the transformer model that
                                                                                    we will use as base
                                                                                    (embedding model)
        - categories, list: the list of categories
        - rate, float: the dropout rate for the model training between 0 and 1
        - lr, float: the learning rate for training
        - epochs, int: the epoch number for training
        - batch_size, int: the batch size during training
        - max_length, int: the max lenght of the encoded vector for each text
        - alpha, float: parameter for distillation loss
        - T, int: the temperature for distillation loss
    Outputs:
        - model, tf.keras.Model : model trained 
    """

    steps_per_epoch = int(len(X_train) / batch_size)

    batches = get_batches(X_train, y_train, tokenizer, batch_size, max_length)

    logging.info('Data batches generated')

    model = get_model(max_length, transformer_model, len(categories), rate, name_model=False)

    logging.info('Model loaded')

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss=distill_loss_T(alpha=alpha, T=T))

    model.fit(batches, epochs=epochs, steps_per_epoch=steps_per_epoch)

    return model



def get_model(max_length, transformer_model, num_labels, rate, name_model=False):
    """
    Get a model from scratch or if we have weights load it to the model.

    Inputs:
        - max_length, int: the input shape of the data
        - transformer_model, transformers.modeling_tf_distilbert.TFDistilBertModel: the transformer model that
                                                                                    we will use as base
                                                                                    (embedding model - sentence here)
        - num_labels, int: the number of final outputs
        - name_model (optional), str: look for an already existing model in with the PATH MODELS
        - i, int: if ensembling model otherwise 0 or None
    Outputs:
        - model, tensorflow.python.keras.engine.functional.Functional: the final model we'll train 
    """

    logging.info('Creating architecture...')
    
    input_ids_in = tf.keras.layers.Input(shape=(max_length,), name='input_token', dtype='int32')
    input_masks_in = tf.keras.layers.Input(shape=(max_length,), name='masked_token', dtype='int32') 
    

    embedding_layer = transformer_model(input_ids_in, attention_mask=input_masks_in)[0][:,0,:]
    
    output_layer = tf.keras.layers.Dropout(rate, name='do_layer')(embedding_layer)

    # Define weight initializer with a random seed to ensure reproducibility
    weight_initializer = tf.keras.initializers.GlorotNormal(seed=42)

    for unit_size in [256]:
        output_layer = tf.keras.layers.Dense(unit_size, 
                                             activation='relu',
                                             kernel_initializer=weight_initializer,  
                                             kernel_constraint=None,
                                             bias_initializer='zeros',
                                             name=n'dense_mid'
                                            )(output_layer)

        output_layer = tf.keras.layers.Dropout(rate, name=d'do_mid')(output_layer)
    
    output = tf.keras.layers.Dense(num_labels, activation='sigmoid', name='last_layer')(output_layer)

    model = tf.keras.Model(inputs=[input_ids_in, input_masks_in], outputs = output)

    if name_model:
        try:
            model.load_weights(name_model)
            logging.info('Model {} restored'.format(name_model))
        except:
            logging.warning('Model {} not found'.format(name_model))
            logging.warning('If training: new model from scratch')
            logging.warning('If classifying: the configuration does not fit the architecture and this model is not trained yet!')

    return model



def get_batches(X_train, y_train, tokenizer, batch_size, max_length):
    """
    Objective: from features and labels yield a random batch of batch_size of (features, labels),
               each time we reached all data we shuffle again the (features, labels) 
               and we do it again (infinite loop)

    Inputs:
        - X_train, np.array: the texts (features)
        - y_train, np.array: the labels
        - tokenizer, transformers.tokenization_distilbert.DistilBertTokenizer: the tokenizer of the model
        - batch_size, int: the size of the batch we yield
        - max_length, int: the input shape of the data
    Outputs: (generator)
        - inputs, np.array : two arrays one with ids from the tokenizer, and the masks associated with the padding
        - targets, np.array: the label array of the associated inputs
    """
    X_train, y_train = shuffle(X_train, y_train, random_state=11)

    i, j = 0, 0

    while i > -1:

        if (len(X_train) - j*batch_size) < batch_size:
            j = 0
            X_train, y_train = shuffle(X_train, y_train, random_state=11)

        sentences = X_train[j*batch_size: (j+1) * batch_size]
        targets = y_train[j*batch_size: (j+1) * batch_size, :]
        j += 1

        input_ids, input_masks = [],[]

        # see if puting following before the loop may improve the training in time and RAM used
        inputs = tokenizer.batch_encode_plus(list(sentences), add_special_tokens=True, max_length=max_length, 
                                            padding='max_length',  return_attention_mask=True,
                                            return_token_type_ids=True, truncation=True)

        ids = np.asarray(inputs['input_ids'], dtype='int32')
        masks = np.asarray(inputs['attention_mask'], dtype='int32')

        #till here and use the same shuffle on ids, masks instead of X_train

        inputs = [ids, masks]

        yield inputs, targets