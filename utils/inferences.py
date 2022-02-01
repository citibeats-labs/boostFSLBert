import numpy as np


def get_inputs(tokenizer, sentences, max_length, n=None):
    """
    Objective: tokenize the sentences to get the inputs
    
    Inputs:
        - tokenizer, transformers.tokenization_distilbert.DistilBertTokenizer: the tokenizer of the model
        - sentences, np.array: the sentences pre-processed to classify the intents
        - max_length, int: the maximum number of tokens
        - n, int: the number of inputs if duplicated inputs for ensemble modeling
    Outputs:
        - inputs, list: list of ids and masks from the tokenizer
    """
    inputs = tokenizer.batch_encode_plus(list(sentences), add_special_tokens=True, max_length=max_length, 
                                    padding='max_length',  return_attention_mask=True,
                                    return_token_type_ids=True, truncation=True)

    ids = np.asarray(inputs['input_ids'], dtype='int32')
    masks = np.asarray(inputs['attention_mask'], dtype='int32')

    inputs = [ids, masks] if not n else [ids, masks] * n
    
    return inputs