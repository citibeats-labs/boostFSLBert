# How to boost BERT performances

## Executive summary

In [our blog post]() we have explored a way to boost BERT performances when we deal with a small dataset for a classification problem. We used a [hatred speech dataset](https://www.kaggle.com/rahulgoel1106/hatred-on-twitter-during-metoo-movement) example that is present on [Kaggle](www.kaggle.com).

You can take a look at how we constructed the dataset and get final results in the [notebooks folder](notebooks/). We also provide the colab to do the experiment on your own. 

## Results of the different experiments

### Training and testing set

|Dataset| #Texts | #Hatred |
|:-----:|:------:|:-------:|
|Train  |  100   |   50    |
| Test  | 10,000 |  5,000  |


### Metrics

|Models| F1 | P | R |
|:-----:|:--:|:-:|:-:|
|Random|50.73(0.00)|50.44(0.00)|51.02(0.00)|
|bow|56.40(0.00)|63.07(0.00)|51.00(0.00)|
|FSL distilBERT|64.41(4.65)|65.34(2.87)|64.80(10.39)|
|FSL_mean distilBERT|67.86(0.00)|67.75(0.00)|67.98(0.00)|
|distill distilBERT experts|68.34(0.00)|67.58(0.00)|69.12(0.00)|
|FSL XtremeDistilTransformers|63.05(1.40)|63.70(1.69)|62.46(4.00)|
|FSL_mean XtremeDistilTransformers|63.87(0.00)|64.56(0.00)|63.20(0.00)|
|distill XtremeDistilTransformers experts|64.15(0.00)|64.52(0.00)|63.78(0.00)|


Finally, we provide some command lines to apply the code straight from this repo.

## Pre-processing

To pre-process one text you should just apply copy paste the following CLI:

```bash
python run_pre_processing.py --name test_set
```

Of course this means you have a `text` variable in your `data/my_dataset.csv`  otherwise you should add variables `--text_var my_text_variable`.

You'll find a `data/my_dataset_pp.csv` in the data folder ready for training !


## Training a BERT model

Now you can try to train the model with a transformer architecture. You need to have the dataset already processed as the `data/my_dataset_pp.csv`. To do so you should copy paste the following CLI:

```bash
python run_training.py --name my_dataset_pp
```

You can take a look at all different parameters, but normally the default hyperparameters are already set values used in the blog post. You will have a `models/my_dataset_pp.h5` model trained and saved that you can use for inference. 


## Infer from a BERT model

Now you can use your `models/my_dataset_pp.h5` for inferences. If you have a `data/inference_set.csv` with a `text` variable in it, you can apply your model with the following CLI:

```bash
python run_inferences.py --name_data inference_set --name_model my_dataset_pp
```

To be sure that the `models/my_dataset_pp.h5` has been loaded you should see a:
`INFO:root:Model /YOUR_PATH_TO_THE_MODEL/my_dataset_pp.h5 restored` 
Otherwise you'll see:
`WARNING:root:Model /YOUR_PATH_TO_THE_MODEL/my_dataset_pp.h5 not found` 
`WARNING:root:If training: new model from scratch`
`WARNING:root:If classifying: the configuration does not fit the architecture and this model is not trained yet!`

And you're inferences will be from a random model !

Then you'll have a `data/inference_set_preds.csv` file with all the predicted data.


## Distil a model from experts inferences

Now let's say you have created the augmented dataset, `data/random_set_with_inferences_pp.csv` which is already pre-processed and have a variable `prob_hatred` in this dataset as the mean of all you experts predictions, you can train a model from it with the following CLI:

```bash
python run_training_distilled_BERT.py --name random_set_with_inferences_pp
```

You can take a look at all different parameters, but normally the default hyperparameters are already set values used in the blog post. You will have a `models/random_set_with_inferences_pp_distilled.h5` model trained and saved that you can use for inference. 
