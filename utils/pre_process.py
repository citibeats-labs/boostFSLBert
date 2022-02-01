import re

def pre_process(data, text_var='text'):
    """
    Objective: pre-process the data to get rid of hashtags, mentions, usrs of the text_var. it remove numbers, unidecode emojis
                duplicated tokens or punctuations, 

    Inputs:
        - data, pd.DataFrame: dataset to pre-process
        - text_var, str: the variable of text in the dataframe
    Outputs:
        - data, pd.DataFrame: dataset pre-processed
    """
    #quit url
    data.loc[:, 'text_pp'] = data.loc[:, text_var].apply(lambda x: 
                re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', 'URL', x))

    #quit HT
    data.loc[:, 'text_pp'] = data.loc[:, 'text_pp'].apply(lambda x:  re.sub(r'(#\w+)', 'HTG', x))

    #quit mentiosn
    data.loc[:, 'text_pp'] = data.loc[:, 'text_pp'].apply(lambda x:  re.sub(r'(@\w+)', 'USR', x))

    #quit unicode
    data.loc[:, 'text_pp'] = data.loc[:, 'text_pp'].apply(lambda x: removeUnicode(x).replace('\'', '’'))
    data.loc[:, 'text_pp'] = data.loc[:, 'text_pp'].apply(lambda x: 
                                                x.replace('\x92', '’').replace('ŕ', 'à').replace('č', 'è'))
    #quit contractions in english only
    data.loc[:, 'text_pp'] = data.loc[:, 'text_pp'].apply(lambda x: get_rid_contraction(x.lower(), 
                                                                            contractions).replace('  ', ' '))


    #remove emojis
    data.loc[:, 'text_pp'] = data.loc[:, 'text_pp'].apply(lambda x: get_emojis(x, token='')).values

    #remove duplicated tokens
    data.loc[:, 'text_pp'] = data.loc[:, 'text_pp'].apply(lambda x: re.sub(r'\b(\w+)(?:\W+\1\b)+', r'\1',
                                                                       x, flags=re.IGNORECASE))
    #remove htg with - or _ between it
    data.loc[:, 'text_pp'] = data.loc[:, 'text_pp'].apply(lambda x: re.sub(r'htg[-_]\w+', r'htg',
                                                                       x, flags=re.IGNORECASE))

    # quit mutli punctuation
    data.loc[:, 'text_pp'] = data.loc[:, 'text_pp'].apply(lambda x: replaceMultiExclamationMark(x).strip())
    data.loc[:, 'text_pp'] = data.loc[:, 'text_pp'].apply(lambda x: replaceMultiQuestionMark(x).strip())
    data.loc[:, 'text_pp'] = data.loc[:, 'text_pp'].apply(lambda x: replaceMultiStopMark(x).strip())

    #remove numbers
    data.loc[:, 'text_pp'] = data.loc[:, 'text_pp'].apply(lambda x: removeNumbers(x).replace('\\n', ' ').strip())
    data.loc[:, 'text_pp'] = data.loc[:, 'text_pp'].apply(lambda x: re.sub(
                        r'num[,.]num', 'num', 
                        x.replace('  ', ' ').strip()).strip())

    #remove weird unicode
    data.loc[:, 'text_pp'] = data.loc[:, 'text_pp'].apply(lambda x: 
                                                          re.sub(r'(?:\\xe|\\xa)', '', x).replace('  ', ' ').strip())

    #remove sequences url htg usr in random orders
    data.loc[:, 'text_pp'] = data.loc[:, 'text_pp'].apply(lambda x: 
                                                          re.sub(r'^((url\s*)*(?:usr|htg)\b[\s\n]*(url\s*)*){1,}',
                                                                           '', x).strip())
    data.loc[:, 'text_pp'] = data.loc[:, 'text_pp'].apply(lambda x: 
                                                          re.sub(r'((url\s*)*(?:usr|htg)\b[\s\n]*(url\s*)*){1,}$',
                                                                           '', x).strip())

    data.loc[:, 'text_pp'] = data.loc[:, 'text_pp'].apply(lambda x: re.sub(
                        r'^((?:num|url)*(?:usr|htg)\b[\s\n]*(?:num|url)*){1,}', '', 
                        x.replace('  ', ' ').strip()).strip())

    data.loc[:, 'text_pp'] = data.loc[:, 'text_pp'].apply(lambda x: re.sub(
                        r'url$', '', 
                        x.replace('  ', ' ').strip()).strip())    

    return data

def removeUnicode(text):
    """ Removes unicode strings like "\u002c" printable such as \t \r and \n"""
    text = re.sub(r'(\\u[0-9A-Fa-f]+)',r'', text)       
    text = re.sub(r'&amp;','&', text.replace('\t', ' ').replace('\r', ' ').replace('\n', ' ')).strip()
    return text


def get_rid_contraction(text, contractions):
    """
    Objective: get rid of contractions by puting it in normal words from a dictionary

    Inputs:
        - text, str: the text to pre-process
        - contractions, dict: the dict to replace contractions
    Outputs:
        - text, str: the text without contractions
    """
    for key, value in contractions.items():
        if key in text or key.replace("’", "'") in text:
            text = re.sub(key, value, text)
            text = re.sub(key.replace("’", "'"), value, text)
        
        if key.capitalize() in text or key.capitalize().replace("’", "'") in text:
            text = re.sub(key.capitalize(), value, text)
            text = re.sub(key.capitalize().replace("’", "'"), value, text)
            
    return text


def get_emojis(text, token=' EMOJI '):
    """
    Objectives: get all emojis from the text an replace it by a token
    
    Inputs:
        - text, str: the text to clean
        - token, str: the token to replace emoji with
    Outputs:
        - text, str: the cleaned text
        - emojis, list: the list of emojis
    """
    EMOJI_PATTERN = re.compile(
                            "["
                            "\U0001F1E0-\U0001F1FF"  # flags (iOS)
                            "\U0001F300-\U0001F5FF"  # symbols & pictographs
                            "\U0001F600-\U0001F64F"  # emoticons
                            "\U0001F680-\U0001F6FF"  # transport & map symbols
                            "\U0001F700-\U0001F77F"  # alchemical symbols
                            "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
                            "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
                            "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
                            "\U0001FA00-\U0001FA6F"  # Chess Symbols
                            "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
                            "\U00002702-\U000027B0"  # Dingbats
                            "\U000024C2-\U0001F251" 
                            "]+"
                        )
    
    emojis = re.findall(EMOJI_PATTERN, text) + re.findall(r'\\u\w+', text.encode('unicode_escape').decode('ascii'))
    text = re.sub(EMOJI_PATTERN, token, text)
    text = re.sub(r'\\[uU]\w+', '', text.encode('unicode_escape').decode('ascii'))
    
    return text


def replaceMultiExclamationMark(text):
    """ Replaces repetitions of exlamation marks """
    text = re.sub(r"(\!)\1+", r'!', text)
    return text

def replaceMultiQuestionMark(text):
    """ Replaces repetitions of question marks """
    text = re.sub(r"(\?)\1+", r'?', text)
    return text

def replaceMultiStopMark(text):
    """ Replaces repetitions of stop marks """
    text = re.sub(r"(\.)\1+", r'.', text)
    return text

def removeNumbers(text):
    """
    Objective: pre-process the numbers within the text

    Inputs:
        - text, str:
    Outputs:
        - text, str: pre-processed texts
        - len(dates), int: the number of dates detected
        - len(hours), int: the number of hours detected
        - len(years), int: the number of years detected
        - len(prices), int: the number of prices detected
        - len(numbers), int: the number of numbers detected
        - len(durations), int: the number of durations detected
    """

    text = re.sub(r'\d+', 'num', text)
    
    return text



contractions = { 
"ain’t": "are not",
"aren’t": "are not",
"can’t": "cannot",
"can’t’ve": "cannot have",
"’cause": "because",
"could’ve": "could have",
"couldn’t": "could not",
"couldn’t’ve": "could not have",
"didn’t": "did not",
"doesn’t": "does not",
"don’t": "do not",
"hadn’t": "had not",
"hadn’t've": "had not have",
"hasn’t": "has not",
"haven’t": "have not",
"he’d": "he would",
"he’d’ve": "he would have",
"he’ll": "he will",
"he’ll’ve": "he will have",
"he’s": " he is",
"how’d": "how did",
"how’d’y": "how do you",
"how’ll": "how will",
"how’s": "how is",
"i’d": "i would",
"i’d’ve": "i would have",
"i’ll": "i will",
"i’ll’ve": "i will have",
"i’m": "i am",
"i’ve": "i have",
"isn’t": "is not",
"it’d": "it would",
"it’d’ve": "it would have",
"it’ll": "it will",
"it’ll’ve": "it will have",
"it’s": "it is",
"let’s": "let us",
"ma’am": "madam",
"mayn’t": "may not",
"might’ve": "might have",
"mightn’t": "might not",
"mightn’t’ve": "might not have",
"must’ve": "must have",
"mustn’t": "must not",
"mustn’t’ve": "must not have",
"needn’t": "need not",
"needn’t’ve": "need not have",
"o’clock": "of the clock",
"oughtn’t": "ought not",
"oughtn’t’ve": "ought not have",
"shan’t": "shall not",
"sha’n’t": "shall not",
"shan’t’ve": "shall not have",
"she’d": "she would",
"she’d’ve": "she would have",
"she’ll": "she will",
"she’ll’ve": "she will have",
"she’s": "she is",
"should’ve": "should have",
"shouldn’t": "should not",
"shouldn’t’ve": "should not have",
"so’ve": "so have",
"so’s": "so is",
"that’d": "that had",
"that’d’ve": "that would have",
"that’s": "that is",
"there’d": "there would",
"there’d’ve": "there would have",
"there’s": "there is",
"they’d": "they would",
"they’d’ve": "they would have",
"they’ll": "they will",
"they’ll’ve": "they will have",
"they’re": "they are",
"they’ve": "they have",
"to’ve": "to have",
"wasn’t": "was not",
"we’d": "we would",
"we’d’ve": "we would have",
"we’ll": "we will",
"we’ll’ve": "we will have",
"we’re": "we are",
"we’ve": "we have",
"weren’t": "were not",
"what’ll": "what will",
"what’ll’ve": "what will have",
"what’re": "what are",
"what’s": "what is",
"what’ve": "what have",
"when’s": "when is",
"when’ve": "when have",
"where’d": "where did",
"where’s": "where is",
"where’ve": "where have",
"who’ll": "ho will",
"who’ll’ve": "who will have",
"who’s": "who is",
"who’ve": "who have",
"why’s": "why is",
"why’ve": "why have",
"will’ve": "will have",
"won’t": "will not",
"won’t’ve": "will not have",
"would’ve": "would have",
"wouldn’t": "would not",
"wouldn’t’ve": "would not have",
"y’all": "you all",
"y’all’d": "you all would",
"y’all’d've": "you all would have",
"y’all’re": "you all are",
"y’all’ve": "you all have",
"you’d": "you would",
"you’d've": "you would have",
"you’ll": "you will",
"you’ll've": "you will have",
"you’re": "you are",
"you’ve": "you have"
}