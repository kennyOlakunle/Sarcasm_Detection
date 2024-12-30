
#ignore warnings
import warnings
warnings.filterwarnings('ignore')


import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import re
import itertools    
import wordcloud


# For data preprocessing
from sklearn.model_selection import train_test_split, RandomizedSearchCV # type: ignore
from sklearn.preprocessing import LabelEncoder # type: ignore
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import WordNetLemmatizer

# For building our Models
import tensorflow as tf
from tensorflow.keras.layers import Input,Dense,LSTM,Embedding,Conv1D,Bidirectional,SpatialDropout1D,Dropout
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import cross_val_score, KFold, cross_val_predict, cross_validate

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.models import Model

# For Lazy Predict
# from lazypredict.Supervised import LazyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier # type: ignore
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


# For hyperparameter tuning
from scipy.stats import uniform

#Reduce dimensions to 2 for faster training
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


# For creating vocabulary dictionary
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# For model evaluation
from sklearn.model_selection import LearningCurveDisplay, learning_curve
from sklearn.metrics import confusion_matrix,classification_report, log_loss, make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, auc, DetCurveDisplay, RocCurveDisplay, roc_curve, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV


# For processing texts
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

print(tf.__version__)



data = pd.read_csv('dataset/sarcasm_headlines.csv')

# print(df.head())

#Duplicated Values

def check_duplicates(data):
    duplicate = data.duplicated().sum()
    return duplicate

# Applying check_duplicates function
print(check_duplicates(data))

#show the special characters in the text column

data[data['text'].str.contains(r'[^A-Za-z0-9 ]',regex=True)]

# Function to remove special characters
def remove_special_characters(text):
    text = re.sub(r'[^A-Za-z0-9 ]', '', text)
    return text

# Apply function to remove special characters
data['text'] = data['text'].apply(remove_special_characters)


#Check if there is any special characters in the text column

def special_characters(data):
    special = data.str.contains(r'[^A-Za-z0-9 ]',regex=True).sum()
    return special

# Applying special_characters function
special_characters(data['text'])

nltk.download('stopwords')

nltk.download ('wordnet')


#Removing other Noises

def remove_URL(text):
    return re.sub(r"https?://\S+|www\.\S+", "", text)

def remove_html(text):
    html = re.compile(r"<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});")
    return re.sub(html, "", text)

def remove_non_ascii(text):
    return re.sub(r'[^\x00-\x7f]',r'', text)

def remove_punct(text):
    return re.sub(r'[]!"$%&\'()*+,./:;=#@?[\\^_`{|}~-]+', "", text)

def other_clean(text):
        """
            Other manual text cleaning techniques
        """
        # Typos, slang and other 
        sample_typos_slang = {
                                "w/e": "whatever",
                                "usagov": "usa government",
                                "recentlu": "recently",
                                "ph0tos": "photos",
                                "amirite": "am i right",
                                "exp0sed": "exposed",
                                "<3": "love",
                                "luv": "love",
                                "amageddon": "armageddon",
                                "trfc": "traffic",
                                "16yr": "16 year"
                                }

        # Acronyms
        sample_acronyms =  {
                            "mh370": "malaysia airlines flight 370",
                            "okwx": "oklahoma city weather",
                            "arwx": "arkansas weather",
                            "gawx": "georgia weather",
                            "scwx": "south carolina weather",
                            "cawx": "california weather",
                            "tnwx": "tennessee weather",
                            "azwx": "arizona weather",
                            "alwx": "alabama weather",
                            "usnwsgov": "united states national weather service",
                            "2mw": "tomorrow"
                            }


        # Some common abbreviations
        sample_abbr = {
                        "$" : " dollar ",
                        "€" : " euro ",
                        "4ao" : "for adults only",
                        "a.m" : "before midday",
                        "a3" : "anytime anywhere anyplace",
                        "aamof" : "as a matter of fact",
                        "acct" : "account",
                        "adih" : "another day in hell",
                        "afaic" : "as far as i am concerned",
                        "afaict" : "as far as i can tell",
                        "afaik" : "as far as i know",
                        "afair" : "as far as i remember",
                        "afk" : "away from keyboard",
                        "app" : "application",
                        "approx" : "approximately",
                        "apps" : "applications",
                        "asap" : "as soon as possible",
                        "asl" : "age, sex, location",
                        "atk" : "at the keyboard",
                        "ave." : "avenue",
                        "aymm" : "are you my mother",
                        "ayor" : "at your own risk",
                        "b&b" : "bed and breakfast",
                        "b+b" : "bed and breakfast",
                        "b.c" : "before christ",
                        "b2b" : "business to business",
                        "b2c" : "business to customer",
                        "b4" : "before",
                        "b4n" : "bye for now",
                        "b@u" : "back at you",
                        "bae" : "before anyone else",
                        "bak" : "back at keyboard",
                        "bbbg" : "bye bye be good",
                        "bbc" : "british broadcasting corporation",
                        "bbias" : "be back in a second",
                        "bbl" : "be back later",
                        "bbs" : "be back soon",
                        "be4" : "before",
                        "bfn" : "bye for now",
                        "blvd" : "boulevard",
                        "bout" : "about",
                        "brb" : "be right back",
                        "bros" : "brothers",
                        "brt" : "be right there",
                        "bsaaw" : "big smile and a wink",
                        "btw" : "by the way",
                        "bwl" : "bursting with laughter",
                        "c/o" : "care of",
                        "cet" : "central european time",
                        "cf" : "compare",
                        "cia" : "central intelligence agency",
                        "csl" : "can not stop laughing",
                        "cu" : "see you",
                        "cul8r" : "see you later",
                        "cv" : "curriculum vitae",
                        "cwot" : "complete waste of time",
                        "cya" : "see you",
                        "cyt" : "see you tomorrow",
                        "dae" : "does anyone else",
                        "dbmib" : "do not bother me i am busy",
                        "diy" : "do it yourself",
                        "dm" : "direct message",
                        "dwh" : "during work hours",
                        "e123" : "easy as one two three",
                        "eet" : "eastern european time",
                        "eg" : "example",
                        "embm" : "early morning business meeting",
                        "encl" : "enclosed",
                        "encl." : "enclosed",
                        "etc" : "and so on",
                        "faq" : "frequently asked questions",
                        "fawc" : "for anyone who cares",
                        "fb" : "facebook",
                        "fc" : "fingers crossed",
                        "fig" : "figure",
                        "fimh" : "forever in my heart",
                        "ft." : "feet",
                        "ft" : "featuring",
                        "ftl" : "for the loss",
                        "ftw" : "for the win",
                        "fwiw" : "for what it is worth",
                        "fyi" : "for your information",
                        "g9" : "genius",
                        "gahoy" : "get a hold of yourself",
                        "gal" : "get a life",
                        "gcse" : "general certificate of secondary education",
                        "gfn" : "gone for now",
                        "gg" : "good game",
                        "gl" : "good luck",
                        "glhf" : "good luck have fun",
                        "gmt" : "greenwich mean time",
                        "gmta" : "great minds think alike",
                        "gn" : "good night",
                        "g.o.a.t" : "greatest of all time",
                        "goat" : "greatest of all time",
                        "goi" : "get over it",
                        "gps" : "global positioning system",
                        "gr8" : "great",
                        "gratz" : "congratulations",
                        "gyal" : "girl",
                        "h&c" : "hot and cold",
                        "hp" : "horsepower",
                        "hr" : "hour",
                        "hrh" : "his royal highness",
                        "ht" : "height",
                        "ibrb" : "i will be right back",
                        "ic" : "i see",
                        "icq" : "i seek you",
                        "icymi" : "in case you missed it",
                        "idc" : "i do not care",
                        "idgadf" : "i do not give a damn fuck",
                        "idgaf" : "i do not give a fuck",
                        "idk" : "i do not know",
                        "ie" : "that is",
                        "i.e" : "that is",
                        "ifyp" : "i feel your pain",
                        "IG" : "instagram",
                        "iirc" : "if i remember correctly",
                        "ilu" : "i love you",
                        "ily" : "i love you",
                        "imho" : "in my humble opinion",
                        "imo" : "in my opinion",
                        "imu" : "i miss you",
                        "iow" : "in other words",
                        "irl" : "in real life",
                        "j4f" : "just for fun",
                        "jic" : "just in case",
                        "jk" : "just kidding",
                        "jsyk" : "just so you know",
                        "l8r" : "later",
                        "lb" : "pound",
                        "lbs" : "pounds",
                        "ldr" : "long distance relationship",
                        "lmao" : "laugh my ass off",
                        "lmfao" : "laugh my fucking ass off",
                        "lol" : "laughing out loud",
                        "ltd" : "limited",
                        "ltns" : "long time no see",
                        "m8" : "mate",
                        "mf" : "motherfucker",
                        "mfs" : "motherfuckers",
                        "mfw" : "my face when",
                        "mofo" : "motherfucker",
                        "mph" : "miles per hour",
                        "mr" : "mister",
                        "mrw" : "my reaction when",
                        "ms" : "miss",
                        "mte" : "my thoughts exactly",
                        "nagi" : "not a good idea",
                        "nbc" : "national broadcasting company",
                        "nbd" : "not big deal",
                        "nfs" : "not for sale",
                        "ngl" : "not going to lie",
                        "nhs" : "national health service",
                        "nrn" : "no reply necessary",
                        "nsfl" : "not safe for life",
                        "nsfw" : "not safe for work",
                        "nth" : "nice to have",
                        "nvr" : "never",
                        "nyc" : "new york city",
                        "oc" : "original content",
                        "og" : "original",
                        "ohp" : "overhead projector",
                        "oic" : "oh i see",
                        "omdb" : "over my dead body",
                        "omg" : "oh my god",
                        "omw" : "on my way",
                        "p.a" : "per annum",
                        "p.m" : "after midday",
                        "pm" : "prime minister",
                        "poc" : "people of color",
                        "pov" : "point of view",
                        "pp" : "pages",
                        "ppl" : "people",
                        "prw" : "parents are watching",
                        "ps" : "postscript",
                        "pt" : "point",
                        "ptb" : "please text back",
                        "pto" : "please turn over",
                        "qpsa" : "what happens", #"que pasa",
                        "ratchet" : "rude",
                        "rbtl" : "read between the lines",
                        "rlrt" : "real life retweet",
                        "rofl" : "rolling on the floor laughing",
                        "roflol" : "rolling on the floor laughing out loud",
                        "rotflmao" : "rolling on the floor laughing my ass off",
                        "rt" : "retweet",
                        "ruok" : "are you ok",
                        "sfw" : "safe for work",
                        "sk8" : "skate",
                        "smh" : "shake my head",
                        "sq" : "square",
                        "srsly" : "seriously",
                        "ssdd" : "same stuff different day",
                        "tbh" : "to be honest",
                        "tbs" : "tablespooful",
                        "tbsp" : "tablespooful",
                        "tfw" : "that feeling when",
                        "thks" : "thank you",
                        "tho" : "though",
                        "thx" : "thank you",
                        "tia" : "thanks in advance",
                        "til" : "today i learned",
                        "tl;dr" : "too long i did not read",
                        "tldr" : "too long i did not read",
                        "tmb" : "tweet me back",
                        "tntl" : "trying not to laugh",
                        "ttyl" : "talk to you later",
                        "u" : "you",
                        "u2" : "you too",
                        "u4e" : "yours for ever",
                        "utc" : "coordinated universal time",
                        "w/" : "with",
                        "w/o" : "without",
                        "w8" : "wait",
                        "wassup" : "what is up",
                        "wb" : "welcome back",
                        "wtf" : "what the fuck",
                        "wtg" : "way to go",
                        "wtpa" : "where the party at",
                        "wuf" : "where are you from",
                        "wuzup" : "what is up",
                        "wywh" : "wish you were here",
                        "yd" : "yard",
                        "ygtr" : "you got that right",
                        "ynk" : "you never know",
                        "zzz" : "sleeping bored and tired"
                        }

        sample_typos_slang_pattern = re.compile(r'(?<!\w)(' + '|'.join(re.escape(key) for key in sample_typos_slang.keys()) + r')(?!\w)')
        sample_acronyms_pattern = re.compile(r'(?<!\w)(' + '|'.join(re.escape(key) for key in sample_acronyms.keys()) + r')(?!\w)')
        sample_abbr_pattern = re.compile(r'(?<!\w)(' + '|'.join(re.escape(key) for key in sample_abbr.keys()) + r')(?!\w)')

        text = sample_typos_slang_pattern.sub(lambda x: sample_typos_slang[x.group()], text)
        text = sample_acronyms_pattern.sub(lambda x: sample_acronyms[x.group()], text)
        text = sample_abbr_pattern.sub(lambda x: sample_abbr[x.group()], text)

        return text

# Applying the cleaining functions defined above
data["text"] = data["text"].apply(lambda x: remove_URL(x))
data["text"] = data["text"].apply(lambda x: remove_html(x))
data["text"] = data["text"].apply(lambda x: remove_non_ascii(x))
data["text"] = data["text"].apply(lambda x: remove_punct(x))
data["text"] = data["text"].apply(lambda x: other_clean(x))


# Removing Stopwords
stop = stopwords.words('english')

data['removed_stopwords'] = data['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

#Lemmatization
# Defining and applying the lemmatization function to lemmatize texts
def lemmatized_text(corpus):
    lemmatizer = WordNetLemmatizer()
    return [' '.join([lemmatizer.lemmatize(word) for word in review.split()]) for review in corpus]

data['lemmatized_texts'] = lemmatized_text(data['removed_stopwords'])




#GloVe Embeddings
import os

file_path = "dataset/glove.6B.100d.txt"
print("File exists:", os.path.isfile(file_path))

# Load GloVe Embeddings and Create Embedding Matrix
def load_glove_embeddings(file_path):
    embeddings_index = {}
    with open(file_path, encoding="utf8") as file:
        for line in file:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = vector
    return embeddings_index

# Check if file exists before attempting to load
if os.path.isfile(file_path):
    embeddings_index = load_glove_embeddings(file_path)
else:
    raise FileNotFoundError("GloVe file not found. Please check the file path.")

print("Number of words in GloVe embeddings:", len(embeddings_index))

# Tokenize the text
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(data['lemmatized_texts'])
word_index = tokenizer.word_index

# Create the embedding matrix
embedding_dim = 100
vocab_size = len(word_index) + 1
embedding_matrix = np.zeros((vocab_size, embedding_dim))

for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# Convert text to sequences
X_seq = tokenizer.texts_to_sequences(data['lemmatized_texts'])
X_pad = pad_sequences(X_seq, maxlen=100)

print("Shape of padded sequences:", X_pad.shape)


#Feature Matrix

def create_feature_matrix(sequences, embedding_matrix):
    features = np.zeros((sequences.shape[0], embedding_matrix.shape[1]))
    for i, seq in enumerate(sequences):
        features[i] = np.mean(embedding_matrix[seq], axis=0)
    return features

X_features = create_feature_matrix(X_pad, embedding_matrix)
print("Shape of feature matrix:", X_features.shape)


#Base Model for LSTM

# Extract texts and labels
texts = data['text'].tolist()
labels = data['is_sarcastic'].tolist()


MAX_SEQUENCE_LENGTH = 30
EMBEDDING_DIM = 100

# Tokenize the texts
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
word_index = tokenizer.word_index

# Convert texts to sequences
sequences = tokenizer.texts_to_sequences(texts)
X_data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
y_data = np.array(labels)

# Load GloVe embeddings
def load_glove_embeddings(file_path, embedding_dim):
    embeddings_index = {}
    with open(file_path, 'r', encoding='utf8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    return embeddings_index

def create_embedding_matrix(word_index, embeddings_index, embedding_dim):
    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix

# Path to the GloVe file
# glove_file = 'path/to/glove.6B.100d.txt'  # Update this path
embeddings_index = load_glove_embeddings(file_path, EMBEDDING_DIM)
embedding_matrix = create_embedding_matrix(word_index, embeddings_index, EMBEDDING_DIM)


vocab_size = len(word_index) + 1

def LSTM_RNN(vocab_size, embed_dim, embed_matrix, max_seq_len):
    embedding_layer = Embedding(vocab_size, embed_dim, weights=[embed_matrix], input_length=max_seq_len, trainable=False)

    sequence_input = Input(shape=(max_seq_len,), dtype='int32')
    embedding_sequences = embedding_layer(sequence_input)
    x = Dropout(0.2)(embedding_sequences)
    x = Conv1D(64, 5, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = Bidirectional(LSTM(64, dropout=0.3, recurrent_dropout=0.2))(x)
    x = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(1, activation='sigmoid')(x)
    model = tf.keras.Model(sequence_input, outputs)

    model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

lstm_model = LSTM_RNN(vocab_size, EMBEDDING_DIM, embedding_matrix, MAX_SEQUENCE_LENGTH)


batch_size = 100
epochs = 10

# Callbacks for early stopping and learning rate reduction
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-5)
model_checkpoint = ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_loss')

history = lstm_model.fit(
    X_data, y_data, 
    epochs=epochs, 
    batch_size=batch_size, 
    validation_split=0.2, 
    verbose=1,
    callbacks=[early_stopping, reduce_lr, model_checkpoint]
)

# # Now dump the tokenizer
joblib.dump(tokenizer, 'tokenizer.joblib')

#use joblib to save the model
lstm_model.save('best_model.h5')





