import json
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from tensorflow.keras import Input,Model
from tensorflow.keras.layers import Dense, Embedding,  TimeDistributed
from tensorflow.keras.utils import to_categorical
import zipfile
import pickle

np.random.seed(1234567890)

# word embedding with GloVe: load embedding as a dict
def load_embedding(filename_embedding,vocabulary,dim=300):
    embedding = dict()
    with open(filename_embedding,'r', encoding='utf8') as file:
        token = '##padding_token##'
        embedding[token] = dict()
        embedding[token]['vector'] = np.asarray(np.random.rand(dim), dtype='float32')
        embedding[token]['token_id'] = int(0)
        token_id = 1
        for line in tqdm(file):
            parts = line.split()
            token = parts[0]
            if token in vocabulary:
                embedding[token] = dict()
                embedding[token]['vector'] = np.asarray(parts[1:], dtype='float32')
                embedding[token]['token_id'] = int(token_id)
                token_id += 1
        # add embedding for unknown tokens
        token = '##unknown_token##'
        embedding[token] = dict()
        embedding[token]['vector'] = np.asarray(np.random.rand(dim), dtype='float32')
        embedding[token]['token_id'] = int(token_id)
        
    return embedding

def convert_tokens_labels_list_to_ids_list(tokens_list, labels_list,embedding, max_seq_length,labelclass_to_id):
    token_ids_list, label_ids_list = [], []
    n_tags = len(list(labelclass_to_id.keys()))
    for index in tqdm(range(len(tokens_list)), desc="Converting tokens & labels to ids "):
        tokens = tokens_list[index]
        labels = labels_list[index]   
        token_ids = []
        for token in tokens:
            if token not in embedding.keys():
                token = '##unknown_token##'
            token_ids.append(embedding[token]['token_id'])
        label_ids = [labelclass_to_id[label] for label in labels]
    
        # Zero-pad up to the sequence length.
        while len(token_ids) < max_seq_length:
            token_ids.append(0)
            label_ids.append(0)
        token_ids_list.append(token_ids)
        label_ids_list.append(label_ids)
        
    return (
        np.array(token_ids_list),
        to_categorical(np.array(label_ids_list), num_classes=n_tags),
    )


# import data
# specify the type of information which shall be extracted
#extraction_of = 'contexts'
extraction_of = 'sentiments'
# extraction_of = 'aspects'


# specify filenames in the next line
if extraction_of in ['contexts']:
    filename = r'data_laptop_ctxt.json'
elif extraction_of in ['sentiments','aspects']:
    filename = r'data_laptop_absa.json'

## in this example, we use the glove word embeddings as input for the neural network
## download glove.42B.300d.txt from http://nlp.stanford.edu/data/glove.42B.300d.zip
filename_embedding_zip = r'glove.42B.300d.zip' # folder of downloaded glove zip file
## specify folder where to store the glove embeddings
filepath_embedding = filename_embedding_zip.replace('.zip','')
## unzip and save glove to a folder manually or with the next lines
if not os.path.exists(filepath_embedding):
    with zipfile.ZipFile(filename_embedding_zip,"r") as zip_ref:
        zip_ref.extractall(filepath_embedding)
os.listdir(filepath_embedding)[0]
filename_embedding = filepath_embedding + '/' + os.listdir(filepath_embedding)[0]


with open(filename,'r', encoding='utf8') as infile:
    example_data = json.load(infile)



for i,(k,v) in enumerate(example_data.items()):
    tokens = v.get('tokens')
    tokens = [token.lower() for token in tokens]
    example_data[k]['tokens'] = tokens

# split 80 20 in train and test tokens
print(len(example_data))

raise Error

train_tokens = [v['tokens'] for k,v in example_data.items()]

# this should be chagend: start
# this should be chagend: end
train_labels = list()
train_labels_uncertainty = list()
for k,v in example_data.items():
    curr_users = [s for s in v.keys() if s !='tokens']
    
    #print(curr_users)
    
    # for illlustration only the annotation of one user is used here -> curr_users[0]
    train_labels.append(v[curr_users[0]][extraction_of])
    train_labels_uncertainty.append(v[curr_users[0]][extraction_of+'_uncertainty'])
# this has to be changed (start)!!!! right now, train and test sets are the same
test_tokens = train_tokens
test_labels = train_labels
# this has to be changed (end)

# generate vocabulary,token ids and embeddings for vocabulary
vocabulary = set(['##unknown_token##','##padding_token##'])
for v in example_data.values():
    vocabulary.update(set(v['tokens']))
vocabulary = list(vocabulary)
vocabulary.sort()
filename_rel_embedding = filename_embedding.replace('.txt','_rel.pkl')
if not os.path.exists(filename_rel_embedding):
    embedding = load_embedding(filename_embedding, vocabulary)
    pickle.dump(embedding,open(filename_rel_embedding,'wb'))
else:
    embedding = pickle.load(open(filename_rel_embedding,'rb'))
vocab_size = len (vocabulary)
embed_size = list(embedding.values())[0]['vector'].shape[0]
embedding_vectors = np.zeros((vocab_size, embed_size))
for v in embedding.values():
    vector = v['vector']
    token_id = v['token_id']
    embedding_vectors[token_id] = vector

all_labelclasses = set()
for row in train_labels:
    all_labelclasses.update(row)
all_labelclasses=list(all_labelclasses)
all_labelclasses.sort()

labelclass_to_id = dict(zip(all_labelclasses,list(range(len(all_labelclasses)))))


n_tags = len(list(labelclass_to_id.keys()))


max_seq_length = 100

# Create datasets (Only take up to max_seq_length words for memory)
train_tokens = [t[0:max_seq_length] for t in train_tokens]
test_tokens = [t[0:max_seq_length] for t in test_tokens]
train_labels = [t[0:max_seq_length] for t in train_labels]
test_labels = [t[0:max_seq_length] for t in test_labels]

# Convert data to Input format for neural network
x_train, y_train = convert_tokens_labels_list_to_ids_list(train_tokens, train_labels, embedding, max_seq_length,labelclass_to_id)
x_test, y_test = convert_tokens_labels_list_to_ids_list(test_tokens, test_labels, embedding, max_seq_length,labelclass_to_id)

# neural network model in keras
# see keras documentation for functions of different layers and structure of networks.

# the following two layers should not be changed.
input_layer = Input(shape=(max_seq_length,))
embedding_layer = Embedding(vocab_size, 300, weights=[embedding_vectors], input_length=max_seq_length, trainable= False)(input_layer)

# here, attention models have to be implemented in this model
# ...

# this last layer can/should be modified
output_layer = TimeDistributed(Dense(n_tags, activation="softmax"))(embedding_layer)

model = Model(inputs=input_layer, outputs=output_layer)
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["categorical_accuracy"])
model.summary()

# fit model on train data
model.fit(x_train, y_train,
          batch_size=128,
          epochs=10)

## evaluation
# predict labels of test data
y_test_pred_prob = model.predict(x_test)
y_test_pred_sparse = y_test_pred_prob.argmax(axis=-1)
y_test_pred = to_categorical(np.array(y_test_pred_sparse), num_classes=n_tags)

# compute confusion matrix
conf_matrix = np.zeros((n_tags, n_tags))
for i,tokens in enumerate(test_tokens):
    for j,_ in enumerate(tokens):
        class_true = y_test[i,j].argmax()
        class_pred = y_test_pred[i,j].argmax()
        conf_matrix[class_true,class_pred] += 1
names_rows = list(s+'_true' for s in labelclass_to_id.keys())
names_columns = list(s+'_pred' for s in labelclass_to_id.keys())
conf_matrix = pd.DataFrame(data=conf_matrix,index=names_rows,columns=names_columns)

# compute final evaluation measures
precision_per_class = np.zeros((n_tags,))
recall_per_class = np.zeros((n_tags,))
for i in range(n_tags):
    if conf_matrix.values[i,i] > 0:
        precision_per_class[i] = conf_matrix.values[i,i]/sum(conf_matrix.values[:,i])
        recall_per_class[i] = conf_matrix.values[i,i]/sum(conf_matrix.values[i,:])
precision = np.mean(precision_per_class)
recall = np.mean(recall_per_class)
f1 = 2*(precision*recall)/(precision+recall)

print('Precision: '+str(precision))
print('Recall: '+str(recall))
print('F1-measure: '+str(f1))
