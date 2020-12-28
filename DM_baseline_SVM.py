import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn_crfsuite import metrics
from sklearn import svm


file_path='C:/Users/ALEX/Desktop/DataMining/Proj/data/SampleData_deid.txt'
file_path2='C:/Users/ALEX/Desktop/DataMining/Proj/data/development_2.txt'


def loadInputFile(path):
    trainingset = list()  # store trainingset [content,content,...]
    position = list()  # store position [article_id, start_pos, end_pos, entity_text, entity_type, ...]
    mentions = dict()  # store mentions[mention] = Type
    with open(file_path, 'r', encoding='utf8') as f:
        file_text = f.read().encode('utf-8').decode('utf-8-sig')
    datas = file_text.split('\n\n--------------------\n\n')[:-1]
    for data in datas:
        data = data.split('\n')
        content = data[0]
        trainingset.append(content)
        annotations = data[1:]
        for annot in annotations[1:]:
            annot = annot.split('\t')  # annot= article_id, start_pos, end_pos, entity_text, entity_type
            position.extend(annot)
            mentions[annot[3]] = annot[4]

    return trainingset, position, mentions


def CRFFormatData(trainingset, position, path):
    if (os.path.isfile(path)):
        os.remove(path)
    outputfile = open(path, 'a', encoding='utf-8')

    # output file lines
    count = 0  # annotation counts in each content
    tagged = list()
    for article_id in range(len(trainingset)):
        trainingset_split = list(trainingset[article_id])
        while '' or ' ' in trainingset_split:
            if '' in trainingset_split:
                trainingset_split.remove('')
            else:
                trainingset_split.remove(' ')
        start_tmp = 0
        for position_idx in range(0, len(position), 5):
            if int(position[position_idx]) == article_id:
                count += 1
                if count == 1:
                    start_pos = int(position[position_idx + 1])
                    end_pos = int(position[position_idx + 2])
                    entity_type = position[position_idx + 4]
                    if start_pos == 0:
                        token = list(trainingset[article_id][start_pos:end_pos])
                        whole_token = trainingset[article_id][start_pos:end_pos]
                        for token_idx in range(len(token)):
                            if len(token[token_idx].replace(' ', '')) == 0:
                                continue
                            # BIO states
                            if token_idx == 0:
                                label = 'B-' + entity_type
                            else:
                                label = 'I-' + entity_type

                            output_str = token[token_idx] + ' ' + label + '\n'
                            outputfile.write(output_str)

                    else:
                        token = list(trainingset[article_id][0:start_pos])
                        whole_token = trainingset[article_id][0:start_pos]
                        for token_idx in range(len(token)):
                            if len(token[token_idx].replace(' ', '')) == 0:
                                continue

                            output_str = token[token_idx] + ' ' + 'O' + '\n'
                            outputfile.write(output_str)

                        token = list(trainingset[article_id][start_pos:end_pos])
                        whole_token = trainingset[article_id][start_pos:end_pos]
                        for token_idx in range(len(token)):
                            if len(token[token_idx].replace(' ', '')) == 0:
                                continue
                            # BIO states
                            if token[0] == '':
                                if token_idx == 1:
                                    label = 'B-' + entity_type
                                else:
                                    label = 'I-' + entity_type
                            else:
                                if token_idx == 0:
                                    label = 'B-' + entity_type
                                else:
                                    label = 'I-' + entity_type

                            output_str = token[token_idx] + ' ' + label + '\n'
                            outputfile.write(output_str)

                    start_tmp = end_pos
                else:
                    start_pos = int(position[position_idx + 1])
                    end_pos = int(position[position_idx + 2])
                    entity_type = position[position_idx + 4]
                    if start_pos < start_tmp:
                        continue
                    else:
                        token = list(trainingset[article_id][start_tmp:start_pos])
                        whole_token = trainingset[article_id][start_tmp:start_pos]
                        for token_idx in range(len(token)):
                            if len(token[token_idx].replace(' ', '')) == 0:
                                continue
                            output_str = token[token_idx] + ' ' + 'O' + '\n'
                            outputfile.write(output_str)

                    token = list(trainingset[article_id][start_pos:end_pos])
                    whole_token = trainingset[article_id][start_pos:end_pos]
                    for token_idx in range(len(token)):
                        if len(token[token_idx].replace(' ', '')) == 0:
                            continue
                        # BIO states
                        if token[0] == '':
                            if token_idx == 1:
                                label = 'B-' + entity_type
                            else:
                                label = 'I-' + entity_type
                        else:
                            if token_idx == 0:
                                label = 'B-' + entity_type
                            else:
                                label = 'I-' + entity_type

                        output_str = token[token_idx] + ' ' + label + '\n'
                        outputfile.write(output_str)
                    start_tmp = end_pos

        token = list(trainingset[article_id][start_tmp:])
        whole_token = trainingset[article_id][start_tmp:]
        for token_idx in range(len(token)):
            if len(token[token_idx].replace(' ', '')) == 0:
                continue

            output_str = token[token_idx] + ' ' + 'O' + '\n'
            outputfile.write(output_str)

        count = 0

        output_str = '\n'
        outputfile.write(output_str)
        ID = trainingset[article_id]

        if article_id % 10 == 0:
            print('Total complete articles:', article_id)

    # close output file
    outputfile.close()

trainingset, position, mentions=loadInputFile(file_path)

data_path='data/sample.data'
CRFFormatData(trainingset, position, data_path)


# load pretrained word vectors
# get a dict of tokens (key) and their pretrained word vectors (value)
# pretrained word2vec CBOW word vector: https://fgc.stpi.narl.org.tw/activity/videoDetail/4b1141305ddf5522015de5479f4701b1
dim = 0
word_vecs = {}
# open pretrained word vector file
with open('cna.cbow.cwe_p.tar_g.512d.0.txt', encoding="utf-8") as f:
    for line in f:
        tokens = line.strip().split()

        # there 2 integers in the first line: vocabulary_size, word_vector_dim
        if len(tokens) == 2:
            dim = int(tokens[1])
            continue

        word = tokens[0]
        vec = np.array([float(t) for t in tokens[1:]])
        word_vecs[word] = vec

print('vocabulary_size: ',len(word_vecs),' word_vector_dim: ',vec.shape)

# load `train.data` and separate into a list of labeled data of each text
# return:
#   data_list: a list of lists of tuples, storing tokens and labels (wrapped in tuple) of each text in `train.data`
#   traindata_list: a list of lists, storing training data_list splitted from data_list
#   testdata_list: a list of lists, storing testing data_list splitted from data_list
from sklearn.model_selection import train_test_split


def Dataset(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        data = f.readlines()  # .encode('utf-8').decode('utf-8-sig')
    data_list, data_list_tmp = list(), list()
    article_id_list = list()
    idx = 0
    for row in data:
        data_tuple = tuple()
        if row == '\n':
            article_id_list.append(idx)
            idx += 1
            data_list.append(data_list_tmp)
            data_list_tmp = []
        else:
            row = row.strip('\n').split(' ')
            data_tuple = (row[0], row[1])
            data_list_tmp.append(data_tuple)
    if len(data_list_tmp) != 0:
        data_list.append(data_list_tmp)

    # here we random split data into training dataset and testing dataset
    # but you should take `development data` or `test data` as testing data
    # At that time, you could just delete this line,
    # and generate data_list of `train data` and data_list of `development/test data` by this function
    traindata_list, testdata_list, traindata_article_id_list, testdata_article_id_list = train_test_split(data_list,
                                                                                                          article_id_list,
                                                                                                          test_size=0.2,
                                                                                                          random_state=42)

    return data_list, traindata_list, testdata_list, traindata_article_id_list, testdata_article_id_list

# look up word vectors
# turn each word into its pretrained word vector
# return a list of word vectors corresponding to each token in train.data
def Word2Vector(data_list, embedding_dict, dim):
    x_train = []
    embedding_list = list()

    # No Match Word (unknown word) Vector in Embedding
    unk_vector=np.random.rand(*(list(embedding_dict.values())[0].shape))

    for idx_list in range(len(data_list)):
        embedding_list_tmp = list()
        for idx_tuple in range(len(data_list[idx_list])):
            key = data_list[idx_list][idx_tuple][0] # token

            if key in embedding_dict:
                value = embedding_dict[key]
            else:
                value = unk_vector
            embedding_list_tmp.append(value[0:dim])
        embedding_list.append(embedding_list_tmp)
    for i in range(len(data_list)):
        for k in range(len(data_list[i])):
            features=embedding_list[i][k]
            x_train.append(features)

    return x_train

# get the labels of each tokens in train.data
# return a list of lists of labels
def Preprocess(data_list):
    y_train = []
    label_list = list()
    for idx_list in range(len(data_list)):
        label_list_tmp = list()
        for idx_tuple in range(len(data_list[idx_list])):
            label_list_tmp.append(data_list[idx_list][idx_tuple][1])
        label_list.append(label_list_tmp)

    for i in range(len(data_list)):
        for k in range(len(data_list[i])):
            labels=label_list[i][k]
            y_train.append(labels)

    return y_train

data_list, traindata_list, testdata_list, traindata_article_id_list, testdata_article_id_list = Dataset(data_path)

# Load Word Embedding
dim = 127
x_train = Word2Vector(traindata_list, word_vecs,dim)
x_test = Word2Vector(testdata_list, word_vecs,dim)

y_train = Preprocess(traindata_list)
y_test = Preprocess(testdata_list)

#y_pred, y_pred_mar, f1score = SVM(x_train, y_train, x_test, y_test)
clf = svm.SVC()
clf.fit(x_train, y_train)
# print(crf)
y_pred = clf.predict(x_test)

labels = list(clf.classes_)
labels.remove('O')
f1_score = f1_score(y_test, y_pred, average='weighted', labels=labels)
sorted_labels = sorted(labels,key=lambda name: (name[1:], name[0])) # group B and I results
print(classification_report(y_test, y_pred, labels=sorted_labels, digits=3))
print(f1_score)

# Test
uploadtest_data_id = []  # store trainingset [content,content,...]
uploadtest_data_text = []

with open(file_path2, 'r', encoding='utf8') as f:
    file_text = f.read().encode('utf-8').decode('utf-8-sig')
datas = file_text.split('\n\n--------------------\n\n')[:-1]
for data in datas:
    data = data.split('\n')
    uploadtest_data_id.append(data[0][12:])
    uploadtest_data_text.append(data[1])

testing_list = Word2Vector(uploadtest_data_text, word_vecs, dim)
# CRF - Test Data (Golden Standard)
y_pred_upload = clf.predict(testing_list)

output = ""
for pred_id in range(len(y_pred_upload)):
    output += str(y_pred_upload[pred_id]) + '\n'

output_path = 'output_SVM_1207_64D_pred.tsv'
with open(output_path, 'w', encoding='utf-8') as f:
    f.write(output)

output_text = []
for i in range(len(uploadtest_data_id)):
    for k in range(len(uploadtest_data_text[i])):
        output_text.append(uploadtest_data_text[i][k])

output = "article_id\tstart_position\tend_position\tentity_text\tentity_type\n"
pos = 0
start_pos = None
end_pos = None
entity_text = None
entity_type = None
for pred_id in range(len(y_pred_upload)):
    if y_pred_upload[pred_id][0] == 'B':
        start_pos = pos
        entity_type = y_pred_upload[pred_id][2:]
    elif start_pos is not None and y_pred_upload[pred_id][0] == 'I' and y_pred_upload[pred_id + 1][0] == 'O':
        end_pos = pos

        testdata_article_id = 0
        L_testdata_list = 0
        final_start_pos = start_pos
        for i in range(len(uploadtest_data_id)):
            L_testdata_list = L_testdata_list + len(uploadtest_data_text[i])
            final_end_pos = end_pos
            if start_pos > L_testdata_list:
                testdata_article_id = i
                final_start_pos = final_start_pos - L_testdata_list
                final_end_pos = end_pos - L_testdata_list

        entity_text = ''.join([output_text[position] for position in range(start_pos, end_pos + 1)])

        line = str(uploadtest_data_id[testdata_article_id]) + '\t'
        line = line + str(final_start_pos) + '\t'
        line = line + str(final_end_pos + 1) + '\t'
        line = line + entity_text + '\t'
        line = line + entity_type
        output += line + '\n'
    pos += 1

output_path='output_SVM_1207_128D.tsv'
with open(output_path,'w',encoding='utf-8') as f:
    f.write(output)
