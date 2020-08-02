'''
Task2 baseline model
'''
import csv
import nltk
import pycrfsuite
import ast
import numpy as np
import sys
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
def get_antecedent_begin_end(sentence_token, antetoken):
    '''
    :param sentence_token: the whole tokenized sentence
    :param antetoken: antecedent of this tokenized sentence
    :return: ante_begin：the begin index of antecedent
             ante_end:the end index of antecedent
    '''
    flag = 0
    j = 0
    begin = end = 0
    for index, words in enumerate(sentence_token):
        if words == antetoken[0] and flag == 0:
            begin = index
            j = 1
        elif j >= len(antetoken) - 1:
            end = index
            j = len(antetoken) - 1
            break
        elif words == antetoken[j] and j != 0:
            flag = 1
            j += 1
        else:
            flag = 0
    if begin != 0 and end == 0:
        for index, words in enumerate(sentence_token):
            if '-' in words:
                subgroup = words.split('-', 1)
                sentence_token.insert(int(index), subgroup[0])
                sentence_token.insert(int(index) + 1, subgroup[1])
                sentence_token.remove(words)
        flag = 0
        j = 0
        begin = end = 0
        for index, words in enumerate(sentence_token):
            if words == antetoken[0] and flag == 0:
                begin = index
                j = 1
            elif j >= len(antetoken) - 1:
                end = index
                j = len(antetoken) - 1
                break
            elif words == antetoken[j] and j != 0:
                flag = 1
                j += 1
            else:
                flag = 0
    ante_begin=begin
    ante_end=end
    return ante_begin, ante_end
def get_consequence_begin_end(sentence_token, consetoken):
    '''

    :param sentence_token: the whole tokenized sentence
    :param consetoken: the consequence of sentence(if exists)
    :return: conse_begin:the begin index of consequence conse_end: the end index of consequence
    '''
    flag = 0
    j = 0
    begin = end = 0
    for index, words in enumerate(sentence_token):
        if words == consetoken[0] and flag == 0:
            begin = index
            j = 1
        elif j >= len(consetoken) - 1:
            end = index
            j = len(consetoken) - 1
            break
        elif words == consetoken[j] and j != 0:
            flag = 1
            j += 1

        else:
            flag = 0
    if begin != 0 and end == 0:
        for index, words in enumerate(sentence_token):
            if '-' in words:
                subgroup = words.split('-', 1)
                sentence_token.insert(int(index), subgroup[0])
                sentence_token.insert(int(index) + 1, subgroup[1])
                sentence_token.remove(words)
        flag = 0
        j = 0
        begin = end = 0
        for index, words in enumerate(sentence_token):
            if words == consetoken[0] and flag == 0:
                begin = index
                j = 1
            elif j >= len(consetoken) - 1:
                end = index
                j = len(consetoken) - 1
                break
            elif words == consetoken[j] and j != 0:
                flag = 1
                j += 1
            else:
                flag = 0
    conse_begin=begin
    conse_end=end
    return conse_begin, conse_end

def word2features(doc, i):
    '''
    For the input word, create its feature list
    :param doc: tokenized sentence
    :param i: index
    :return: features: the word's feature list
    '''
    word = doc[i][0]
    postag = doc[i][1]

    features = [
        'bias',
        'word.lower=' + word.lower(),
        'word[-3:]=' + word[-3:],
        'word[-2:]=' + word[-2:],
        'word.isupper=%s' % word.isupper(),
        'word.istitle=%s' % word.istitle(),
        'word.isdigit=%s' % word.isdigit(),
        'postag=' + postag
    ]


    if i > 0:
        word1 = doc[i - 1][0]
        postag1 = doc[i - 1][1]
        features.extend([
            '-1:word.lower=' + word1.lower(),
            '-1:word.istitle=%s' % word1.istitle(),
            '-1:word.isupper=%s' % word1.isupper(),
            '-1:word.isdigit=%s' % word1.isdigit(),
            '-1:postag=' + postag1
        ])
    else:

        features.append('BOS')



    if i < len(doc) - 1:
        word1 = doc[i + 1][0]
        postag1 = doc[i + 1][1]
        features.extend([
            '+1:word.lower=' + word1.lower(),
            '+1:word.istitle=%s' % word1.istitle(),
            '+1:word.isupper=%s' % word1.isupper(),
            '+1:word.isdigit=%s' % word1.isdigit(),
            '+1:postag=' + postag1
        ])
    else:
        features.append('EOS')

    return features
def extract_features(doc):
    '''
    extract the features in one sentence
    :param doc: tokenized sentence
    :return: a feature list for every word in this sentence
    '''
    return [word2features(doc, i) for i in range(len(doc))]
def get_labels(doc):
    '''
    extract the labels in one sentence
    :param doc: tokenized sentence
    :return: a label list for every word in this sentence
    '''
    return [label for (token, postag, label) in doc]
def is_correct_predict(y_sentence):
    '''
    check if the submission result makes legal prediction
    :param y_sentence: submission result
    :return: True for legal, False for illegal
    '''
    cnt_Bant=0
    cnt_Bcon=0
    for word in y_sentence:
        if word=='B-Con':
            cnt_Bcon+=1
        if word=='B-Ant':
            cnt_Bant+=1
        else:
            continue
    if cnt_Bant>1 or cnt_Bcon>1:
        return False
    else:
        return True
def get_coordinate(x_data, y_data):
    '''
    get the coordinate of y_data
        :param x_data: dataset
        :param y_data: labelset
        :return: conse_begin: the begin index of consequence
                 conse_end: the end index of consequence
                 ante_begin: the begin index of antecedent
                 ante_end: the end index of antecedent

    '''
    conse_begin = 0
    conse_begin_mask = 1
    conse_end = 0
    conse_end_mask = 1
    ante_begin = 0
    ante_begin_mask = 1
    ante_end = 0
    ante_end_mask = 1
    coordinate_sentence = []
    coordinate = []
    for index_sentence, y_sentence in enumerate(y_data):
        for index_word, y_word in enumerate(y_sentence[1]):
            if is_correct_predict(y_sentence[1]) == False:
                ante_begin = -1
                ante_end = -1
                conse_begin = -1
                conse_end = -1
            else:
                if y_word == "O":
                    conse_begin += (len(x_data[index_sentence][1][index_word][1]) - 11 + 1) * conse_begin_mask
                    conse_end += (len(x_data[index_sentence][1][index_word][1]) - 11 + 1) * conse_end_mask
                    ante_begin += (len(x_data[index_sentence][1][index_word][1]) - 11 + 1) * ante_begin_mask
                    ante_end += (len(x_data[index_sentence][1][index_word][1]) - 11 + 1) * ante_end_mask
                    continue
                elif y_word == 'B-Con':
                    conse_begin_mask = 0
                    conse_end += (len(x_data[index_sentence][1][index_word][1]) - 11 + 1) * conse_end_mask
                    ante_begin += (len(x_data[index_sentence][1][index_word][1]) - 11 + 1) * ante_begin_mask
                    ante_end += (len(x_data[index_sentence][1][index_word][1]) - 11 + 1) * ante_end_mask
                    continue
                elif y_word == 'I-Con' and (
                        (index_word < (len(y_sentence[1]) - 1)) and (y_sentence[1][index_word + 1] == 'I-Con')):
                    conse_begin += (len(x_data[index_sentence][1][index_word][1]) - 11 + 1) * conse_begin_mask
                    conse_end += (len(x_data[index_sentence][1][index_word][1]) - 11 + 1) * conse_end_mask
                    ante_begin += (len(x_data[index_sentence][1][index_word][1]) - 11 + 1) * ante_begin_mask
                    ante_end += (len(x_data[index_sentence][1][index_word][1]) - 11 + 1) * ante_end_mask
                    continue
                elif y_word == 'I-Con' and (
                        (index_word == (len(y_sentence[1]) - 1)) or (y_sentence[1][index_word + 1] != 'I-Con')):
                    conse_end += (len(x_data[index_sentence][1][index_word][1]) - 11) * conse_end_mask
                    conse_begin += (len(x_data[index_sentence][1][index_word][1]) - 11 + 1) * conse_begin_mask
                    ante_begin += (len(x_data[index_sentence][1][index_word][1]) - 11 + 1) * ante_begin_mask
                    ante_end += (len(x_data[index_sentence][1][index_word][1]) - 11 + 1) * ante_end_mask
                    conse_end_mask = 0
                    continue
                elif y_word == 'B-Ant':
                    ante_begin_mask = 0
                    conse_begin += (len(x_data[index_sentence][1][index_word][1]) - 11 + 1) * conse_begin_mask
                    conse_end += (len(x_data[index_sentence][1][index_word][1]) - 11 + 1) * conse_end_mask
                    ante_end += (len(x_data[index_sentence][1][index_word][1]) - 11 + 1) * ante_end_mask
                    continue
                elif y_word == 'I-Ant' and (
                        (index_word < (len(y_sentence[1]) - 1)) and (y_sentence[1][index_word + 1] == 'I-Ant')):
                    conse_begin += (len(x_data[index_sentence][1][index_word][1]) - 11 + 1) * conse_begin_mask
                    conse_end += (len(x_data[index_sentence][1][index_word][1]) - 11 + 1) * conse_end_mask
                    ante_begin += (len(x_data[index_sentence][1][index_word][1]) - 11 + 1) * ante_begin_mask
                    ante_end += (len(x_data[index_sentence][1][index_word][1]) - 11 + 1) * ante_end_mask
                    continue
                elif y_word == 'I-Ant' and (
                        (index_word == (len(y_sentence[1]) - 1)) or (y_sentence[1][index_word + 1] != 'I-Ant')):
                    conse_begin += (len(x_data[index_sentence][1][index_word][1]) - 11 + 1) * conse_begin_mask
                    conse_end += (len(x_data[index_sentence][1][index_word][1]) - 11 + 1) * conse_end_mask
                    ante_begin += (len(x_data[index_sentence][1][index_word][1]) - 11 + 1) * ante_begin_mask
                    ante_end += (len(x_data[index_sentence][1][index_word][1]) - 11) * ante_end_mask
                    ante_end_mask = 0
                    continue
        ante_end -= 1
        if conse_end == conse_begin:
            conse_end = -1
            conse_begin = -1
        else:
            conse_end -= 1
        coordinate_sentence = [y_sentence[0], ante_begin, ante_end, conse_begin, conse_end]
        coordinate.append(coordinate_sentence)

        conse_begin = 0
        conse_end = 0
        ante_begin = 0
        ante_end = 0
        conse_begin_mask = 1
        conse_end_mask = 1
        ante_begin_mask = 1
        ante_end_mask = 1
    return coordinate

def format_judge(submission):
    '''
    judge if the submission file's format is legal
    :param submission: submission file
    :return: False for illegal
             True for legal
    '''
    # submission: [sentenceID,antecedent_startid,antecedent_endid,consequent_startid,consequent_endid]

    if submission[1] == '-1' or submission[2] == '-1':
        return False
    if (submission[3] == '-1' and submission[4] != '-1') or (submission[3] != '-1' and submission[4] == '-1'):
        return False
    if (int(submission[1]) >= int(submission[2])) or (int(submission[3]) > int(submission[4])):
        return False
    if not (int(submission[1]) >= -1 and int(submission[2]) >= -1 and int(submission[3]) >= -1 and int(submission[4]) >= -1):
        return False
    return True
def get_inter_id(submission_idx, truth_idx):
    # print(submission_idx)
    # print(truth_idx)
    sub_start = int(submission_idx[0])
    sub_end = int(submission_idx[1])
    truth_start = int(truth_idx[0])
    truth_end = int(truth_idx[1])
    if sub_end < truth_start or sub_start > truth_end:
        return False, -1, -1
    return True, max(sub_start, truth_start), min(sub_end, truth_end)

def metrics_task2(submission_list, truth_list):
    # submission_list:  [[sentenceID,antecedent_startid,antecedent_endid,consequent_startid,consequent_endid], ...]
    # truth_list:       [[sentenceID,sentence, antecedent_startid,antecedent_endid,consequent_startid,consequent_endid], ...]
    f1_score_all = []
    precision_all = []
    recall_all = []

    for i in range(len(submission_list)):
        assert submission_list[i][0] == truth_list[i][0]
        submission = submission_list[i]
        truth = truth_list[i]

        precision = 0
        recall = 0
        f1_score = 0

        if format_judge(submission):
            # truth processing
            sentence = truth[1]

            t_a_s = int(truth[2])       # truth_antecedent_startid
            t_a_e = int(truth[3])       # truth_antecedent_endid
            t_c_s = int(truth[4])       # truth_consequent_startid
            t_c_e = int(truth[5])       # truth_consequent_endid

            s_a_s = int(submission[1])  # submission_antecedent_startid
            s_a_e = int(submission[2])  # submission_antecedent_endid
            s_c_s = int(submission[3])  # submission_consequent_startid
            s_c_e = int(submission[4])  # submission_consequent_endid

            truth_ante_len = len(sentence[t_a_s : t_a_e].split())
            if truth[4] == '-1':
                truth_cons_len = 0
            else:
                truth_cons_len = len(sentence[t_c_s : t_c_e].split())
            truth_len = truth_ante_len + truth_cons_len

            # submission processing
            submission_ante_len = len(sentence[s_a_s : s_a_e].split())
            if submission[3] == '-1':
                submission_cons_len = 0
            else:
                submission_cons_len = len(sentence[s_c_s : s_c_e].split())
            submission_len = submission_ante_len + submission_cons_len

            # intersection
            inter_ante_flag, inter_ante_startid, inter_ante_endid = get_inter_id([s_a_s, s_a_e], [t_a_s, t_a_e])
            if truth_cons_len == 0 or submission_cons_len == 0:
                inter_cons_startid = 0
                inter_cons_endid = 0
                inter_cons_flag = False
            else:
                inter_cons_flag, inter_cons_startid, inter_cons_endid = get_inter_id([s_c_s, s_c_e], [t_c_s, t_c_e])

            inter_ante_len = 0
            inter_cons_len = 0
            if inter_ante_flag:
                inter_ante_len = len(sentence[inter_ante_startid : inter_ante_endid].split())
            if inter_cons_flag:
                inter_cons_len = len(sentence[inter_cons_startid : inter_cons_endid].split())
            inter_len = inter_ante_len + inter_cons_len

            # calculate precision, recall, f1-score
            if inter_len > 0:
                precision = inter_len / submission_len
                recall = inter_len / truth_len
                f1_score = 2 * precision * recall / (precision + recall)

        precision_all.append(precision)
        recall_all.append(recall)
        f1_score_all.append(f1_score)

    f1_mean = np.mean(f1_score_all)
    precision_mean = np.mean(precision_all)
    recall_mean = np.mean(recall_all)
    return f1_mean, precision_mean, recall_mean,f1_score_all
def evaluate2(truth_reader,submission_list,true_sentence):
    truth_list=[]
    not_em = 0
    for idx, line in enumerate(truth_reader):
        tmp = []
        submission_line = submission_list[idx]
        if line[0] != submission_line[0]:
            # print("the sentence id is not matched")
            sys.exit("Sorry, the sentence id is not matched.")
        tmp.append(line[0])    # sentenceID
        tmp.append(true_sentence[idx][1])
        tmp.extend(line[1:])  # ante_start, ante_end, conq_start, conq_end
        truth_list.append(tmp)

        if submission_line[1] != tmp[2] or submission_line[2] != tmp[3] or submission_line[3] != tmp[4] or submission_line[4] != tmp[5]:
            not_em += 1

    if len(truth_list) != len(submission_list):
        # print("please check the rows#")
        sys.exit("Please check the number of rows in your .csv file! It should consistent with 'train.csv' in practice stage, and should be consistent with 'test.csv' in evaluation stage.")

    exact_match = (len(truth_list) - not_em) / len(truth_list)
    f1_mean, recall_mean, precision_mean,f1_score_all = metrics_task2(submission_list, truth_list)

    return f1_mean, recall_mean, precision_mean, exact_match,f1_score_all
def turn_list_to_str(sentence):
    '''
    recover list to string
    :param sentence: token list
    :return: string
    '''
    sentence_str=""
    for index,word in enumerate(sentence):
        if str(word).find('\'')==0 and len(str(word))!=1 and str(word[0]).find('\'')==-1:
            sentence_str+=('\''+" "+str(word[1:]))
        elif str(word).find('\'') == 0 and len(str(word)) != 1:
            sentence_str =sentence_str.rstrip ()+str(word)+" "
        elif str(word).find('\'')>0:
            sentence_str=sentence_str.rstrip()+str(word)+" "
        elif str(word)=='{' or str(word)=='}' or str(word)==':':
            sentence_str+=str(word)
        elif str(word)=='/' or str(word)==')'or str(word)=='\''or str(word)=='\"'or str(word)=='(' or str(word).find('-')!=-1:
            sentence_str=sentence_str.rstrip()+str(word)
        elif str(word)==',' or str(word)=='.':
            sentence_str=sentence_str.rstrip()+str(word)+" "
        else:
            sentence_str+=(str(word)+" ")
    return sentence_str.strip()
def turn_sentences_to_str(predict_sentences):
    '''
    recover token sequence to original sentence
    :param predict_sentences: token sequence
    :return: original sentence
    '''
    for sentence in predict_sentences:
        sentence[0]=sentence[0]
        sentence[1]=turn_list_to_str(sentence[1])
        sentence[2]=turn_list_to_str(sentence[2])
        sentence[3]=turn_list_to_str(sentence[3])
    return predict_sentences
def get_predict_content(test_data_reader,y_test):
    '''
    given the three-element-tuple list and label of each instance, return the original sentence of this instance
    :param test_data_reader:the three-element-tuple list for test_data
    :param y_test: label of each test instance
    :return:original sentence
    '''
    predict_sentences=[]
    for index,predict_label in enumerate(y_test):
        predict_sentence=[]
        predict_sentence.append(predict_label[0])
        antecedent=[]
        consequence=[]
        sentence=[]
        for test_data in test_data_reader:
            if test_data[0]==predict_label[0]:
                for index,tag in enumerate(predict_label[1]):
                    if tag=='O':
                        sentence.append(test_data[index+1][0])
                        continue
                    if tag=='B-Ant' or tag=='I-Ant':
                        sentence.append(test_data[index+1][0])
                        antecedent.append(test_data[index+1][0])
                        continue
                    if tag=='B-Con' or tag=='I-Con':
                        sentence.append(test_data[index +1][0])
                        consequence.append(test_data[index+1][0])
                        continue
        if len(consequence)==0:
            consequence.append('{')
            consequence.append('}')
        predict_sentence.append(sentence)
        predict_sentence.append(antecedent)
        predict_sentence.append(consequence)
        predict_sentences.append(predict_sentence)
    return predict_sentences
docs_train = []
docs_test=[]
filepath_train = r"task2_train.csv"
filepath_test=r"task2_test.csv"
with open(filepath_train, 'r', encoding="utf-8",errors="ignore") as readFile:  # file.close()
    reader = csv.reader(readFile)  # csv reader

    lines = list(reader)
    for line in lines:
        if line[1]!='sentence':
            sentence_index=line[0]
            subdoc=[]
            senten=line[1]
            label=[]
            sentence_token=nltk.tokenize.word_tokenize(senten)
            antf=0
            contf=0
            ant=line[3]
            con=line[4]

            if ant!='N/A':
                antf=1
            if antf!=0:
                anttoken=nltk.tokenize.word_tokenize(ant)


            if con!='N/A':
                conf=1
            if conf!=0:
                contoken=nltk.tokenize.word_tokenize(con)


            if (antf==1):
                ante_begin_index,ante_end_index=get_antecedent_begin_end(sentence_token=sentence_token,antetoken=anttoken)
            if (conf==1):
                conse_begin_index,conse_end_index=get_consequence_begin_end(sentence_token=sentence_token,consetoken=contoken)
            for i in range(0, len(sentence_token)):
                label.append('O')
            if antf != 0:
                label[ante_begin_index] = 'B-Ant'
                for i in range(1, len(anttoken)):
                    label[ante_begin_index + i] = 'I-Ant'
            if conf != 0:
                label[conse_begin_index] = 'B-Con'
                for i in range(1, len(contoken)):
                    label[conse_begin_index + i] = 'I-Con'

            for i in range(0, len(sentence_token)):
                subdoc.append(tuple((sentence_token[i], label[i])))

            subdoc.insert(0,sentence_index)

            docs_train.append(subdoc)
readFile.close()

with open(filepath_test, 'r', encoding="utf-8",errors="ignore") as readFile:  # file.close()
    reader = csv.reader(readFile)

    lines = list(reader)
    for line in lines:
        if line[1]!='sentence':
            sentence_index=line[0]
            subdoc=[]
            senten=line[1]
            label=[]
            sentence_token=nltk.tokenize.word_tokenize(senten)
            antf=0
            contf=0
            ant=line[3]
            con=line[4]

            if ant!='N/A':
                antf=1
            if antf!=0:
                anttoken=nltk.tokenize.word_tokenize(ant)


            if con!='N/A':
                conf=1
            if conf!=0:
                contoken=nltk.tokenize.word_tokenize(con)


            if (antf==1):
                ante_begin_index,ante_end_index=get_antecedent_begin_end(sentence_token=sentence_token,antetoken=anttoken)
            if (conf==1):
                conse_begin_index,conse_end_index=get_consequence_begin_end(sentence_token=sentence_token,consetoken=contoken)
            for i in range(0, len(sentence_token)):
                label.append('O')
            if antf != 0:
                label[ante_begin_index] = 'B-Ant'
                for i in range(1, len(anttoken)):
                    label[ante_begin_index + i] = 'I-Ant'
            if conf != 0:
                label[conse_begin_index] = 'B-Con'
                for i in range(1, len(contoken)):
                    label[conse_begin_index + i] = 'I-Con'

            for i in range(0, len(sentence_token)):
                subdoc.append(tuple((sentence_token[i], label[i])))

            subdoc.insert(0,sentence_index)

            docs_test.append(subdoc)
readFile.close()
data_train = []
data_test=[]
for i, doc in enumerate(docs_train):
    tokens = [t for t, label in doc[1:]]
    tagged = nltk.pos_tag(tokens)


    subdata=[(w, pos, label) for (w, label), (word, pos) in zip(doc[1:], tagged)]
    subdata.insert(0,doc[0])

    data_train.append(subdata)
for i, doc in enumerate(docs_test):
    tokens = [t for t, label in doc[1:]]
    tagged = nltk.pos_tag(tokens)

    subdata=[(w, pos, label) for (w, label), (word, pos) in zip(doc[1:], tagged)]
    subdata.insert(0,doc[0])

    data_test.append(subdata)
with open('train_v2_after_proecessing.csv','wt',encoding='utf-8',newline='') as save_data:
    cw=csv.writer(save_data)
    for row in data_train:
        cw.writerow(row)
with open('test_v2_after_proecessing.csv','wt',encoding='utf-8',newline='') as save_data:
    cw=csv.writer(save_data)
    for row in data_test:
        cw.writerow(row)


train_data_reader=[]
train_data_reader_line=[]
with open('train_v2_after_proecessing.csv','r',encoding='utf-8',errors="ignore") as csvfile:
    csv_reader = csv.reader(csvfile)
    for row in csv_reader:
        train_data_reader_line=[ast.literal_eval(word)for word in row[1:]]
        train_data_reader_line.insert(0,row[0])
        train_data_reader.append(train_data_reader_line)
test_data_reader=[]
test_data_reader_line=[]
with open('test_v2_after_proecessing.csv','r',encoding='utf-8',errors="ignore") as csvfile:
    csv_reader = csv.reader(csvfile)
    for row in csv_reader:
        test_data_reader_line=[ast.literal_eval(word)for word in row[1:]]
        test_data_reader_line.insert(0,row[0])
        test_data_reader.append(test_data_reader_line)

train_sentence_indexs=[doc[0] for doc in train_data_reader]
test_sentence_indexs=[doc[0] for doc in test_data_reader]
X_train = [extract_features(doc[1:]) for doc in train_data_reader]
y_train = [get_labels(doc[1:]) for doc in train_data_reader]
X_test = [extract_features(doc[1:]) for doc in test_data_reader]
y_test = [get_labels(doc[1:]) for doc in test_data_reader]

X_train=[[train_sentence_indexs[index],x_sentence] for index, x_sentence in enumerate(X_train)]
y_train=[[train_sentence_indexs[index],y_sentence] for index, y_sentence in enumerate(y_train)]
X_test=[[test_sentence_indexs[index],x_sentence] for index, x_sentence in enumerate(X_test)]
y_test=[[test_sentence_indexs[index],y_sentence] for index, y_sentence in enumerate(y_test)]

trainer = pycrfsuite.Trainer(verbose=True)
for xseq, yseq in zip(X_train, y_train):
    trainer.append(xseq[1], yseq[1])
trainer.set_params({
    'c1':1.0,
    'c2':1.0,
    'max_iterations': 200,
    'feature.possible_transitions': True
})
trainer.train(r"crf.model")
tagger1=pycrfsuite.Tagger()
tagger1.open("crf.model")
y_pred=[]
for xseq in X_test:
    y_pred_sentence = tagger1.tag(xseq[1])
    y_pred.append([xseq[0], y_pred_sentence])
coordinate_pred = get_coordinate(X_test, y_pred)
coordinate_true = get_coordinate(X_test, y_test)
true_sentence=get_predict_content(test_data_reader,y_test)
true_sentence=turn_sentences_to_str(true_sentence)
f1_mean, recall_mean, precision_mean, exact_match,f1_score_all=evaluate2(coordinate_true,coordinate_pred,true_sentence)
print("{}: precision :{:.3f}\t recall:{:.3f}\t f1_score:{:.3f}\t exact_match:{:.3f}".format("average",precision_mean,recall_mean,f1_mean,exact_match))