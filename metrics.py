'''
文件说明
三个评价指标，分别有中文版和英文版，中英文互不兼容。
文件末尾有一段使用样例，可直接运行。


bleu_zh(main_string,strings,N,r)    ->  n,N_gram_score,total_score
    字符串main_string/candidate为待打分的语句
    字符串数组strings/references/refs=[string1,string2,string3]为参考语句构成的数组
    整数N为N-gram中的N，可取1，2，3，4，即N=3时会计算1——gram、2——gram、3——gram
    整数r为期望的输出长度（汉字、单词数），待打分语句长度小于r时会有惩罚
    n=min(字符串长度，N)
    N_gram_score=[1_gram_score,2_gram_score,······,N_gram_score]
    可根据输出的n和N_gram_score调整N和R

meteor_zh(candidate,references,alpha,beta,gamma)    ->  score
    字符串main_string/candidate为待打分的语句
    字符串数组strings/references/refs=[string1,string2,string3]为参考语句构成的数组
    浮点数alpha,beta,gamma均为超参数，一般建议取alpha=0.9,beta= 3.0,gamma=0.5

cider_zh(candidate, refs)    ->  score
    字符串main_string/candidate为待打分的语句
    字符串数组strings/references/refs=[string1,string2,string3]为参考语句构成的数组
'''



import math
from collections import Counter
from math import log

import jieba
import nltk
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.translate.meteor_score import meteor_score
from nltk.stem import WordNetLemmatizer


def bleu_zh(main_string,strings,N,r):
    string1 = strings[0]
    string2 = strings[1]
    string3 = strings[2]
    N_gram_score=np.zeros(N)
    unique_words = set(list(main_string))

    total_occurrences, matching_occurrences = 0, 0

    for word in unique_words:
        count_main_string = main_string.count(word)
        total_occurrences += count_main_string
        matching_occurrences += min(count_main_string,
                                    max(string1.count(word), string2.count(word), string3.count(word)))

    if total_occurrences!=0:
        N_gram_score[0] = matching_occurrences / total_occurrences

    n = min(N, total_occurrences)
    nn=total_occurrences
    word_tokens = list(main_string)
    if n>=2:
        # 计算双词
        bigrams = set([f"{word_tokens[i]}{word_tokens[i + 1]}" for i in range(len(word_tokens) - 1)])
        total_occurrences, matching_occurrences = 0, 0

        for bigram in bigrams:
            count_main_string = main_string.count(bigram)
            total_occurrences += count_main_string
            matching_occurrences += min(count_main_string,
                                        max(string1.count(bigram), string2.count(bigram), string3.count(bigram)))

        N_gram_score[1] = matching_occurrences / total_occurrences
    if n>=3:
        trigrams = set([f"{word_tokens[i]}{word_tokens[i + 1]}{word_tokens[i + 2]}" for i in range(len(word_tokens) - 2)])
        total_occurrences, matching_occurrences = 0, 0

        for trigram in trigrams:
            count_main_string = main_string.count(trigram)
            total_occurrences += count_main_string
            matching_occurrences += min(count_main_string,
                                        max(string1.count(trigram), string2.count(trigram), string3.count(trigram)))

        N_gram_score[2] = matching_occurrences / total_occurrences

    if n>=4:
        four_grams = set([f"{word_tokens[i]}{word_tokens[i + 1]}{word_tokens[i + 2]}{word_tokens[i + 3]}" for i in range(len(word_tokens) - 3)])
        total_occurrences, matching_occurrences = 0, 0

        for four_gram in four_grams:
            count_main_string = main_string.count(four_gram)
            total_occurrences += count_main_string
            matching_occurrences += min(count_main_string,
                                        max(string1.count(four_gram), string2.count(four_gram), string3.count(four_gram)))

        N_gram_score[3] = matching_occurrences / total_occurrences

    total_score = sum((1. / N) * math.log(p + 1) for p in N_gram_score)
    if nn < r:
        total_score *= math.exp(1 - float(r) / nn)
    return nn,N_gram_score,total_score


def bleu_en(main_string,strings,N,r):

    string1 = strings[0]
    string2 = strings[1]
    string3 = strings[2]
    N_gram_score=np.zeros(N)
    unique_words = set(main_string.split())

    total_occurrences, matching_occurrences = 0, 0

    for word in unique_words:
        count_main_string = main_string.count(word)
        total_occurrences += count_main_string
        matching_occurrences += min(count_main_string,
                                    max(string1.count(word), string2.count(word), string3.count(word)))

    if total_occurrences!=0:
        N_gram_score[0] = matching_occurrences / total_occurrences

    n = min(N, total_occurrences)
    nn = total_occurrences
    word_tokens = main_string.split()
    if n>=2:
        # 计算双词
        bigrams = set([f"{word_tokens[i]} {word_tokens[i + 1]}" for i in range(len(word_tokens) - 1)])
        total_occurrences, matching_occurrences = 0, 0

        for bigram in bigrams:
            count_main_string = main_string.count(bigram)
            total_occurrences += count_main_string
            matching_occurrences += min(count_main_string,
                                        max(string1.count(bigram), string2.count(bigram), string3.count(bigram)))

        if total_occurrences!=0:
            N_gram_score[1] = matching_occurrences / total_occurrences
    if n>=3:
        trigrams = set([f"{word_tokens[i]} {word_tokens[i + 1]} {word_tokens[i + 2]}" for i in range(len(word_tokens) - 2)])
        total_occurrences, matching_occurrences = 0, 0

        for trigram in trigrams:
            count_main_string = main_string.count(trigram)
            total_occurrences += count_main_string
            matching_occurrences += min(count_main_string,
                                        max(string1.count(trigram), string2.count(trigram), string3.count(trigram)))

        if total_occurrences!=0:
            N_gram_score[2] = matching_occurrences / total_occurrences

    if n>=4:
        four_grams = set([f"{word_tokens[i]} {word_tokens[i + 1]} {word_tokens[i + 2]} {word_tokens[i + 3]}" for i in range(len(word_tokens) - 3)])
        total_occurrences, matching_occurrences = 0, 0

        for four_gram in four_grams:
            count_main_string = main_string.count(four_gram)
            total_occurrences += count_main_string
            matching_occurrences += min(count_main_string,
                                        max(string1.count(four_gram), string2.count(four_gram), string3.count(four_gram)))

        if total_occurrences!=0:
            N_gram_score[3] = matching_occurrences / total_occurrences

    total_score = sum((1. / N) * math.log(p+1) for p in N_gram_score)
    if nn<r:
        total_score *= math.exp(1 - float(r)/nn)

    return nn,N_gram_score,total_score

def meteor_zh(candidate,references,alpha,beta,gamma):

    reference_tokenized1 = [list(jieba.cut(ref)) for ref in [references[0]]]
    reference_tokenized2 = [list(jieba.cut(ref)) for ref in [references[1]]]
    reference_tokenized3 = [list(jieba.cut(ref)) for ref in [references[2]]]
    candidate_tokenized = list(jieba.cut(candidate))

    # 计算METEOR分数
    score = 0
    score += meteor_score(reference_tokenized1, candidate_tokenized,alpha=alpha,beta=beta,gamma=gamma)
    score += meteor_score(reference_tokenized2, candidate_tokenized,alpha=alpha,beta=beta,gamma=gamma)
    score += meteor_score(reference_tokenized3, candidate_tokenized,alpha=alpha,beta=beta,gamma=gamma)
    score/=3
    return score


def meteor_en(candidate,references,alpha,beta,gamma):

    nltk.download('wordnet')
    nltk.download('punkt')

    reference_tokenized1 = [word_tokenize(ref) for ref in [references[0]]]
    reference_tokenized2 = [word_tokenize(ref) for ref in [references[1]]]
    reference_tokenized3 = [word_tokenize(ref) for ref in [references[2]]]
    candidate_tokenized = word_tokenize(candidate)

    # 计算METEOR分数
    score = 0
    score += meteor_score(reference_tokenized1, candidate_tokenized,alpha=alpha,beta=beta,gamma=gamma)
    score += meteor_score(reference_tokenized2, candidate_tokenized,alpha=alpha,beta=beta,gamma=gamma)
    score += meteor_score(reference_tokenized3, candidate_tokenized,alpha=alpha,beta=beta,gamma=gamma)
    score/=3
    return score

def preprocess_zh(sentence):
    # 使用jieba进行中文分词
    words = jieba.lcut(sentence.lower())
    return words

def preprocess_en(sentence):
    # 分词、词形还原
    words = word_tokenize(sentence.lower())
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    return lemmatized_words

def cosine_similarity(c1, c2):
    # 计算两个词袋向量之间的余弦相似度
    intersection = set(c1.keys()) & set(c2.keys())
    numerator = sum([c1[x] * c2[x] for x in intersection])

    sum1 = sum([c1[x] ** 2 for x in c1.keys()])
    sum2 = sum([c2[x] ** 2 for x in c2.keys()])
    denominator = (sum1 ** 0.5) * (sum2 ** 0.5)

    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator


def cider_zh(candidate, refs):
    # 计算CIDEr评分
    pre_c = preprocess_zh(candidate)
    pre_refs = [preprocess_zh(ref) for ref in refs]

    # 合并所有参考描述和候选描述中的词
    count_c = Counter(pre_c)
    df0 = Counter(pre_refs[0])
    df1 = Counter(pre_refs[1])
    df2 = Counter(pre_refs[2])#统计所有词的文档频率
    all_words = [word for ref in pre_refs for word in ref] + pre_c
    # 计算每个词的 IDF 值
    idf = {word: log((1 + len(refs)) / (min(1,count_c[word]) + min(1,df0[word]) + min(1,df1[word]) + min(1,df2[word])))  for word in all_words}

    # 统计每个参考描述和候选描述中的词频

    c={word: count_c[word] * idf[word] /len(pre_c) for word in all_words}
    s0={word: df0[word] * idf[word] /len(pre_refs[0]) for word in all_words}
    s1={word: df1[word] * idf[word] /len(pre_refs[1]) for word in all_words}
    s2={word: df2[word] * idf[word] /len(pre_refs[2]) for word in all_words}
    score=np.zeros(3)
    mmoodd=np.zeros(4)
    for word in all_words:
        score[0]+=c[word]*s0[word]
        score[1] += c[word] * s1[word]
        score[2] += c[word] * s2[word]
        mmoodd[0]+=c[word]*c[word]
        mmoodd[1]+=s0[word] * s0[word]
        mmoodd[2] += s1[word] * s1[word]
        mmoodd[3] += s2[word] * s2[word]

    mmoodd[0]=math.sqrt(mmoodd[0])
    mmoodd[1] = math.sqrt(mmoodd[1])
    mmoodd[2] = math.sqrt(mmoodd[2])
    mmoodd[3] = math.sqrt(mmoodd[3])
    total_score=0
    for i in range(3):
        if(mmoodd[0]!=0 and mmoodd[1+i]!=0):
            total_score += score[i]/mmoodd[0]/mmoodd[1+i]
    return total_score

# candidate_description = "一只猫坐在窗台上看着外面的鸟儿。"
# reference_descriptions = [
#     "一只猫坐在窗台上。",
#     "猫儿在窗台上观察鸟儿。",
#     "窗台上的猫正在看鸟儿。"
# ]
#
# print(meteor_zh(candidate_description,reference_descriptions,0.9,3.0,0.5))
# print(cider_zh(candidate_description,reference_descriptions))
# print(bleu_zh(candidate_description,reference_descriptions,4,5))

#
# candidate_description = "A cat sitting on a window ledge watching the birds."
# reference_descriptions = [
#     "A cat is sitting on a window ledge.",
#     "A cat observing the birds from the window.",
#     "A cat is watching the birds from the window ledge."
# ]
#
# print(meteor_en(candidate_description,reference_descriptions,0.9,3.0,0.5))
# print(cider_en(candidate_description,reference_descriptions))
# print(bleu_en(candidate_description,reference_descriptions,4,5))
