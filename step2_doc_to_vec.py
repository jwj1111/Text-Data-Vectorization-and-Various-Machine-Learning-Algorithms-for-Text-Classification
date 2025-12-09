import math
import numpy as np
from collections import Counter
from scipy import sparse
from tqdm import tqdm

def load_processed_docs(filepath): #加载预处理好的文本
    processed_docs=[]
    with open(filepath,'r',encoding='utf-8') as infile:
        for doc in infile:
            processed_docs.append(doc.strip())
    return processed_docs

def create_vocab_dict(docs,freq_threshold=0): #基于预处理后训练集创建vocab{token：idx}，可筛选词频
    docs_tokenized=[doc.split(''' ''') for doc in docs]
    token_freq_pairs=Counter(token for doc_tokenized in docs_tokenized for token in doc_tokenized).most_common() #按词频高到低排列，便于后续top-K切片
    vocab_list=[token for token,freq in token_freq_pairs if freq>=freq_threshold]
    vocab_dict={token:idx for idx,token in enumerate(vocab_list)}
    size_og,size_filtered=len(token_freq_pairs),len(vocab_dict)
    print(f'''原vocab大小为{size_og}\n词频大于等于{freq_threshold}时vocab大小为{size_filtered}（占比{(size_filtered/size_og)*100:.2f}%）''')
    return vocab_dict

def tfidf_doc2vec(docs,vocab): #文档向量化
    docs_tokenized=[doc.split(''' ''') for doc in docs]
    idf_array=cal_idf_array(docs_tokenized,vocab) #计算idf array
    tf_matrix=cal_tf_matrix(docs_tokenized,vocab) #计算tf matrix
    print("生成tfidf_matrix")
    tfidf_matrix=tf_matrix.multiply(idf_array).tocsr() #生成tfidf matrix
    print("tfidf_matrix已生成")
    tfidf_matrix_normalized=l2_normalize(tfidf_matrix) #l2归一化
    return tfidf_matrix_normalized

def cal_idf_array(docs_tokenized,vocab): #计算idf array
    docs_num,vocab_size=len(docs_tokenized),len(vocab)
    idf_array=np.zeros(vocab_size)
    tkn_docnum_count=Counter(token for doc_tokenized in docs_tokenized for token in set(doc_tokenized) if token in vocab)
    for token_vocab,idx in tqdm(vocab.items(),total=vocab_size,desc='计算idf数组'):
        docs_with_tkn_num=tkn_docnum_count.get(token_vocab,0)
        idf_token=1+math.log(docs_num/(1+docs_with_tkn_num))
        idf_array[idx]=idf_token
    return idf_array

def cal_tf_matrix(docs_tokenized,vocab): #计算tf matrix
    docs_num,vocab_size=len(docs_tokenized),len(vocab)
    rows=[]
    columns=[]
    values=[]
    for idx,doc_tokenized in tqdm(enumerate(docs_tokenized),total=docs_num,desc='计算tf稀疏矩阵'):
        word_freq=Counter(token for token in doc_tokenized if token in vocab)
        for token,freq in word_freq.items():
            rows.append(idx)
            columns.append(vocab[token])
            values.append(freq)
    tf_matrix=sparse.csr_matrix((values,(rows,columns)),shape=(docs_num,vocab_size))
    return tf_matrix

def l2_normalize(tfidf_matrix): #计算l2归一化
    print("tfidf_matrix归一化(L2)")
    norms=np.sqrt(tfidf_matrix.multiply(tfidf_matrix).sum(axis=1))
    norms[norms==0]=1
    tfidf_matrix_normalized=tfidf_matrix.multiply(1/norms).tocsr()
    print("tfidf_matrix归一化(L2)已完成")
    return tfidf_matrix_normalized

if __name__=='__main__':
    path_for_train_docs_processed='./data_home/news_train_docs_processed.txt'
    path_for_test_docs_processed='./data_home/news_test_docs_processed.txt'
    train_docs=load_processed_docs(path_for_train_docs_processed) #加载预处理好的文本
    test_docs=load_processed_docs(path_for_test_docs_processed)
    vocab_dict=create_vocab_dict(train_docs,freq_threshold=5) #基于预处理后训练集创建vocab{token：idx}，可筛选词频
    train_vectorized=tfidf_doc2vec(docs=train_docs,vocab=vocab_dict) #文档向量化
    test_vectorized=tfidf_doc2vec(docs=test_docs,vocab=vocab_dict)
    path_for_train_docs_vectorized='./data_home/news_train_docs_vectorized.npz'
    path_for_test_docs_vectorized='./data_home/news_test_docs_vectorized.npz'
    sparse.save_npz(path_for_train_docs_vectorized,train_vectorized) #保存向量化结果
    print(f'''train_vectorized保存至路径:{path_for_train_docs_vectorized}''')
    sparse.save_npz(path_for_test_docs_vectorized,test_vectorized)
    print(f'''test_vectorized保存至路径:{path_for_test_docs_vectorized}''')
