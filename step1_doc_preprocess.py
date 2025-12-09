from sklearn.datasets import fetch_20newsgroups
import spacy
from tqdm import tqdm

def txt_preprocess(docs,batchsize=666): #使用spacy预处理
    try:
        print("尝试使用GPU运行spaCy")
        import cupy
        spacy.require_gpu()
        nlp=spacy.load('en_core_web_sm')
        docs=[doc.lower() for doc in docs]
        docs_processed=[]
        for doc in tqdm(nlp.pipe(docs,batchsize),total=len(docs),desc='处理文档(GPU)'):
            tokens=[token.lemma_ for token in doc if not (token.is_stop or token.is_punct or token.is_space)] #去除停用词、标点、空白字符，lemmatization
            docs_processed.append(''' '''.join(tokens))
    except Exception as e:
        print(f"GPU运行失败：{e}\n尝试使用CPU运行spaCy")
        spacy.require_cpu()
        nlp=spacy.load('en_core_web_sm')
        docs=[doc.lower() for doc in docs]
        docs_processed=[]
        for doc in tqdm(nlp.pipe(docs,batchsize),total=len(docs),desc='处理文档(CPU)'):
            tokens=[token.lemma_ for token in doc if not (token.is_stop or token.is_punct or token.is_space)]
            docs_processed.append(''' '''.join(tokens))
    return docs_processed

def save_processed_docs(processed_docs,filepath): #保存预处理文本
    with open(filepath,'w',encoding='utf-8') as outfile:
        for doc in tqdm(processed_docs,total=len(processed_docs),desc=f'''保存至{filepath}'''):
            outfile.write(doc+'\n')


if __name__=='__main__':
    news_train=fetch_20newsgroups(subset='train',data_home='data_home') #在data_home保存数据，便于加载
    news_test=fetch_20newsgroups(subset='test',data_home='data_home')
    news_train_docs,news_test_docs=news_train.data,news_test.data
    print(f'''训练集文档数量：{len(news_train_docs)}\n测试集文档数量：{len(news_test_docs)}''')
    news_train_docs_processed=txt_preprocess(news_train_docs,batchsize=666) #可设置批处理数量
    news_test_docs_processed=txt_preprocess(news_test_docs,batchsize=666)
    path_for_train_docs_processed='./data_home/news_train_docs_processed.txt'
    path_for_test_docs_processed='./data_home/news_test_docs_processed.txt'
    save_processed_docs(news_train_docs_processed,filepath=path_for_train_docs_processed) #保存预处理文本
    save_processed_docs(news_test_docs_processed,filepath=path_for_test_docs_processed)
