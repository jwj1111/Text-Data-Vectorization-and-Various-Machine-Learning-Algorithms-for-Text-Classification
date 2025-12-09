'''
结果也可参考step3_evaluation.ipynb
'''
from scipy import sparse
from sklearn.datasets import fetch_20newsgroups
from sklearn.naive_bayes import MultinomialNB,GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report,precision_recall_fscore_support,confusion_matrix
import pandas as pd
import numpy as np

def tag_to_coarse(tags): #粗分类标签映射
    tag_map={0:4,1:0,2:0,3:0,4:0,5:0,6:4,7:1,8:1,9:1,10:1,11:2,12:2,13:2,14:2,15:2,16:3,17:3,18:3,19:3}
    tags_coarse=np.array([tag_map[tag] for tag in tags])
    return tags_coarse

def evaluate_MultinomialNB(train_docs,train_tags,test_docs,test_tags,tagnames): #测试MultinomialNB分类效果，以下测试同理（训练集网格搜索交叉验证得到最佳参数和模型，测试集查看分类得分、宏平均得分、混淆矩阵）
    parameters={'alpha':[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]}
    clf=MultinomialNB()
    GS=GridSearchCV(clf,parameters,cv=10,scoring='f1_macro',n_jobs=-1) #网格搜索十折交叉验证
    dim_nums=[1000,2000,5000,8000,10000] #不同top-k维度，因dim已从高到低排序，可直接切片比较
    scores=[] #保存每种top-k维度的最佳模型的交叉验证得分
    parameter_dicts=[] #保存每种top-k维度的最佳模型的参数
    models=[] #保存每种top-k维度的最佳模型
    for dim_num in dim_nums:
        train_docs_sliced=train_docs[:,:dim_num]
        GS.fit(train_docs_sliced,train_tags)
        scores.append(GS.best_score_)
        parameter_dicts.append(GS.best_params_)
        models.append(GS.best_estimator_)
    best_score=max(scores)
    best_idx=scores.index(best_score)
    best_dim=dim_nums[best_idx]
    best_parameters=parameter_dicts[best_idx]
    report_str=f'''MultinomialNB最佳模型f1 macro(on train): {best_score}\n(alpha: {best_parameters['alpha']}; 特征数量(TopNbyFreq): {best_dim})'''
    print(report_str)
    with open('./data_home/report/parameters_scores.txt','a',encoding='utf-8') as outfile:
        outfile.write(report_str+'\n\n')
    best_model=models[best_idx] #最佳top-k维度的最佳模型（依据得分）
    test_docs_sliced=test_docs[:,:best_dim] #测试集保持与训练输入相同的最佳top-k维度
    predict_tags=best_model.predict(test_docs_sliced)
    classification_report_dict=classification_report(test_tags,predict_tags,target_names=tagnames,output_dict=True) #分类报告
    classification_report_df=pd.DataFrame(classification_report_dict).transpose()
    print(f'''MultinomialNB分类报告：{classification_report_df}''')
    classification_report_df.to_csv('./data_home/report/MultinomialNB_report.csv',encoding='utf-8',float_format='%.2f') #保存分类报告
    macro_prf=precision_recall_fscore_support(test_tags,predict_tags,average='macro')[:3] #宏平均precision、recall、f1
    with open('./data_home/report/clfs_prf.txt','a',encoding='utf-8') as outfile: #保存宏平均precision、recall、f1
        outfile.write(f'''MultinomialNB\t{macro_prf[0]}\t{macro_prf[1]}\t{macro_prf[2]}'''+'\n')
    cm=confusion_matrix(test_tags,predict_tags) #混淆矩阵
    cm_df=pd.DataFrame(cm,index=tagnames,columns=tagnames)
    cm_df.to_csv('./data_home/report/MultinomialNB_cmatrix.csv',encoding='utf-8') #保存混淆矩阵

def evaluate_GaussianNB(train_docs,train_tags,test_docs,test_tags,tagnames):
    parameters={'var_smoothing':[1e-9, 1e-8, 1e-7, 1e-6, 1e-5]}
    clf=GaussianNB()
    GS=GridSearchCV(clf,parameters,cv=10,scoring='f1_macro',n_jobs=-1)
    dim_nums=[1000,2000,5000,8000,10000]
    scores=[]
    parameter_dicts=[]
    models=[]
    for dim_num in dim_nums:
        train_docs_sliced=train_docs[:,:dim_num].toarray()
        GS.fit(train_docs_sliced,train_tags)
        scores.append(GS.best_score_)
        parameter_dicts.append(GS.best_params_)
        models.append(GS.best_estimator_)
    best_score=max(scores)
    best_idx=scores.index(best_score)
    best_dim=dim_nums[best_idx]
    best_parameters=parameter_dicts[best_idx]
    report_str=f'''GaussianNB最佳模型f1 macro(on train): {best_score}\n(var_smoothing: {best_parameters['var_smoothing']}; 特征数量(TopNbyFreq): {best_dim})'''
    print(report_str)
    with open('./data_home/report/parameters_scores.txt','a',encoding='utf-8') as outfile:
        outfile.write(report_str+'\n\n')
    best_model=models[best_idx]
    test_docs_sliced=test_docs[:,:best_dim].toarray()
    predict_tags=best_model.predict(test_docs_sliced)
    classification_report_dict=classification_report(test_tags,predict_tags,target_names=tagnames,output_dict=True)
    classification_report_df=pd.DataFrame(classification_report_dict).transpose()
    print(f'''GaussianNB分类报告：{classification_report_df}''')
    classification_report_df.to_csv('./data_home/report/GaussianNB_report.csv',encoding='utf-8',float_format='%.2f')
    macro_prf=precision_recall_fscore_support(test_tags,predict_tags,average='macro')[:3]
    with open('./data_home/report/clfs_prf.txt','a',encoding='utf-8') as outfile:
        outfile.write(f'''GaussianNB\t{macro_prf[0]}\t{macro_prf[1]}\t{macro_prf[2]}'''+'\n')
    cm=confusion_matrix(test_tags,predict_tags)
    cm_df=pd.DataFrame(cm,index=tagnames,columns=tagnames)
    cm_df.to_csv('./data_home/report/GaussianNB_cmatrix.csv',encoding='utf-8')

def evaluate_DecisionTreeClassifier(train_docs,train_tags,test_docs,test_tags,tagnames):
    parameters={'criterion':["gini", "entropy", "log_loss"]}
    clf=DecisionTreeClassifier(random_state=42) #random_state均设为42
    GS=GridSearchCV(clf,parameters,cv=10,scoring='f1_macro',n_jobs=-1)
    dim_nums=[1000,2000,5000,8000,10000]
    scores=[]
    parameter_dicts=[]
    models=[]
    for dim_num in dim_nums:
        train_docs_sliced=train_docs[:,:dim_num]
        GS.fit(train_docs_sliced,train_tags)
        scores.append(GS.best_score_)
        parameter_dicts.append(GS.best_params_)
        models.append(GS.best_estimator_)
    best_score=max(scores)
    best_idx=scores.index(best_score)
    best_dim=dim_nums[best_idx]
    best_parameters=parameter_dicts[best_idx]
    report_str=f'''DecisionTreeClassifier最佳模型f1 macro(on train): {best_score}\n(criterion: {best_parameters['criterion']}; 特征数量(TopNbyFreq): {best_dim})'''
    print(report_str)
    with open('./data_home/report/parameters_scores.txt','a',encoding='utf-8') as outfile:
        outfile.write(report_str+'\n\n')
    best_model=models[best_idx]
    test_docs_sliced=test_docs[:,:best_dim]
    predict_tags=best_model.predict(test_docs_sliced)
    classification_report_dict=classification_report(test_tags,predict_tags,target_names=tagnames,output_dict=True)
    classification_report_df=pd.DataFrame(classification_report_dict).transpose()
    print(f'''DecisionTreeClassifier分类报告：{classification_report_df}''')
    classification_report_df.to_csv('./data_home/report/DecisionTreeClassifier_report.csv',encoding='utf-8',float_format='%.2f')
    macro_prf=precision_recall_fscore_support(test_tags,predict_tags,average='macro')[:3]
    with open('./data_home/report/clfs_prf.txt','a',encoding='utf-8') as outfile:
        outfile.write(f'''DecisionTreeClassifier\t{macro_prf[0]}\t{macro_prf[1]}\t{macro_prf[2]}'''+'\n')
    cm=confusion_matrix(test_tags,predict_tags)
    cm_df=pd.DataFrame(cm,index=tagnames,columns=tagnames)
    cm_df.to_csv('./data_home/report/DecisionTreeClassifier_cmatrix.csv',encoding='utf-8')

def evaluate_KNN(train_docs,train_tags,test_docs,test_tags,tagnames):
    parameters=[{"n_neighbors":[1,3,5,7],"p":[2],"metric":["minkowski"]},
                {"n_neighbors":[1,3,5,7],"metric":["cosine"]},
                {"n_neighbors":[1,3,5,7],"p":[1],"metric":["minkowski"]}]
    clf=KNeighborsClassifier()
    GS=GridSearchCV(clf,parameters,cv=10,scoring='f1_macro',n_jobs=-1)
    dim_nums=[1000,2000,5000,8000,10000]
    scores=[]
    parameter_dicts=[]
    models=[]
    for dim_num in dim_nums:
        train_docs_sliced=train_docs[:,:dim_num]
        GS.fit(train_docs_sliced,train_tags)
        scores.append(GS.best_score_)
        parameter_dicts.append(GS.best_params_)
        models.append(GS.best_estimator_)
    best_score=max(scores)
    best_idx=scores.index(best_score)
    best_dim=dim_nums[best_idx]
    best_parameters=parameter_dicts[best_idx]
    report_str=f'''KNN最佳模型f1 macro(on train): {best_score}\n(n_neighbors: {best_parameters['n_neighbors']}; metric: {best_parameters['metric']}; p: {best_parameters.get('p','None')}; 特征数量(TopNbyFreq): {best_dim})'''
    print(report_str)
    with open('./data_home/report/parameters_scores.txt','a',encoding='utf-8') as outfile:
        outfile.write(report_str+'\n\n')
    best_model=models[best_idx]
    test_docs_sliced=test_docs[:,:best_dim]
    predict_tags=best_model.predict(test_docs_sliced)
    classification_report_dict=classification_report(test_tags,predict_tags,target_names=tagnames,output_dict=True)
    classification_report_df=pd.DataFrame(classification_report_dict).transpose()
    print(f'''KNN分类报告：{classification_report_df}''')
    classification_report_df.to_csv('./data_home/report/KNN_report.csv',encoding='utf-8',float_format='%.2f')
    macro_prf=precision_recall_fscore_support(test_tags,predict_tags,average='macro')[:3]
    with open('./data_home/report/clfs_prf.txt','a',encoding='utf-8') as outfile:
        outfile.write(f'''KNN\t{macro_prf[0]}\t{macro_prf[1]}\t{macro_prf[2]}'''+'\n')
    cm=confusion_matrix(test_tags,predict_tags)
    cm_df=pd.DataFrame(cm,index=tagnames,columns=tagnames)
    cm_df.to_csv('./data_home/report/KNN_cmatrix.csv',encoding='utf-8')

def evaluate_SVC(train_docs,train_tags,test_docs,test_tags,tagnames):
    parameters={"kernel":['linear','poly'],"degree":[2,3,5]}
    clf=SVC(random_state=42)
    GS=GridSearchCV(clf,parameters,cv=10,scoring='f1_macro',n_jobs=-1)
    dim_nums=[1000,2000,5000,8000,10000]
    scores=[]
    parameter_dicts=[]
    models=[]
    for dim_num in dim_nums:
        train_docs_sliced=train_docs[:,:dim_num]
        GS.fit(train_docs_sliced,train_tags)
        scores.append(GS.best_score_)
        parameter_dicts.append(GS.best_params_)
        models.append(GS.best_estimator_)
    best_score=max(scores)
    best_idx=scores.index(best_score)
    best_dim=dim_nums[best_idx]
    best_parameters=parameter_dicts[best_idx]
    report_str=f'''SVC最佳模型f1 macro(on train): {best_score}\n(kernel: {best_parameters['kernel']}; degree: {best_parameters['degree']}; 特征数量(TopNbyFreq): {best_dim})'''
    print(report_str)
    with open('./data_home/report/parameters_scores.txt','a',encoding='utf-8') as outfile:
        outfile.write(report_str+'\n\n')
    best_model=models[best_idx]
    test_docs_sliced=test_docs[:,:best_dim]
    predict_tags=best_model.predict(test_docs_sliced)
    classification_report_dict=classification_report(test_tags,predict_tags,target_names=tagnames,output_dict=True)
    classification_report_df=pd.DataFrame(classification_report_dict).transpose()
    print(f'''SVC分类报告：{classification_report_df}''')
    classification_report_df.to_csv('./data_home/report/SVC_report.csv',encoding='utf-8',float_format='%.2f')
    macro_prf=precision_recall_fscore_support(test_tags,predict_tags,average='macro')[:3]
    with open('./data_home/report/clfs_prf.txt','a',encoding='utf-8') as outfile:
        outfile.write(f'''SVC\t{macro_prf[0]}\t{macro_prf[1]}\t{macro_prf[2]}'''+'\n')
    cm=confusion_matrix(test_tags,predict_tags)
    cm_df=pd.DataFrame(cm,index=tagnames,columns=tagnames)
    cm_df.to_csv('./data_home/report/SVC_cmatrix.csv',encoding='utf-8')

def evaluate_LogisticRegression(train_docs,train_tags,test_docs,test_tags,tagnames):
    parameters={"tol":[1e-3,1e-4,1e-5],"C":[0.1,0.5,1.0,2.5]}
    clf=LogisticRegression(solver='saga',random_state=42,max_iter=1000) #saga更适合处理大规模文本数据，max_iter为1000确保收敛
    GS=GridSearchCV(clf,parameters,cv=10,scoring='f1_macro',n_jobs=-1)
    dim_nums=[1000,2000,5000,8000,10000]
    scores=[]
    parameter_dicts=[]
    models=[]
    for dim_num in dim_nums:
        train_docs_sliced=train_docs[:,:dim_num]
        GS.fit(train_docs_sliced,train_tags)
        scores.append(GS.best_score_)
        parameter_dicts.append(GS.best_params_)
        models.append(GS.best_estimator_)
    best_score=max(scores)
    best_idx=scores.index(best_score)
    best_dim=dim_nums[best_idx]
    best_parameters=parameter_dicts[best_idx]
    report_str=f'''LogisticRegression最佳模型f1 macro(on train): {best_score}\n(tol: {best_parameters['tol']}; C: {best_parameters['C']}; 特征数量(TopNbyFreq): {best_dim})'''
    print(report_str)
    with open('./data_home/report/parameters_scores.txt','a',encoding='utf-8') as outfile:
        outfile.write(report_str+'\n\n')
    best_model=models[best_idx]
    test_docs_sliced=test_docs[:,:best_dim]
    predict_tags=best_model.predict(test_docs_sliced)
    classification_report_dict=classification_report(test_tags,predict_tags,target_names=tagnames,output_dict=True)
    classification_report_df=pd.DataFrame(classification_report_dict).transpose()
    print(f'''LogisticRegression分类报告：{classification_report_df}''')
    classification_report_df.to_csv('./data_home/report/LogisticRegression_report.csv',encoding='utf-8',float_format='%.2f')
    macro_prf=precision_recall_fscore_support(test_tags,predict_tags,average='macro')[:3]
    with open('./data_home/report/clfs_prf.txt','a',encoding='utf-8') as outfile:
        outfile.write(f'''LogisticRegression\t{macro_prf[0]}\t{macro_prf[1]}\t{macro_prf[2]}'''+'\n')
    cm=confusion_matrix(test_tags,predict_tags)
    cm_df=pd.DataFrame(cm,index=tagnames,columns=tagnames)
    cm_df.to_csv('./data_home/report/LogisticRegression_cmatrix.csv',encoding='utf-8')

def create_clf_macro_report(): #生成宏平均得分报告
    clf_macro_report_df=pd.read_csv('./data_home/report/clfs_prf.txt',sep='\t',index_col=0,header=None,encoding='utf-8')
    clf_macro_report_df.columns=['MacroAvg-Precision','MacroAvg-Recall','MacroAvg-F1']
    clf_macro_report_df.index.name='Classifiers'
    print(clf_macro_report_df)
    clf_macro_report_df.to_csv('./data_home/report/clf_macro_report.csv',encoding='utf-8',float_format='%.4f')

if __name__=='__main__':
    train_docs=sparse.load_npz('./data_home/news_train_docs_vectorized.npz') #加载本地向量
    test_docs=sparse.load_npz('./data_home/news_test_docs_vectorized.npz')
    train,test=fetch_20newsgroups(subset='train',data_home='data_home'),fetch_20newsgroups(subset='test',data_home='data_home')
    train_tags,test_tags=train.target,test.target
    tagnames=test.target_names
    tagnames_coarse=['comp','rec','sci','talk','other']
    print(f'''细分类tagnames为{tagnames}\n粗分类tagnames为{tagnames_coarse}''')
    train_tags_coarse,test_tags_coarse=tag_to_coarse(train_tags),tag_to_coarse(test_tags) #映射为粗分类
    evaluate_MultinomialNB(train_docs,train_tags_coarse,test_docs,test_tags_coarse,tagnames_coarse) #测试模型
    evaluate_GaussianNB(train_docs,train_tags_coarse,test_docs,test_tags_coarse,tagnames_coarse)
    evaluate_DecisionTreeClassifier(train_docs,train_tags_coarse,test_docs,test_tags_coarse,tagnames_coarse)
    evaluate_KNN(train_docs,train_tags_coarse,test_docs,test_tags_coarse,tagnames_coarse)
    evaluate_SVC(train_docs,train_tags_coarse,test_docs,test_tags_coarse,tagnames_coarse)
    evaluate_LogisticRegression(train_docs,train_tags_coarse,test_docs,test_tags_coarse,tagnames_coarse)
    create_clf_macro_report() #生成宏平均得分报告
