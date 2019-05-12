#%%
import re
import datetime
import numpy as np
import pandas as pd
import jieba
from datetime import datetime
from datetime import timedelta
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

#%%
stopwords = []
with open('stopwords.txt', 'r', encoding='UTF-8') as f1:
    for line in f1:
        stopwords = line.split()
stopwords[-1] = ' '

def get_company(company, stock_np, name_index = 0):
    indexs = []
    stock_np = np.array(stock_np)
    for i in range(stock_np.shape[0]):
        if company in stock_np[i][name_index]:
            indexs.append(i)
    return stock_np[indexs,:]

def get_stock_data(file, company_name = '中鋼'):
    stock_data = pd.read_csv(file, encoding='UTF-8')
    stock_data_np = stock_data.values
    stock = get_company(company_name, stock_data_np)
    return stock[::-1,:]

def conv_num(nums):
    for i in range(len(nums)):
        nums[i] = nums[i].replace(',', '')
        nums[i] = float(nums[i])
    return nums

def conv_date(date):
    # date = np.array(date)
    for i in range(len(date)):
        d = re.findall('[0-9]+/[0-9]+/[0-9]+', date[i])[0]
        date[i] = datetime.datetime.strptime(date[i],'%Y/%M/%d').date()
        return date
    
def text_clean(text):
    text = text.lower()
    text = re.sub('<br>', ' ', text)
    text = re.sub('[0-9]+', '', text)
    text = re.sub('[^\w\s]', ' ', text)
    text = re.sub(' +', ' ', text)
    return text

def get_up_down_index(values, n = 2, r = 0.03):
    up = []
    down = []
    not_change = []
	#後三天股價-原股價/原股價 ->漲跌幅度超過5%者
    for i in range(len(values) - n):
        dis  = values[i + n] - values[i]
        if abs(dis)/ values[i] >= r:
            up.append(i) if dis > 0 else down.append(i)
        else:
            not_change.append(i)
    return up, down, not_change

def get_doc_by_date(dates, data,before = 2,  date_index = 4, title_index = 5, content_index = 7):
    docs = []
    index = []
    days_before = []
    for d in dates:
        datetime_object = datetime.strptime(d, '%Y/%m/%d')
        for i in range(before):
            datetime_object = datetime_object - timedelta(days=1)
            days_before.append(datetime_object.strftime('%Y/%m/%d'))
        # yesterday = datetime_object - timedelta(days=1)
        # twoDaysBefore = yesterday - timedelta(days=1)
        # two_days_before.append(yesterday.strftime('%Y/%m/%d'))
        # two_days_before.append(twoDaysBefore.strftime('%Y/%m/%d'))
    # print(days_before)

    cnt = 0
    for r in data:
        if r[date_index][:-9] in days_before:
            temp = '%s %s %s' %(r[title_index], r[title_index], r[content_index])
            docs.append(text_clean(temp))
            index.append(cnt)
        cnt += 1
    return docs, index

def keyword_select(docs, index, kewords):
    selected = []
    cnt = 0
    docs_index = []
    for text in docs:
        for i in range(len(kewords)):
            if kewords[i] in text:
                selected.append(text)
                docs_index.append(index[cnt])
                break
        cnt += 1
    return selected, docs_index

def tfidf_tool(corpus):
    vectorizer = TfidfVectorizer(tokenizer=jieba.cut, analyzer='word', min_df=2, stop_words = stopwords, max_features = 1500, max_df=0.8)
    X = vectorizer.fit_transform(corpus)
    print(X.shape)
    return X, vectorizer

def get_top_feature(X, vectorizer):
    m = X.mean(axis=0).getA().reshape(-1)
    max_index = np.argsort(m)[::-1] 
    tokens = np.array(vectorizer.get_feature_names())
    return tokens[max_index]

def get_count_tool(corpus):
    vectorizer = CountVectorizer(tokenizer=jieba.cut, analyzer='word', min_df=2, stop_words = stopwords)
    X = vectorizer.fit_transform(corpus)
    print(X.shape)
    return X, vectorizer

def chi2(all_vectors, clas_vectors):
    expected = all_vectors.sum(axis=0)*len(clas_vectors)/len(all_vectors)
    clas_value = clas_vectors.sum(axis=0)
    signs = np.where(clas_value-expected, 2)/expected
    chi2_value[signs] *= -1
    return chi2_value

def find_week_month_cross (values, n=4, r = 0.03):
    result_true = []
    result_false = []
    pre_month = values[:20].mean()
    pre_week = values[15:20].mean()
    month = []
    week = []
    for i in range (21,len (values)):
        month_mov = values[i-20:i].mean()
        week_mov = values[i-5:i].mean()  
        month.append(month_mov)
        week.append(week_mov)
        if pre_month > pre_week and month_mov <= week_mov:
            if abs(values[i-n]-values[i])/values[i-n] > r:
                result_true.append (i-n)
        elif pre_month < pre_week and month_mov >= week_mov:
            if abs (values [i-n]-values[i])/values[i-n] > r:
               result_true.append(i-n)
        elif abs (week_mov-month_mov)/min(week_mov, month_mov) < 0.01:
            result_false.append (i-n)
        pre_month = month_mov
        pre_week = week_mov
    return result_true, result_false


#%%[markdown]
## Requirement 1 - key word search
#%%
stock = get_stock_data('2018_stock_data.csv', '中鋼')
news = pd.read_csv('news.csv', encoding= 'big5')
news = np.array(news)
end_value = conv_num(stock[:, 5])
stock_date = stock[:, 1]

#%%
up, down, even = get_up_down_index(end_value, 2, 0.03)
up_date = stock_date[up]
down_date = stock_date[down]
even_date = stock_date[even]
# select important keyword to filter documents
up_docs, up_docs_index =  get_doc_by_date(up_date, news)
down_docs, down_docs_index =  get_doc_by_date(down_date, news)
up_docs, up_docs_index = keyword_select(up_docs, up_docs_index, ['友達', '轉盈','漲','拉抬','外資','旺季','回溫','扭轉','提升'])
down_docs, down_docs_index = keyword_select(down_docs, down_docs_index, ['友達', '衰退', '淡季', '跌', '報價下滑', '虧損'])
even_docs, even_docs_index = get_doc_by_date(even_date, news)
#%%
up_x, up_vec = tfidf_tool(up_docs)
down_x, down_vec = tfidf_tool(down_docs)
even_x, even_vec = tfidf_tool(even_docs)
print("up - feature = " , get_top_feature(up_x, up_vec)[:100])
print("down - feature = ", get_top_feature(even_x, even_vec)[:100])

#%%[markdown]
## Request 3 - Total evaluation (find up down factor)
#%%
docs_x, docs_vector = tfidf_tool(up_docs+ even_docs +down_docs)
X = docs_x.toarray()
Y = []
for i in range(len(up_docs)):
    Y.append(1)
for i in range(len(even_docs)):
    Y.append(0)
for i in range(len(down_docs)):
    Y.append(-1)
Y = np.array(Y)
Z = up_docs_index + even_docs_index + down_docs_index
#%%=====================shuffle data and bind split into test and train=========================
np.random.seed(100)
np.random.shuffle(X) 
np.random.seed(100)
np.random.shuffle(Y)
np.random.seed(100)
np.random.shuffle(Z)
X_train, X_test, y_train, y_test, index_train, index_test = train_test_split(X, Y, Z, test_size=0.2)
#%%
#======================= model training ====================================
up_down_classifier = RandomForestClassifier(n_estimators=5, random_state=1)
up_down_classifier.fit(X_train, y_train)
y_pred = up_down_classifier.predict(X_test)
#===================== model evaluation ======================================
print(confusion_matrix(y_test,y_pred)) 
print(classification_report(y_test,y_pred))  
print(accuracy_score(y_test, y_pred))

#%%
pred_date = []
for i in range(len(y_pred)):
    pred_date.append(news[index_test[i]][4])
pred_dic = {
    'date':pred_date,
    'pred':y_pred,
    'actual':y_test
}
pred_table = pd.DataFrame(pred_dic)
print(pred_table)
print('correct', len(pred_table[pred_table['pred'] == pred_table['actual']]))
print('up correct', len(pred_table[pred_table['pred'] == pred_table['actual'] == 1]))
#==============================try predict 2017==========================
#%%
stock_test = get_stock_data('2017_stock_data.csv', '友達')
end_value_test = conv_num(stock_test[:, 5])
stock_date_test = stock_test[:, 1]
up_test, down_test, even_date = get_up_down_index(end_value_test, 2, 0.03)
up_date_test = stock_date[up_test]
down_date_test = stock_date[down_test]
even_date_test = stock_date[even_test]
# select important keyword to filter documents
up_docs_test, up_docs_test_index =  get_doc_by_date(up_date, news)
down_docs_test, down_docs_test_index =  get_doc_by_date(down_date, news)
up_docs_test, up_docs_test_index = keyword_select(up_docs_test, up_docs_test_index, ['友達', '轉盈','漲','拉抬','外資','旺季','回溫','扭轉','提升'])
down_docs_test = keyword_select(down_docs_test, down_docs_test_index, ['友達', '衰退', '淡季', '跌', '報價下滑', '虧損'])
even_docs_test, even_docs_test_index = get_doc_by_date(even_date, news)

temp = TfidfVectorizer(vocabulary = docs_vector.get_feature_names())
total_test = up_docs_test+even_docs_test+down_docs_test
X_test = temp.fit_transform(test).toarray()
y_test = []
for i in range(len(up_docs_test)):
    y_test.append(1)
for i in range(len(even_docs_test)):
    y_test.append(0)
for i in range(len(down_docs_test)):
    y_test.append(-1)
y_test = np.array(y_test)
np.random.seed(150)
np.random.shuffle(X_test) 
np.random.seed(150)
np.random.shuffle(y_test)




#%%
y_pred = up_down_classifier.predict(X_test)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))







# #=========================== Phase 2 =====================================
#%%[markdown]
## Requirement 2 - cross search
#use function find_week_month_cross(end_value) to get the index that satisfied our expectation
#data have two class: 1. the moving average that has significant change or 2.the moving average that is indifferent
#get the date of both two and find the document in those two days to train the predict model
#%%
sigdif, indif = find_week_month_cross(end_value, 3, 0.03)
sigdif_date = stock_date[sigdif]
indif_date = stock_date[indif]
sigdif_docs = get_doc_by_date(sigdif_date, news)
indif_docs = get_doc_by_date(indif_date, news)

#%%
cross_x, cross_vector = tfidf_tool(sigdif_docs + indif_docs)
X = cross_x.toarray()
Y = []
for i in range(len(sigdif_docs)):
    Y.append(1)
for i in range(len(indif_docs)):
    Y.append(0)
Y = np.array(Y)
np.random.seed(100)
np.random.shuffle(X) 
np.random.seed(100)
np.random.shuffle(Y)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

#%%
sig_x, sig_vec = tfidf_tool(up_docs)
print("significant - feature = " , get_top_feature(sig_x, sig_vec)[:200])
#%%
#======================= model training ====================================
cross_classifier = RandomForestClassifier(n_estimators=10, random_state=0) 
cross_classifier.fit(X_train, y_train)  
y_pred = cross_classifier.predict(X_test) 
#====================== model evaluation ======================================
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))

# #%%
# df = pd.DataFrame(get_top_feature(sig_x, sig_vec))
# df.to_csv("中鋼-股價反轉點關鍵字.csv", encoding='big5')