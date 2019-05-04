#%%
import re
import datetime
import numpy as np
import pandas as pd
import jieba
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

#%%
stopwords = []
with open('stopwords.txt', 'r', encoding='UTF-8') as f1:
    for line in f1:
        # print(line,"\n")
        # print(line.split())
        stopwords = line.split()
stopwords[-1] = ' '

def get_company(company, stock_np, name_index = 0):
    indexs = []
    stock_np = np.array(stock_np)
    for i in range(stock_np.shape[0]):
        if company in stock_np[i][name_index]:
            indexs.append(i)
    return stock_np[indexs,:]

def get_stock_data(file, company_name = '大立光'):
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

def get_up_down_index(values, n = 3, r = 0.05):
    up = []
    down = []
	#後三天股價-原股價/原股價 ->漲跌幅度超過5%者
    for i in range(len(values) - n):
        dis  = values[i + n] - values[i]
        # print(dis)
        # print(abs(dis)/ values[i])
        if abs(dis)/ values[i] >= r:
            up.append(i) if dis > 0 else down.append(i)
    return up, down

def get_doc_by_date(dates, data, date_index = 4, title_index = 5, content_index = 7):
    docs = []
    # print('dates :', dates[0])
    # print('dates - type :', type(dates[0]))
    for r in data:
        # print('r[date_index] :', r[date_index])
        # print('r[date_index] :', type(r[date_index]))
        if r[date_index][:-9] in dates:
            # print('r[date_index] :', r[date_index][:-9])
            temp = '%s %s %s' %(r[title_index], r[title_index], r[content_index])
            docs.append(text_clean(temp))
    return docs

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

def find_week_month_cross (values, n=3, r = 0.03):
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
stock = get_stock_data('2016_stock_data.csv', '大立光')
news = pd.read_csv('news.csv', encoding= 'big5')
news = np.array(news)
end_value = conv_num(stock[:, 5])
stock_date = stock[:, 1]
# stock_date = conv_date(stock[:, 1])
#%%
up, down = get_up_down_index(end_value)
up_date = stock_date[up]
down_date = stock_date[down]

up_docs = get_doc_by_date(up_date, news)
down_docs = get_doc_by_date(down_date, news)

#%%[markdown]
## Request 3 - Total evaluation (find up down factor)
#%%
docs_x, docs_vector = tfidf_tool(up_docs+down_docs)
X = docs_x.toarray()
# X = np.concatenate((up_x, down_x), axis=0)
Y = []
for i in range(len(up_x)):
    Y.append(1)
for i in range(len(down_x)):
    Y.append(0)
Y = np.array(Y)
#%%
np.random.seed(100)
np.random.shuffle(X) 
np.random.seed(100)
np.random.shuffle(Y)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)

#%%
#======================= model training ====================================
from sklearn.ensemble import RandomForestClassifier
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)  
classifier1 = RandomForestClassifier(n_estimators=5, random_state=1)  
classifier1.fit(X_train, y_train)  
y_pred = classifier1.predict(X_test) 
#====================== model evaluation ======================================
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(confusion_matrix(y_test,y_pred))  
print(classification_report(y_test,y_pred))  
print(accuracy_score(y_test, y_pred))




#%%[markdown]
## Requirement 2 - cross search

#%%
sigdif, indif = find_week_month_cross(end_value)
sigdif_date = stock_date[sigdif]
indif_date = stock_date[indif]
sigdif_docs = get_doc_by_date(sigdif_date, news)
indif_docs = get_doc_by_date(indif_date, news)

#%%
cross_x, cross_vector = tfidf_tool(sigdif_docs + indif_docs)
X = cross_x.toarray()
# X = np.concatenate((sig_x, indif_x), axis=0)
Y = []
for i in range(len(sig_x)):
    Y.append(1)
for i in range(len(indif_x)):
    Y.append(0)
Y = np.array(Y)
np.random.seed(100)
np.random.shuffle(X) 
np.random.seed(100)
np.random.shuffle(Y)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
#%%
#%%
#======================= model training ====================================
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)  
classifier2 = RandomForestClassifier(n_estimators=5, random_state=0)  
classifier2.fit(X_train, y_train)  
y_pred = classifier2.predict(X_test) 
#====================== model evaluation ======================================
print(confusion_matrix(y_test,y_pred))  
print(classification_report(y_test,y_pred))  
print(accuracy_score(y_test, y_pred))
