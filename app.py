# import libs
import pandas as pd
import numpy as np
import sklearn
import lazypredict
from lazypredict.Supervised import LazyClassifier
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV
# import plotly.express as px
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# read data
df=pd.read_csv('train.csv')
cols=df.select_dtypes('object').columns
cols.tolist()

def visualize_dataset():
    px.histogram(df,x='HomePlanet',color='Transported',barmode='group')
    px.histogram(df,x='CryoSleep',color='Transported',barmode='group')
    px.histogram(df,x='Destination',color='Transported',barmode='group')
    px.histogram(df,x='VIP',color='Transported',barmode='group')


def missingvalue(df):
    cols=df.select_dtypes('object').columns
    cols=cols.tolist()
    df['HomePlanet'].fillna(df['HomePlanet'].value_counts().index[0],inplace=True)
    df['CryoSleep'].fillna(df['CryoSleep'].value_counts().index[0],inplace=True)
    df['Destination'].fillna(df['Destination'].value_counts().index[0],inplace=True)
    df['VIP'].fillna(df['VIP'].value_counts().index[0],inplace=True)

    cols1=df.select_dtypes('float64').columns
    cols1=cols1.tolist()
    for i in cols1:
        df[i]=df[i].fillna(df[i].mean())
    return df

def Onehotencoding(df1):
    df1=df1.join(pd.get_dummies(df['HomePlanet'],prefix='HomePlanet',prefix_sep='_'))
    df1=df1.join(pd.get_dummies(df['CryoSleep'],prefix='CryoSleep',prefix_sep='_'))
    df1=df1.join(pd.get_dummies(df['Destination'],prefix='Destination',prefix_sep='_'))
    df1=df1.join(pd.get_dummies(df['VIP'],prefix='VIP',prefix_sep='_'))
    df1.drop(['HomePlanet','CryoSleep','Destination','VIP'],axis=1,inplace=True)
    return df1

def pre_processing(df):
    df.drop(['PassengerId','Name','Cabin'],axis=1,inplace=True)
    df=missingvalue(df)
    cols=df.select_dtypes('object').columns.tolist()
    df=Onehotencoding(df)
    return df

def find_bestmodel(x_train, x_test, y_train, y_test):
    clf=LazyClassifier()
    model,predictions=clf.fit(x_train,x_test,y_train,y_test)
    print(model)
    return model.index[0]

def train_classifier(model, x_train, y_train, x_test, y_test):
    # clf=lgb.LGBMClassifier(random_state=5)
    clf = eval(model)(random_state=5)
    print(clf)
    clf.fit(x_train, y_train)
    # pred = clf.predict(x_test)
    clf.get_params()
    print(clf.score(x_train, y_train))
    print(clf.score(x_test, y_test))
    y_pred_train=clf.predict(x_train)
    print('{0:0.4f}'.format(accuracy_score(y_train, y_pred_train)))
    return clf

def hyperparameter_tuning(clf):
    param_grid={'max_bin':[150,250],'learning_rate':[0.13,0.03],'num_iterations':[150,300],'min_gain_to_split':[0.1,1],'max_depth':[10,20]}
    clf=RandomizedSearchCV(estimator=clf,param_distributions=param_grid, scoring='accuracy')
    search=clf.fit(x_train,y_train)
    print(search.best_params_)
    print(search.best_score_)
    return search.best_params_

def optimize_model(model, best_params, x_train, y_train, x_test):
    clf = eval(model)(**best_params)
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    print("Accuracy on train data",clf.score(x_train,y_train))
    print("Accuracy on test data",clf.score(x_test,y_test))
    return clf, pred


def generate_preds(clf):
    df2 = pd.read_csv('test.csv')
    sub = pd.DataFrame(df2['PassengerId'])
    df3 = pre_processing(df2)
    pred1 = clf.predict(df3)
    sub['Transported'] = pred1
    sub.to_csv('submission.csv',index=False)


if __name__ == '__main__':
    # visualize_dataset()
    df1 = pre_processing(df)
    
    target = df1['Transported'] #y
    col = df1.columns
    col = col.delete(6)
    features = df1[col] #x

    x_train, x_test, y_train, y_test = train_test_split(features, target, random_state=5, stratify=target, test_size=0.40)
    best_model = find_bestmodel(x_train, x_test, y_train, y_test)
    clf = train_classifier(best_model, x_train, y_train, x_test, y_test)
    best_params = hyperparameter_tuning(clf)
    opt_clf, pred = optimize_model(best_model, best_params, x_train, y_train, x_test)

    cm = confusion_matrix(y_test, pred)
    print("Confusion matrix:\n", cm)
    # sns.heatmap(cm,annot=True)
    cr = classification_report(y_test, pred)
    print("Classification report:\n",cr)

    generate_preds(opt_clf)



