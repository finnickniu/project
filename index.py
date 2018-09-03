from __future__ import print_function
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import os
from keras.callbacks import EarlyStopping
from flask import *
from werkzeug.utils import secure_filename
import codecs
from statsmodels.tsa.arima_model import ARIMA
import re
import numpy
import pandas as pd
from pandas import *
import csv
import time
import math
from keras.models import Sequential
from keras.layers import *
from matplotlib import pyplot
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from keras import optimizers
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import random
import io
from io import *
from flask import Flask, make_response
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import base64


UPLOAD_FOLDER = '/Users/ushisensei/datalake/trafficdata'
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = set(['csv', 'pdf','png', 'jpg', 'xls', 'JPG', 'PNG', 'xlsx', 'gif', 'GIF'])
path='/Users/ushisensei/datalake/'
pathproject='/Users/ushisensei/PycharmProjects/Arima/'
#Scripping
def matches(date):
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--disable-gpu')
    driver = webdriver.Chrome(executable_path='/usr/local/bin/chromedriver', chrome_options=chrome_options)
    times = date
    # a='2015/3/'
    url="http://www.worldfootball.net/teams/manchester-united/" + times + "/3/"
    driver.get("http://www.worldfootball.net/teams/manchester-united/" + times + "/3/")
    html = driver.page_source
    soup = BeautifulSoup(html, 'lxml')  # creat soup

    matches = soup.find_all(name='a', attrs={"href": re.compile(r'.\w{13}.\d{4}.\w{3}.\d.')})  # match date
    matches1 = soup.find_all(name='a', attrs={
        "href": re.compile(r'.(teams).\w*(-)\w*(-)?(\w*)(/)' + times + '(/3/)')})  # match team
    matches4 = soup.find_all('td')  # match place
    # match palce
    tag = []
    for a in matches4:
        tag1 = re.findall(r'(<td)\s(align="center")\s(class="hell">|class="dunkel">)(H|N|A)(</td>)', str(a))
        if len(tag1):
            tag.append(tag1)

    tag2 = []
    tag11 = re.sub(r'(\'|,)', '', str(tag))
    soup2 = BeautifulSoup(tag11, 'lxml')
    matches5 = soup2.find_all('td')
    for f in matches5:
        if len(f):
            tag2.append(f.string)

    team = []
    date = []
    # match date
    for i in matches:
        date.append(i.string)
    # match team
    for k in matches1:
        team.append(k.string)
    for j in range(10000):
        # "/Users/ushisensei/Desktop/matchesdata/"
        csvFile = open(path+times + "footballmatch.csv", "w+")
        del (team[0])
        try:
            writer = csv.writer(csvFile)
            writer.writerow(('date', 'place', 'team'))
            for j in range(len(team)):
                writer.writerow((date[j], tag2[j], team[j]))



        finally:
            csvFile.close()
        break

    xd = pd.read_csv(path+times + "footballmatch.csv")
    xd.to_excel(path+times + "footballmatch.xlsx", sheet_name='data')
    xd2 = pd.ExcelFile(path+times + "footballmatch.xlsx")
    df = xd2.parse()
    with codecs.open(pathproject+"templates/"+times + 'footballmatch.html', 'w',
                     'utf-8') as html_file:
        html_file.write(df.to_html(header=True, index=False))

    return url

    #Arima


def loaddata(i, times, result,total):

    week = (datetime.strptime(times, "%Y-%m-%d").weekday()) + 1
    print(times, i)
    p_values = range(0, 4)
    d_values = range(0, 3)
    q_values = range(0, 3)
    warnings.filterwarnings("ignore")

    # data prepare
    # train data
    a = i - 1
    b = i + 1
    c = i + 2
    if i < 10:
        i = str('0' + str(i))
    if a < 10:
        a = str('0' + str(a))
    if b < 10:
        b = str('0' + str(b))
    if c < 10:
        c = str('0' + str(c))
    data12 = pd.read_csv(path+'alldata2.csv')
    df = pd.DataFrame(data12)
    if int(week) == 1:
        data13 = df.loc[
            df['date'].isin(['2013-11-04', '2013-11-11', '2013-11-18', '2013-11-25', '2014-12-29', '2016-11-07',
                             '2016-11-28', times])]
        df1 = pd.DataFrame(data13)
        data14 = df1[df['LaneDescription'] == 'NB_NS']
        df2 = pd.DataFrame(data14)
        data15 = df2.loc[
            df['hours'].isin([str(a) + ':00:00', str(i) + ':00:00', str(b) + ':00:00', str(c) + ':00:00'])]
        pd.DataFrame(data15).to_csv(path+'Tuetest.csv', index=None)
    elif int(week) == 2:
        data13 = df.loc[
            df['date'].isin(['2013-11-12', '2013-11-19', '2013-11-26', '2013-12-03', '2013-12-17', '2014-11-15',
                             '2014-11-22', times])]
        df1 = pd.DataFrame(data13)
        data14 = df1[df['LaneDescription'] == 'NB_OS']
        df2 = pd.DataFrame(data14)
        data15 = df2.loc[
            df['hours'].isin([str(a) + ':00:00', str(i) + ':00:00', str(b) + ':00:00', str(c) + ':00:00'])]
        pd.DataFrame(data15).to_csv(path+'Tuetest.csv', index=None)
    elif int(week) == 3:
        data13 = df.loc[
            df['date'].isin(['2013-11-13', '2013-11-20', '2013-11-27', '2015-11-04', '2016-11-02', '2016-11-23',
                             '2016-12-07', times])]
        df1 = pd.DataFrame(data13)
        data14 = df1[df['LaneDescription'] == 'NB_OS']
        df2 = pd.DataFrame(data14)
        data15 = df2.loc[
            df['hours'].isin([str(a) + ':00:00', str(i) + ':00:00', str(b) + ':00:00', str(c) + ':00:00'])]
        pd.DataFrame(data15).to_csv(path+'Tuetest.csv', index=None)
    elif int(week) == 4:
        data13 = df.loc[
            df['date'].isin(['2013-11-14', '2013-11-21', '2013-12-12', '2015-11-05', '2016-12-01', '2012-11-08',
                             '2012-11-15', times])]
        df1 = pd.DataFrame(data13)
        data14 = df1[df['LaneDescription'] == 'NB_OS']
        df2 = pd.DataFrame(data14)
        data15 = df2.loc[
            df['hours'].isin([str(a) + ':00:00', str(i) + ':00:00', str(b) + ':00:00', str(c) + ':00:00'])]
        pd.DataFrame(data15).to_csv(path+'Tuetest.csv', index=None)

    elif int(week) == 5:
        data13 = df.loc[
            df['date'].isin(['2013-11-15', '2013-11-22', '2013-12-13', '2015-11-06', '2012-11-23', '2012-11-09',
                             '2012-11-16', times])]
        df1 = pd.DataFrame(data13)
        data14 = df1[df['LaneDescription'] == 'NB_OS']
        df2 = pd.DataFrame(data14)
        data15 = df2.loc[
            df['hours'].isin([str(a) + ':00:00', str(i) + ':00:00', str(b) + ':00:00', str(c) + ':00:00'])]
        pd.DataFrame(data15).to_csv(path+'Tuetest.csv', index=None)
    elif int(week) == 6:
        data13 = df.loc[
            df['date'].isin(['2012-11-10', '2012-11-17', '2013-11-02', '2013-11-09', '2013-11-16', '2015-11-21',
                             '2015-11-28', times])]
        df1 = pd.DataFrame(data13)
        data14 = df1[df['LaneDescription'] == 'NB_OS']
        df2 = pd.DataFrame(data14)
        data15 = df2.loc[
            df['hours'].isin([str(a) + ':00:00', str(i) + ':00:00', str(b) + ':00:00', str(c) + ':00:00'])]
        pd.DataFrame(data15).to_csv(path+'Tuetest.csv', index=None)
    elif int(week) == 7:
        data13 = df.loc[
            df['date'].isin(['2012-12-02', '2012-11-18', '2013-11-03', '2015-12-06', '2013-11-24', '2015-11-22',
                             '2015-11-29', times])]
        df1 = pd.DataFrame(data13)
        data14 = df1[df['LaneDescription'] == 'NB_OS']
        df2 = pd.DataFrame(data14)
        data15 = df2.loc[
            df['hours'].isin([str(a) + ':00:00', str(i) + ':00:00', str(b) + ':00:00', str(c) + ':00:00'])]
        pd.DataFrame(data15).to_csv(path+'Tuetest.csv', index=None)
    # test data
    data16 = df[df['date'] == times]
    df3 = pd.DataFrame(data16)
    data17 = df3[df['LaneDescription'] == 'NB_OS']
    df4 = pd.DataFrame(data17)
    data18 = df4.loc[df['hours'].isin([str(a) + ':00:00', str(i) + ':00:00', str(b) + ':00:00', str(c) + ':00:00'])]

    pd.DataFrame(data18).to_csv(path+'Tuetest.csv', index=None)
    #
    #
    # #process data
    dateparse = lambda dates: pd.datetime.strptime(dates, "%Y-%m-%d %H:%M:%S")
    indata2 = pd.read_csv(path+'Tuetest.csv', encoding='utf-8', index_col='Sdate', date_parser=dateparse)  # single data

    date = indata2['date'][0]
    ts = indata2['Volume']
    total.append(ts)
    tsf = ts.diff(1)
    #
    # plt.plot(tsf, 'x-', color='blue')
    # plt.show()
    ts_log = np.log(ts)  # normalize

    return evaluate_models(ts_log, p_values, d_values, q_values, i, c, result, times)

# test model error
def evaluate_arima_model(X, arima_order):
    X = X.values
    X1 = X.astype('float32')
    train, test = X1[0:-4], X1[-4:]
    history = [X1 for X1 in train]
    predictions = list()
    for t in range(len(test)):
        model = ARIMA(history, order=arima_order)
        model_fit = model.fit(disp=0, start_ar_lags=13, method='mle')
        output = model_fit.forecast()[0]
        # yhat = output[0]
        predictions.append(output)
        # obs = test[t]
        history.append(test[t])
        # print('predicted=%f, expected=%f' % (yhat, obs))

    error = mean_squared_error(test, predictions)
    print('Test MSE: %.3f' % error)
    return error


# find the best model
def evaluate_models(dataset, p_values, d_values, q_values, i, c, result, times):
    dataset = dataset.astype('float32')
    best_score, best_cfg = float("inf"), None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p, d, q)
                try:
                    mse = evaluate_arima_model(dataset, order)
                    if mse < best_score:
                        best_score, best_cfg = mse, order
                    print('ARIMA%s MSE=%.3f' % (order, mse))
                except:
                    continue

    print('Best ARIMA%s MSE=%.3f' % (best_cfg, best_score))
    return plot(dataset, best_cfg, i, c, result, times)


# plot
def plot(X, best_cfg, i, c, result, times):
    mse=[]

    X = X.values
    X1 = X.astype('float32')
    train, test = X1[0:-4], X1[-4:]
    history = [X1 for X1 in train]
    predictions = list()
    for t in range(len(test)):
        model = ARIMA(history, order=best_cfg)
        model_fit = model.fit(disp=0, start_ar_lags=16, method='mle')
        output = model_fit.forecast()[0]
        # yhat = output[0]
        predictions.append(output)
        # obs = test[t]
        history.append(test[t])
    # print('predicted=%f, expected=%f' % (yhat, obs))
    error = mean_squared_error(test, predictions)
    mse.append(error)

    csvFile = open(path+"outliers.csv", "w+")
    try:
        writer = csv.writer(csvFile)
        writer.writerow(('prediction', 'test'))
        for j in range(len(predictions)):
            writer.writerow((float(predictions[j]), float(test[j])))
    finally:
        csvFile.close()
    # plot
    # dataset = pd.read_csv("outliers.csv")
    # train1= dataset['prediction']
    # test1 = dataset['test']
    # sd = []
    # # boxplot define threshold
    # for i in range(len(train1)):
    #     dist = np.linalg.norm(train1[i] - test1[i])
    #     sd.append(dist)
    # q3 = np.percentile(sd, 75)
    # q1 = np.percentile(sd, 25)
    # iqr = abs(q3 - q1)
    # threshold = q3 + iqr
    # threshold1 = q3 - 1.5*iqr
    # axis=list(range(len(test)))
    # y1axis=[]
    # y2axis=[]
    # for y1 in range(len(axis)):
    #     y1axis.append(threshold)
    # for y2 in range(len(axis)):
    #     y2axis.append(threshold1)
    #

    return outliers(i, c, result, times,mse)


def outliers(v, n, result, times,mse):
    dateparse = lambda dates: pd.datetime.strptime(dates, "%d/%m/%Y %H:%M:%S").strftime("%Y-%m-%d %H:%M:%S")
    Testset = pd.read_csv(path+'Test.csv', encoding='utf-8')  # single data
    matches = pd.read_csv(path+'allmatches.csv', encoding='utf-8', index_col='Sdate', date_parser=dateparse)
    dataset = pd.read_csv(path+'outliers.csv')
    train = dataset['prediction']
    test = dataset['test']
    number = []
    number1 = []
    number2 = []
    time = []
    sd = []
    out = []
    outliers = []
    outliersx = []
    outliersy = []
    a = ''
    Testset = pd.DataFrame(Testset).reset_index(drop=True)
    # boxplot define threshold
    for i in range(len(train)):
        dist = np.linalg.norm(train[i] - test[i])
        sd.append(dist)
    q3 = np.percentile(sd, 75)
    q1 = np.percentile(sd, 25)
    iqr = abs(q3 - q1)
    # threshold = q3 + 1.5*iqr =0.2143450973426942
    threshold = q3 + iqr
    threshold1 = q1 - 1.5 * iqr
    axis = list(range(len(test)))
    y1axis = []
    y2axis = []
    for y1 in range(len(axis)):
        y1axis.append(threshold)
    for y2 in range(len(axis)):
        y2axis.append(threshold1)

    # count outliers numbers
    for s in range(len(train)):
        dist1 = np.linalg.norm(train[s] - test[s])
        if dist1 > threshold:
            number2.append(s)
            print(s)
            outliers.append(Testset['Sdate'][s])
            number.append(s)

    for j in range(len(train)):
        dist1 = np.linalg.norm(train[j] - test[j])

        if dist1 > threshold:
            for k in range(len(matches)):
                # match print
                mindex = str(matches.index[k])
                mindex1 = re.sub(r'(\s\d{2}:00:00)', '', mindex)
                if mindex == Testset['Sdate'][j]:
                    if matches['place'][k] == ' H ':
                        try:
                            outliers.remove(Testset['Sdate'][j])
                            number.remove(int(j))
                        except:
                            print("Get AError")

                        a = 'There was a football match in Old Trafford on ' + mindex + ' Man utd vs ' + \
                            matches['team'][k]
                        time.append(mindex)
                        result.append(a)
                        outliersx.append(j)

                elif j + 1 < len(train):

                    if mindex1 == Testset['date'][j] and matches['hours'][k] == Testset['hours'][j + 1]:
                        if matches['place'][k] == ' H ':
                            try:
                                outliers.remove(Testset['Sdate'][j])
                                number.remove(int(j))
                            except:
                                print("Get AError")
                            a = 'There was a football match in Old Trafford on ' + mindex + ' Man utd vs ' + \
                                matches['team'][k]
                            outliersx.append(j)
                            result.append(a)





    for oty in outliersx:
        outliersy.append(sd[oty])

    if len(time):
        plt.plot(axis, test, '+-', label='Test')
        plt.plot(axis, train, 'x-', color='red', label='Prediction at ' + str(time[0]))
        plt.legend()
        plt.savefig(path + str(time[0]) + ' fitting.png')
        plt.gcf().clear()
        plt.clf()
        plt.plot(axis, y1axis, color='red', label='Threshold')
        plt.plot(axis, y2axis, color='red', label='Threshold1')
        plt.plot(axis, sd, color='blue', label='The fluctuation of distance at ' + str(time[0]))
        plt.scatter(outliersx, outliersy, color='orange', label='outliers by ftm at ' + str(time[0]))
        plt.legend()
        plt.savefig(path+ str(time[0]) + ' detection.png')
        plt.gcf().clear()
        plt.clf()

    return result,mse


# # @app.route('/arima/', methods=['GET'])
def main(i, times, result,total):
    return loaddata(i, times, result,total)

#


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    """
    Frame a time series as a supervised learning dataset.
    Arguments:
        data: Sequence of observations as a list or NumPy array.
        n_in: Number of lag observations as input (X).
        n_out: Number of observations as output (y).
        dropnan: Boolean whether or not to drop rows with NaN values.
    Returns:
        Pandas DataFrame of series framed for supervised learning.
    """
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

# inverse scaling for a forecasted value
def invert_scale(scaler, X, value):
    new_row = [x for x in X] + [value]
    array = numpy.array(new_row)
    array = array.reshape(1, len(array))
    inverted = scaler.inverse_transform(array)
    return inverted[0, -1]



# fix random seed for reproducibility
def evaluate(model, raw_data, scaled_dataset, scaler):
    # separate

    X, y = scaled_dataset[:,0:2], scaled_dataset[:,-1]
    X1=scaled_dataset[:,1:2]
    # reshape
    reshaped = X.reshape(len(X), 1, 2)
    # forecast dataset
    output = model.predict(reshaped, batch_size=1)

    # invert data transforms on forecast
    predictions = list()
    for i in range(len(output)):
        yhat = output[i,0]

        # invert scaling
        yhat = invert_scale(scaler, X1[i], yhat)
        # store forecast
        predictions.append(yhat)
    # report performance



    rmse = math.sqrt(mean_squared_error(raw_data, predictions[:]))

    # score = model.evaluate(raw_data, predictions, verbose=0)
    # print('Test loss:', score[0])
    # print('Test accuracy:', score[1])
    return rmse

def fit_lstm(train, test, dataset, scaler, batch_size, epochs, neurons,date,class1,raw1,result, outlierdate):

    train_rmse, test_rmse = list(), list()
    X1, y = train[:, 0:2],train[:, -1]
    X = X1.reshape(X1.shape[0], 1, 2)
    model = Sequential()
    model.add(LSTM(neurons, activation='sigmoid', dropout=0.02,input_shape=( X.shape[1], X.shape[2])))
    # model.add(LSTM(neurons, activation='sigmoid'))
    model.add(Dense(1))
    #invent gradient clipping by using sgd
    # sgd = optimizers.SGD(lr=0.01, clipvalue=0.5)
    Adam=optimizers.Adam(lr=0.00004, beta_1=0.9, beta_2=0.999,decay=0.000001, epsilon=None)
    model.compile(loss='mean_squared_error', optimizer=Adam)
    early_stopping = EarlyStopping(monitor='val_loss', patience=2)
    #fit model
    error_scores = list()
    for i in range(epochs):
        print('epoches'+str(i))
    # evaluate model on train data
        model.fit(X, y, epochs=1, batch_size=batch_size,verbose=1,callbacks= [early_stopping])
        # model.reset_states()

        train_rmse.append(evaluate(model, dataset[:3936], train, scaler))
        # model.reset_states()
        # evaluate model on test data
        raw_test = dataset[5400:-1]
        raw_test =  raw_test.reset_index(drop=True)
        test_rmse.append(evaluate(model, raw_test, test, scaler))
        # model.reset_states()

    history = DataFrame()
    history['train'], history['test'] = train_rmse, test_rmse




        # model.predict(X,batch_size=batch_size)

    # make predictions
    #get date index
    #
    raw=pd.DataFrame(dataset[5400:-1])
    raw = raw.reset_index(drop=True)
    X, y =  test[:,0:2],  test[:,-1]
    X1= test[:,1:2]
    # reshape
    reshaped = X.reshape(len(X), 1, 2)
    # forecast dataset
    output = model.predict(reshaped, batch_size=1)

    # invert data transforms on forecast
    testPredict = list()
    expect=[]
    expect1= []
    error=[]
    for i in range(1,len(output)):
        yhat = output[i,0]
        expect1.append(yhat)
        # invert scaling
        yhat = invert_scale(scaler, X1[i], yhat)
        # store forecast
        testPredict.append(yhat)
        expected =raw1[i]
        expect.append(expected)
        print('Time=%s, Predicted=%f, Expected=%f' % (date[5400:][i], yhat, expected))
        dist = np.linalg.norm(yhat - expected)
        error.append(dist)


    # report performance
    rmse = math.sqrt(mean_squared_error(X1[:-1], expect1))


    #
    #
    # pyplot.plot(  X1,color='blue')
    # pyplot.plot(predictions, color='red')
    # pyplot.savefig('test.png')
    # pyplot.gcf().clear()
    # pyplot.clf()


    #oulier detection
    # error=[]
    # for k in range(0,len(testPredict)):
    #     dist = np.linalg.norm(testPredict[k]-expect[k])
    #     error.append(dist)
    # error=pd.read_csv('error1.csv', usecols=[1]).values
    pd.DataFrame(error).to_csv('/datalake/error1.csv',header=None)
    q3 = np.percentile(error, 75)
    q1 = np.percentile(error, 25)
    iqr = abs(q3 - q1)
    array=numpy.array(error)
    threshold3 = q3 +1.5*iqr
    print('threshold3',threshold3)

    threshold1 = 120
    print(threshold1)
    threshold2 = q1 -1.5*iqr
    linet=list(range(len(error)))
    linet1=[]
    linet2=[]
    for t1 in linet:
        t1=threshold1
        linet1.append(t1)

    for t2 in linet:
        t2 = threshold2
        linet2.append(t2)

    outliery=[]
    outlierx = []


    for value in range(len(error)):
        if error[value] > threshold1 or error[value]<threshold2:
            outliery.append(error[value])
            outlierx.append(value)

    outmy=[]
    outmx=[]
    dfo=pd.DataFrame(class1[5400:])
    dfo1 = dfo.reset_index(drop=True)
    dataframe = pd.read_csv('/datalake/lstmdataset1.csv')
    df = dataframe['Sdate'][5400:].values



    for value in range(len(error)):
        if error[value] > threshold1 or error[value] < threshold2:
            if dfo1.values[value+2]==1:
                outmy .append(error[value])
                outmx .append(value)
                result.append(value)
                outlierdate.append(df[value])

            # gaussian

    # # lstm
    pyplot.plot(expect,color='blue',label='Test set')
    pyplot.plot(testPredict, color='red',label='Prediction')
    pyplot.legend()
    pyplot.savefig(pathproject+'static/Lstmfiiting.png')
    pyplot.gcf().clear()
    pyplot.clf()

    #
    #outlier
    pyplot.plot(linet1,color='red')
    pyplot.plot(linet2,color='red')
    pyplot.plot(error,color='blue')
    pyplot.scatter(outlierx,outliery,label='outliers')
    pyplot.scatter(outmx,outmy,color='red',label='outliers by ftm')
    pyplot.title('Prediction Error Distribution')

    pyplot.legend()
    pyplot.savefig(pathproject+'static/Lstmdetecting.png')
    pyplot.gcf().clear()
    pyplot.clf()
    return history,result, rmse




@app.route('/lstm2',methods=['POST'])
def lstmmatches1():
    return render_template('lstm.html')

@app.route('/lstm1',methods=['POST'])
def lstmmatches():
    if request.method=='POST':
        neurons = int(request.form['neurons'])
        # 65 950
        batch_size = int(request.form['batch'])
        repeats = 1
        epochs = int(request.form['epochs'])
        learningrate = float(request.form['learningrate'])
        dateparse = lambda dates: pd.datetime.strptime(dates, "%Y-%m-%d %H:%M:%S").strftime("%Y-%m-%d %H:%M:%S")
        # dataset1 = pd.read_csv('allnove&dec1.csv', encoding='utf-8', index_col='Sdate', date_parser=dateparse)
        week = []
        dataset = pd.read_csv(path+'allnove&dec1.csv')
        dataset1 = dataset['Sdate'].str.split('\s', expand=True)
        pd.DataFrame(dataset1).to_csv(path+'date&hours.csv', index=None, header=['date', 'hours'])
        dataset2 = pd.read_csv(path+'date&hours.csv')
        dataset3 = pd.concat([dataset, dataset2], axis=1)
        pd.DataFrame(dataset3).to_csv(path+'lstmvalidation.csv', index=None)

        dataset4 = pd.read_csv(path+'lstmvalidation.csv')
        dataset11 = pd.read_csv(path+'allnove&dec1.csv')
        date = dataset11['Sdate'].values
        class1 = dataset11['class']

        for i in range(len(dataset4)):
            week1 = datetime.strptime(str(dataset4['date'][i]), "%Y-%m-%d").weekday()
            week.append(week1)
        csvFile = open(path+'week.csv', "w+")
        try:
            writer = csv.writer(csvFile)
            writer.writerow(('weekday',))
            for j in range(len(week)):
                writer.writerow(str(week[j] + 1))
        finally:
            csvFile.close()
        weekday = pd.read_csv(path+'week.csv')
        alldata4 = pd.concat([dataset4, weekday], axis=1)
        pd.DataFrame(alldata4).to_csv(path+'lstmvalidation1.csv', index=None)

        dataset5 = pd.read_csv(path+'lstmvalidation1.csv')
        raw = dataset5['Volume']
        raw1 = dataset5['Volume'].values[5400:]
        raw = raw.reset_index(drop=True)
        outlierdate=[]
        timelist = []
        for k in range(len(dataset5)):
            time1 = time.strptime(dataset5['date'][k], "%Y-%m-%d")
            timestamp = int(time.mktime(time1))

            # pd.DataFrame(timestamp).to_csv('timestamp.csv', mode='a', index=None, header=None)
            timelist.append(timestamp)

        csvFile = open(path+'timestamp.csv', "w+")
        try:
            writer = csv.writer(csvFile)
            writer.writerow(('timestamp',))
            for j in range(len(timelist)):
                writer.writerow((str(timelist[j] + 1),))
        finally:
            csvFile.close()
        numpy.random.seed(7)
        timestamp1 = pd.read_csv(path+"timestamp.csv")
        alldata5 = pd.concat([dataset5, timestamp1], axis=1, join='inner')
        pd.DataFrame(alldata5).to_csv(path+"lstmvalidation2.csv", index=None)
        # data1=pd.read_csv("lstmvalidation2.csv")
        # volume=data1['Volume'].values
        # volume = volume.astype('float32')
        data1 = read_csv(path+"lstmvalidation2.csv", usecols=[6], engine='python')

        volume = data1.values
        volume = data1.astype('float32')
        # transform the scale of the data


        timestamp2 = timestamp1.values

        timestamp2 = timestamp2.astype('float32')
        # scaler = MinMaxScaler(feature_range=(0, 1))
        # scaler1 = scaler.fit_transform(volume)
        #  # transform data to be supervised learning
        # print(DataFrame(scaler1))


        row = pd.concat([timestamp1, volume], axis=1, join='inner')

        pd.DataFrame(row).to_csv(path+'/nrows.csv', index=None)
        row1 = pd.read_csv(path+'/nrows.csv').values
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler1 = scaler.fit_transform(row1)
        supervised_values = series_to_supervised(scaler1)
        train, test = numpy.array(supervised_values[:3936]), numpy.array(supervised_values[5400:])
        result = []
        for i in range(repeats):
            history,result,rmse = fit_lstm(train, test, raw, scaler, batch_size, epochs, neurons, date, class1, raw1,result, outlierdate)
            pyplot.plot(history['train'], color='blue')
            pyplot.plot(history['test'], color='orange')
            # print('%d) TrainRMSE=%f, TestRMSE=%f' % (i, history['train'].iloc[-1], history['test'].iloc[-1]))
        pyplot.savefig(pathproject+'static/epochs_diagnostic.png')

        dateparse = lambda dates: pd.datetime.strptime(dates, "%Y-%m-%d %H:%M:%S").strftime("%Y-%m-%d %H:%M:%S")
        dataset1 = pd.read_csv(path+'allnove&dec1.csv', encoding='utf-8', index_col='Sdate', date_parser=dateparse)
        df = pd.DataFrame(dataset1['class'][5400:])
        count = len(df[df['class'] == 1])
        recall = len(result) / count

    return render_template('lstm.html',result=len(result), count=count, accuracy= recall,date3=str(outlierdate),rmse=rmse)

@app.route('/concat2',methods=['POST'])
def lstmconcat():
    nove2012 = pd.read_csv(path+"20121101.csv")
    nove2013 = pd.read_csv(path+"20131101.csv")
    nove2015 = pd.read_csv(path+"20151101.csv")
    nove2016 = pd.read_csv(path+"20161101.csv")
    allnove = pd.concat([nove2012, nove2013, nove2015, nove2016], axis=0, join='inner')
    pd.DataFrame(allnove).to_csv(path+'allnove.csv', index=None)

    dec2012 = pd.read_csv(path+"20121201.csv")
    dec2013 = pd.read_csv(path+"20131201.csv")
    dec2015 = pd.read_csv(path+"20151201.csv")
    dec2016 = pd.read_csv(path+"20161201.csv")
    alldec = pd.concat([dec2012, dec2013, dec2015, dec2016], axis=0, join='inner')
    pd.DataFrame(alldec).to_csv(path+'alldec.csv', index=None)

    for h in [path+'allnove.csv', path+'alldec.csv']:

        my_file = h
        if os.path.exists(my_file):
            alldata = pd.read_csv(h)
            alldata1 = alldata['Sdate'].str.split('\s', expand=True)
            pd.DataFrame(alldata1).to_csv(path+'date&hours.csv', index=None, header=['date', 'hours'])
            alldata2 = pd.read_csv(path+'date&hours.csv')
            alldata3 = pd.concat([alldata, alldata2], axis=1)
            pd.DataFrame(alldata3).to_csv(h, index=None)
        timedata = pd.read_csv(h)

        week = []
        for i in range(len(timedata)):
            week1 = datetime.strptime(str(timedata['date'][i]), "%Y-%m-%d").weekday()
            week.append(week1)
        csvFile = open(path+"week.csv", "w+")
        try:
            writer = csv.writer(csvFile)
            writer.writerow((path+'weekday',))
            for j in range(len(week)):
                writer.writerow(str(week[j] + 1))
        finally:
            csvFile.close()
        weekday = pd.read_csv(path+"week.csv")
        alldata = pd.concat([timedata, weekday], axis=1)
        pd.DataFrame(alldata).to_csv(h, index=None)

    for k in [path+'/allnove', path+'alldec']:
        lstmtrain = pd.read_csv(k + ".csv")
        df2 = pd.DataFrame(lstmtrain)
        lstmtrain2 = df2[df2['LaneDescription'] == 'NB_OS']
        df3 = pd.DataFrame(lstmtrain2)
        df5 = df3.reset_index(drop=True)

        for s in range(0, 10):
            hour = df5[df5['hours'] == '0' + str(s) + ':00:00']
            df6 = pd.DataFrame(hour)
            df7 = df6.reset_index(drop=True)
            med = df7['Volume'].median()
            q3 = np.percentile(df7['Volume'], 75)
            q1 = np.percentile(df7['Volume'], 25)
            iqr = abs(q3 - q1)
            threshold = q3 + 1.5 * iqr
            threshold1 = q1 - 1.5 * iqr

            for l in range(len(df7)):
                if df7['Volume'][l] < threshold1 or df7['Volume'][l] > threshold:
                    df7.loc[l, 'Volume'] = med
            pd.DataFrame(df7).to_csv(k + str(s) + 'hours.csv', index=None)

        for s1 in range(10, 24):
            hour = df5[df5['hours'] == str(s1) + ':00:00']
            df6 = pd.DataFrame(hour)
            df7 = df6.reset_index(drop=True)
            med = df7['Volume'].median()
            q3 = np.percentile(df7['Volume'], 75)
            q1 = np.percentile(df7['Volume'], 25)
            iqr = abs(q3 - q1)
            threshold = q3 + 1.5 * iqr
            threshold1 = q1 - 1.5 * iqr

            for l in range(len(df7)):
                if df7['Volume'][l] < threshold1 or df7['Volume'][l] > threshold:
                    df7.loc[l, 'Volume'] = med

            pd.DataFrame(df7).to_csv(k + str(s1) + 'hours.csv', index=None)

    nove0 = pd.read_csv(path+"allnove0hours.csv")
    nove1 = pd.read_csv(path+"allnove1hours.csv")
    nove2 = pd.read_csv(path+"allnove2hours.csv")
    nove3 = pd.read_csv(path+"allnove3hours.csv")
    nove4 = pd.read_csv(path+"allnove4hours.csv")
    nove5 = pd.read_csv(path+"allnove5hours.csv")
    nove6 = pd.read_csv(path+"allnove6hours.csv")
    nove7 = pd.read_csv(path+"allnove7hours.csv")
    nove8 = pd.read_csv(path+"allnove8hours.csv")
    nove9 = pd.read_csv(path+"allnove9hours.csv")
    nove10 = pd.read_csv(path+"allnove10hours.csv")
    nove11 = pd.read_csv(path+"allnove11hours.csv")
    nove12 = pd.read_csv(path+"allnove12hours.csv")
    nove13 = pd.read_csv(path+"allnove13hours.csv")
    nove14 = pd.read_csv(path+"allnove14hours.csv")
    nove15 = pd.read_csv(path+"allnove15hours.csv")
    nove16 = pd.read_csv(path+"allnove16hours.csv")
    nove17 = pd.read_csv(path+"allnove17hours.csv")
    nove18 = pd.read_csv(path+"allnove18hours.csv")
    nove19 = pd.read_csv(path+"allnove19hours.csv")
    nove20 = pd.read_csv(path+"allnove20hours.csv")
    nove21 = pd.read_csv(path+"allnove21hours.csv")
    nove22 = pd.read_csv(path+"allnove22hours.csv")
    nove23 = pd.read_csv(path+"allnove23hours.csv")

    allnove1 = pd.concat([nove0, nove1, nove2, nove3, nove4, nove5, nove6, nove7, nove8, nove9,
                          nove10, nove11, nove12, nove13, nove14, nove15, nove16, nove17, nove18, nove19,
                          nove20, nove21, nove22, nove23], axis=0, join='inner')
    nove24 = pd.read_csv(path+'20171101.csv')
    df24 = pd.DataFrame(nove24)
    nove25 = df24[df24['LaneDescription'] == 'NB_OS']
    allnove2 = pd.concat([allnove1, nove25], axis=0, join='inner')
    pd.DataFrame(allnove2).to_csv(path+'allnove1.csv', index=None)
    allnove3 = pd.read_csv(path+'allnove1.csv')  # single data
    df25 = pd.DataFrame(allnove3)
    df25['Sdate'] = pd.to_datetime(df25['Sdate'])
    df25.sort_values('Sdate', inplace=True)
    pd.DataFrame(df25).to_csv(path+'allnove2.csv', index=None)

    dec0 = pd.read_csv(path+"alldec0hours.csv")
    dec1 = pd.read_csv(path+"alldec1hours.csv")
    dec2 = pd.read_csv(path+"alldec2hours.csv")
    dec3 = pd.read_csv(path+"alldec3hours.csv")
    dec4 = pd.read_csv(path+"alldec4hours.csv")
    dec5 = pd.read_csv(path+"alldec5hours.csv")
    dec6 = pd.read_csv(path+"alldec6hours.csv")
    dec7 = pd.read_csv(path+"alldec7hours.csv")
    dec8 = pd.read_csv(path+"alldec8hours.csv")
    dec9 = pd.read_csv(path+"alldec9hours.csv")
    dec10 = pd.read_csv(path+"alldec10hours.csv")
    dec11 = pd.read_csv(path+"alldec11hours.csv")
    dec12 = pd.read_csv(path+"alldec12hours.csv")
    dec13 = pd.read_csv(path+"alldec13hours.csv")
    dec14 = pd.read_csv(path+"alldec14hours.csv")
    dec15 = pd.read_csv(path+"alldec15hours.csv")
    dec16 = pd.read_csv(path+"alldec16hours.csv")
    dec17 = pd.read_csv(path+"alldec17hours.csv")
    dec18 = pd.read_csv(path+"alldec18hours.csv")
    dec19 = pd.read_csv(path+"alldec19hours.csv")
    dec20 = pd.read_csv(path+"alldec20hours.csv")
    dec21 = pd.read_csv(path+"alldec21hours.csv")
    dec22 = pd.read_csv(path+"alldec22hours.csv")
    dec23 = pd.read_csv(path+"alldec23hours.csv")
    alldec1 = pd.concat([dec0, dec1, dec2, dec3, dec4, dec5, dec6, dec7, dec8, dec9,
                         dec10, dec11, dec12, dec13, dec14, dec15, dec16, dec17, dec18, dec19,
                         dec20, dec21, dec22, dec23], axis=0, join='inner')
    dec24 = pd.read_csv(path+'/trafficdata/20171201.csv')
    df26 = DataFrame(dec24)
    dec25 = df26[df26['LaneDescription'] == 'NB_OS']
    alldec2 = pd.concat([alldec1, dec25], axis=0, join='inner')
    pd.DataFrame(alldec2).to_csv(path+'alldec1.csv', index=None)
    alldec3 = pd.read_csv(path+'alldec1.csv')  # single data
    df26 = pd.DataFrame(alldec3)
    df26['Sdate'] = pd.to_datetime(df26['Sdate'])
    df26.sort_values('Sdate', inplace=True)
    pd.DataFrame(df26).to_csv(path+'alldec2.csv', index=None)

    # label data
    for data in (path+'allnove2', path+'alldec2'):
        dateparse = lambda dates: pd.datetime.strptime(dates, "%d/%m/%Y %H:%M:%S").strftime("%Y-%m-%d %H:%M:%S")
        dateparse1 = lambda dates: pd.datetime.strptime(dates, "%Y-%m-%d %H:%M:%S").strftime("%Y-%m-%d %H:%M:%S")

        allmatch = pd.read_csv(path+'allmatches.csv', encoding='utf-8', index_col='Sdate', date_parser=dateparse)
        allnove3 = pd.read_csv(data + '.csv', encoding='utf-8', index_col='Sdate', date_parser=dateparse1)

        label = []
        for lable1 in range(len(allnove3)):
            label.append(0)
            for lable2 in range(len(allmatch)):
                if allnove3.index[lable1] == allmatch.index[lable2]:
                    if allmatch['place'][lable2] == ' H ':
                        label[lable1] = 1
        #
        pd.DataFrame(label).to_csv(path+'labels.csv', header=['class'], index=None)
        allnove4 = pd.read_csv(data + '.csv')
        labels = pd.read_csv(path+"labels.csv")
        allnove5 = pd.concat([allnove4, labels], axis=1)
        pd.DataFrame(allnove5).to_csv(data + '.csv', index=None)
    alldata1 = pd.read_csv(path+'allnove2.csv')
    alldata2 = pd.read_csv(path+'alldec2.csv')
    alldata3 = pd.concat([alldata1, alldata2], axis=0, join='inner')

    pd.DataFrame(alldata3).to_csv(path+'allnove&dec.csv', index=None)

    data = pd.read_csv(path+'allnove&dec.csv')
    df28 = pd.DataFrame(data)
    df28['Sdate'] = pd.to_datetime(df28['Sdate'])
    df28.sort_values('Sdate', inplace=True)
    pd.DataFrame(df28).to_csv(path+'trafficdata/lstmallnove&dec2.csv', index=None)
    return render_template('lstm.html')






def Kmeans(count,outliers,q,w,points4,total):
    # data preprocessing
    dateparse = lambda dates: pd.datetime.strptime(dates, "%d/%m/%Y %H:%M:%S").strftime("%Y-%m-%d %H:%M:%S")
    dateparse1 = lambda dates: pd.datetime.strptime(dates, "%Y-%m-%d %H:%M:%S").strftime("%Y-%m-%d %H:%M:%S")

    data1 = pd.read_csv(path+'alldata2.csv', encoding='utf-8', index_col='Sdate')
    data2 = pd.read_csv(path+'allmatches.csv', encoding='utf-8', index_col='Sdate', date_parser=dateparse)
    df = pd.DataFrame(data1)
    data3 = df[df['LaneDescription'] == 'NB_NS']
    pd.DataFrame(data3).to_csv(path+'/alldaynbns.csv')
    data4 = pd.read_csv(path+'alldaynbns.csv', encoding='utf-8', index_col='Sdate', date_parser=dateparse1)

    label = []
    for i in range(len(data4)):
        label.append(0)
        for j in range(len(data2)):
            if data4.index[i] == data2.index[j]:
                if data2['place'][j] == ' H ':
                    label[i] = 1
    #
    pd.DataFrame(label).to_csv(path+'labels.csv', header=['class'], index=None)
    data44 = pd.read_csv(path+'alldaynbns.csv')
    data5 = pd.read_csv(path+"labels.csv")
    data6 = pd.concat([data44, data5], axis=1)
    pd.DataFrame(data6).to_csv(path+'kmeansdata.csv', index=None)


    plt.gcf().clear()
    plt.gcf().clear()
    data7 = pd.read_csv(path+'kmeansdata.csv')
    df1 = pd.DataFrame(data7)
    data8 = df1[df1['weekday'] == q]
    df2 = pd.DataFrame(data8)
    data9 = df2[df2['hours'] == str(w) + ':00:00']
    data99 = df2[df2['hours'] == str(w - 1) + ':00:00']

    median = data9['Volume'].median()
    data10 = data9['Volume']
    data10 = data10.reset_index(drop=True)

    # drop outliers
    q3 = np.percentile(data9['Volume'], 75)
    q1 = np.percentile(data9['Volume'], 25)
    iqr = abs(q3 - q1)
    threshold = q1 - 1.5 * iqr

    for j in range(len(data9['Volume'])):
        if data10[j] < threshold:
            data10[j] = median

    data11 = (data9['AvgSpeed']) / 100
    data11 = data11.reset_index(drop=True)
    data12 = data9['class']
    data12 = data12.reset_index(drop=True)
    data13 = data9['Sdate']
    data13 = data13.reset_index(drop=True)
    result = pd.concat([data10, data11], axis=1)
    result1 = pd.concat([data13, data10, data11, data12], axis=1)

    pd.DataFrame(result).to_csv(path+'kmeansdataset1.csv', index=None, header=0)
    pd.DataFrame(result1).to_csv(path+'kmeansdataset2.csv', index=None)
    pd.DataFrame(data99).to_csv(path+'kmeansdataset4.csv', index=None)

    # hours-1


    median1 = data99['Volume'].median()
    data100 = data99['Volume']
    data100 = data100.reset_index(drop=True)

    # drop outliers
    q33 = np.percentile(data99['Volume'], 75)
    q11 = np.percentile(data99['Volume'], 25)
    iqr1 = abs(q33 - q11)
    threshold1 = q11 - 1.5 * iqr1

    for f in range(len(data99['Volume'])):
        if data100[f] < threshold1:
            data100[f] = median1

    data111 = (data99['AvgSpeed']) / 100
    data111 = data111.reset_index(drop=True)
    result2 = pd.concat([data100, data111], axis=1)

    pd.DataFrame(result2).to_csv(path+'/kmeansdataset3.csv', index=None, header=0)

    if w == 12:
        # kmeans

        dateparse = lambda dates: pd.datetime.strptime(dates, "%d/%m/%Y %H:%M:%S").strftime("%Y-%m-%d %H:%M:%S")
        dateparse1 = lambda dates: pd.datetime.strptime(dates, "%Y-%m-%d %H:%M:%S").strftime("%Y-%m-%d %H:%M:%S")
        points = np.loadtxt(path+'kmeansdataset1.csv', delimiter=',')
        points1 = pd.read_csv(path+'kmeansdataset2.csv', encoding='utf-8', index_col='Sdate', date_parser=dateparse1)
        matches = pd.read_csv(path+'allmatches.csv', encoding='utf-8', index_col='Sdate', date_parser=dateparse)
        points2 = []
        points3 = []

        for h in points1['class']:
            points2.append(h)
            if h == 1:
                count.append(h)
        total.append(points)

        kmeans1 = KMeans(n_clusters=2, random_state=0).fit(points)

        for l in range(len(kmeans1.labels_)):
            if kmeans1.labels_[l] == np.argmax(np.bincount(kmeans1.labels_)):
                points4.append(l)

        # try:
        for k in range(len(points1)):

            for j in range(len(matches)):
                if points1.index[k] == matches.index[j]:
                    if matches['place'][j] == ' H ':
                        kmeans = KMeans(n_clusters=2, random_state=0).fit(points)
                        points3.append(kmeans.labels_)
        if len(points3):

            for d in range(len(points3[0])):
                if points3[0][d] == np.argmin(np.bincount(points3[0])):
                    if points2[d] == 1:
                        outliers.append(d)
                elif points3[0][d] == np.argmax(np.bincount(points3[0])):
                    points4.append(d)
        # else:
        #     for k in count:
        #         outliers.append(k)
        plt.clf()
        plt.gcf().clear()
        markers = ['^', 'x']
        for i in range(2):
            members = kmeans1.labels_ == i
            plt.scatter(points[members, 1], points[members, 0], s=60, marker=markers[i], c='b', alpha=0.5)
        label_added = False
        for k in outliers:
            if not label_added:
                plt.scatter(points[k][1], points[k][0], color='red', label='Outliers by footballdmatch')
                label_added = True
            else:
                plt.scatter(points[k][1], points[k][0], color='red')

        plt.legend()
        plt.title(' ')
        plt.savefig(path+str(q) + str(w) + 'hour.png')
        plt.gcf().clear()
        plt.clf()



    elif w == 19:
        # kmeans

        dateparse = lambda dates: pd.datetime.strptime(dates, "%d/%m/%Y %H:%M:%S").strftime("%Y-%m-%d %H:%M:%S")
        dateparse1 = lambda dates: pd.datetime.strptime(dates, "%Y-%m-%d %H:%M:%S").strftime("%Y-%m-%d %H:%M:%S")
        points = np.loadtxt(path+'kmeansdataset1.csv', delimiter=',')
        points1 = pd.read_csv(path+'kmeansdataset2.csv', encoding='utf-8', index_col='Sdate', date_parser=dateparse1)
        matches = pd.read_csv(path+'allmatches.csv', encoding='utf-8', index_col='Sdate', date_parser=dateparse)
        points2 = []
        points3 = []

        for h in points1['class']:
            points2.append(h)
            if h == 1:
                count.append(h)
        total.append(points)

        kmeans1 = KMeans(n_clusters=2, random_state=0).fit(points)

        for l in range(len(kmeans1.labels_)):

            if kmeans1.labels_[l] == np.argmax(np.bincount(kmeans1.labels_)):
                points4.append(l)

        # try:
        for k in range(len(points1)):

            for j in range(len(matches)):
                if points1.index[k] == matches.index[j]:
                    if matches['place'][j] == ' H ':
                        kmeans = KMeans(n_clusters=2, random_state=0).fit(points)
                        points3.append(kmeans.labels_)

        if len(points3):

            for d in range(len(points3[0])):
                if points3[0][d] == np.argmin(np.bincount(points3[0])):
                    if points2[d] == 1:
                        outliers.append(d)

        # else:
        #     for k in count:
        #         outliers.append(k)
        plt.clf()
        plt.gcf().clear()
        markers = ['^', 'x']
        for i in range(2):
            members = kmeans1.labels_ == i
            plt.scatter(points[members, 1], points[members, 0], s=60, marker=markers[i], c='b', alpha=0.5)
        label_added = False
        for k in outliers:
            if not label_added:
                plt.scatter(points[k][1], points[k][0], color='red', label='Outliers by footballdmatch')
                label_added = True
            else:
                plt.scatter(points[k][1], points[k][0], color='red')

        plt.legend()
        plt.title(' ')
        plt.savefig(path+str(q) + str(w) + 'hour.png')

        plt.gcf().clear()




    else:

        # kmeans

        dateparse = lambda dates: pd.datetime.strptime(dates, "%d/%m/%Y %H:%M:%S").strftime("%Y-%m-%d %H:%M:%S")
        dateparse1 = lambda dates: pd.datetime.strptime(dates, "%Y-%m-%d %H:%M:%S").strftime("%Y-%m-%d %H:%M:%S")
        points = np.loadtxt(path+'kmeansdataset3.csv', delimiter=',')
        points1 = pd.read_csv(path+'kmeansdataset2.csv', encoding='utf-8', index_col='Sdate', date_parser=dateparse1)
        matches = pd.read_csv(path+'allmatches.csv', encoding='utf-8', index_col='Sdate', date_parser=dateparse)
        points2 = []
        points3 = []
        total.append(points)
        for h in points1['class']:
            points2.append(h)
            if h == 1:
                count.append(h)

        kmeans1 = KMeans(n_clusters=2, random_state=0).fit(points)

        for l in range(len(kmeans1.labels_)):
            if kmeans1.labels_[l] == np.argmax(np.bincount(kmeans1.labels_)):
                points4.append(l)

        # try:
        for k in range(len(points1)):

            for j in range(len(matches)):
                if points1.index[k] == matches.index[j]:
                    if matches['place'][j] == ' H ':
                        kmeans = KMeans(n_clusters=2, random_state=0).fit(points)
                        points3.append(kmeans.labels_)
        if len(points3):

            for d in range(len(points3[0])):

                try:
                    if points3[0][d] == np.argmin(np.bincount(points3[1])):
                        if points2[d - 1] == 1 or points2[d + 1] == 1:
                            outliers.append(d)

                except:
                    pass
        # else:
        #     for k in count:
        #         outliers.append(k)
        plt.clf()
        plt.gcf().clear()
        markers = ['^', 'x']
        for i in range(2):
            members = kmeans1.labels_ == i
            plt.scatter(points[members, 1], points[members, 0], s=60, marker=markers[i], c='b', alpha=0.5)
        label_added = False
        for k in outliers:
            if not label_added:
                plt.scatter(points[k][1], points[k][0], color='red', label='Outliers by footballdmatch')
                label_added = True
            else:
                plt.scatter(points[k][1], points[k][0], color='red')

        plt.legend()
        plt.title(' ')
        plt.savefig(path+ str(q) + str(w) + 'hour.png')
        plt.gcf().clear()
        plt.clf()

@app.route('/concat',methods=['POST'])
def kmeansconcat():

    dateparse = lambda dates: pd.datetime.strptime(dates, "%d/%m/%Y %H:%M").strftime("%Y-%m-%d %H:%M")
    data1 = pd.read_csv(path+'trafficdata/20121101.csv', encoding='utf-8', index_col='Sdate',date_parser=dateparse)
    data2 = pd.read_csv(path+'trafficdata/20121201.csv', encoding='utf-8', index_col='Sdate',date_parser=dateparse)
    data3 = pd.read_csv(path+'trafficdata/20131101.csv', encoding='utf-8', index_col='Sdate',date_parser=dateparse)
    data4 = pd.read_csv(path+'trafficdata/20131201.csv', encoding='utf-8', index_col='Sdate',date_parser=dateparse)
    data5 = pd.read_csv(path+'trafficdata/20141101.csv', encoding='utf-8', index_col='Sdate',date_parser=dateparse)
    data6 = pd.read_csv(path+'trafficdata/20141201.csv', encoding='utf-8', index_col='Sdate',date_parser=dateparse)
    data2015 = pd.read_csv(path+'trafficdata/20151101.csv')
    data2016 = pd.read_csv(path+'trafficdata/20161101.csv')
    data2017= pd.read_csv(path+'trafficdata/20171101.csv')
    data201712= pd.read_csv(path+'trafficdata/20171201.csv')
    data7 = pd.concat([data1, data2, data3, data4, data5, data6], axis=0, join='inner')
    data77 = pd.concat([data2015,data2016,data2017,data201712], axis=0, join='inner')
    pd.DataFrame(data7).to_csv(path+'data.csv')
    pd.DataFrame(data77).to_csv(path+'data1.csv',index=None)
    indata1=pd.read_csv(path+"data.csv")
    indata2=pd.read_csv(path+"data1.csv")
    indata3=pd.concat([indata1,indata2], axis=0, join='inner')
    pd.DataFrame(indata3).to_csv(path+"data3.csv",index=None)
    data8=pd.read_csv(path+'data3.csv')
    data9=data8['Sdate'].str.split('\s',expand=True)
    pd.DataFrame(data9).to_csv(path+'date&hours.csv',index=None,header=['date','hours'])
    data10= pd.read_csv(path+'date&hours.csv')
    data11=pd.concat([data8,data10],axis=1)
    pd.DataFrame(data11).to_csv(path+'alldata.csv', index=None)
    timedata = pd.read_csv(path+"alldata.csv")
    week = []
    for i in range(len(timedata)):
        week1 = datetime.strptime(str(timedata['date'][i]), "%Y-%m-%d").weekday()
        week.append(week1)
    csvFile = open(path+"week.csv", "w+")
    try:
        writer = csv.writer(csvFile)
        writer.writerow(('weekday',))
        for j in range(len(week)):
            writer.writerow(str(week[j] + 1))
    finally:
        csvFile.close()
    weekday = pd.read_csv(path+"week.csv")
    alldata = pd.concat([timedata, weekday], axis=1)
    pd.DataFrame(alldata).to_csv(path+"alldata2.csv", index=None)
    return render_template('Kmeans.html')
@app.route('/concat1',methods=['POST'])
def arimaconcat():

    dateparse = lambda dates: pd.datetime.strptime(dates, "%d/%m/%Y %H:%M").strftime("%Y-%m-%d %H:%M")
    data1 = pd.read_csv(path+'trafficdata/20121101.csv', encoding='utf-8', index_col='Sdate',date_parser=dateparse)
    data2 = pd.read_csv(path+'trafficdata/20121201.csv', encoding='utf-8', index_col='Sdate',date_parser=dateparse)
    data3 = pd.read_csv(path+'trafficdata/20131101.csv', encoding='utf-8', index_col='Sdate',date_parser=dateparse)
    data4 = pd.read_csv(path+'trafficdata/20131201.csv', encoding='utf-8', index_col='Sdate',date_parser=dateparse)
    data5 = pd.read_csv(path+'trafficdata/20141101.csv', encoding='utf-8', index_col='Sdate',date_parser=dateparse)
    data6 = pd.read_csv(path+'trafficdata/20141201.csv', encoding='utf-8', index_col='Sdate',date_parser=dateparse)
    data2015 = pd.read_csv(path+'trafficdata/20151101.csv')
    data2016 = pd.read_csv(path+'trafficdata/20161101.csv')
    data2017= pd.read_csv(path+'trafficdata/20171101.csv')
    data201712= pd.read_csv(path+'trafficdata/20171201.csv')
    data7 = pd.concat([data1, data2, data3, data4, data5, data6], axis=0, join='inner')
    data77 = pd.concat([data2015,data2016,data2017,data201712], axis=0, join='inner')
    pd.DataFrame(data7).to_csv(path+'data.csv')
    pd.DataFrame(data77).to_csv(path+'data1.csv',index=None)
    indata1=pd.read_csv(path+"data.csv")
    indata2=pd.read_csv(path+"data1.csv")
    indata3=pd.concat([indata1,indata2], axis=0, join='inner')
    pd.DataFrame(indata3).to_csv(path+"data3.csv",index=None)
    data8=pd.read_csv(path+'data3.csv')
    data9=data8['Sdate'].str.split('\s',expand=True)
    pd.DataFrame(data9).to_csv(path+'date&hours.csv',index=None,header=['date','hours'])
    data10= pd.read_csv(path+'date&hours.csv')
    data11=pd.concat([data8,data10],axis=1)
    pd.DataFrame(data11).to_csv(path+'alldata.csv', index=None)
    timedata = pd.read_csv(path+"alldata.csv")
    week = []
    for i in range(len(timedata)):
        week1 = datetime.strptime(str(timedata['date'][i]), "%Y-%m-%d").weekday()
        week.append(week1)
    csvFile = open(path+"week.csv", "w+")
    try:
        writer = csv.writer(csvFile)
        writer.writerow(('weekday',))
        for j in range(len(week)):
            writer.writerow(str(week[j] + 1))
    finally:
        csvFile.close()
    weekday = pd.read_csv(path+"week.csv")
    alldata = pd.concat([timedata, weekday], axis=1)
    pd.DataFrame(alldata).to_csv(path+"trafficdata/alldata3.csv", index=None)
    return render_template('Arima.html')

def Kmeans1(q,w):
    data7 = pd.read_csv(path+'kmeansdata.csv')
    df1 = pd.DataFrame(data7)
    data8 = df1[df1['weekday'] == q]
    df2 = pd.DataFrame(data8)
    data9 = df2[df2['hours'] == str(w) + ':00:00']
    data99 = df2[df2['hours'] == str(w - 1) + ':00:00']

    median = data9['Volume'].median()
    data10 = data9['Volume']
    data10 = data10.reset_index(drop=True)

    # drop outliers
    q3 = np.percentile(data9['Volume'], 75)
    q1 = np.percentile(data9['Volume'], 25)
    iqr = abs(q3 - q1)
    threshold = q1 - 1.5 * iqr

    for j in range(len(data9['Volume'])):
        if data10[j] < threshold:
            data10[j] = median

    data11 = (data9['AvgSpeed']) / 100
    data11 = data11.reset_index(drop=True)
    data12 = data9['class']
    data12 = data12.reset_index(drop=True)
    data13 = data9['Sdate']
    data13 = data13.reset_index(drop=True)
    result = pd.concat([data10, data11], axis=1)
    result1 = pd.concat([data13, data10, data11, data12], axis=1)

    pd.DataFrame(result).to_csv(path+'kmeansdataset1.csv', index=None, header=0)
    pd.DataFrame(result1).to_csv(path+'kmeansdataset2.csv', index=None)
    pd.DataFrame(data99).to_csv(path+'kmeansdataset4.csv', index=None)

    # hours-1


    median1 = data99['Volume'].median()
    data100 = data99['Volume']
    data100 = data100.reset_index(drop=True)

    # drop outliers
    q33 = np.percentile(data99['Volume'], 75)
    q11 = np.percentile(data99['Volume'], 25)
    iqr1 = abs(q33 - q11)
    threshold1 = q11 - 1.5 * iqr1

    for f in range(len(data99['Volume'])):
        if data100[f] < threshold1:
            data100[f] = median1

    data111 = (data99['AvgSpeed']) / 100
    data111 = data111.reset_index(drop=True)
    result2 = pd.concat([data100, data111], axis=1)

    pd.DataFrame(result2).to_csv(path+'kmeansdataset3.csv', index=None, header=0)

    count = []
    outliers = []
    index = []

    dateparse = lambda dates: pd.datetime.strptime(dates, "%d/%m/%Y %H:%M:%S").strftime("%Y-%m-%d %H:%M:%S")
    dateparse1 = lambda dates: pd.datetime.strptime(dates, "%Y-%m-%d %H:%M:%S").strftime("%Y-%m-%d %H:%M:%S")
    points = np.loadtxt(path+'kmeansdataset3.csv', delimiter=',')
    points1 = pd.read_csv(path+'kmeansdataset2.csv', encoding='utf-8', index_col='Sdate', date_parser=dateparse1)
    matches = pd.read_csv(path+'allmatches.csv', encoding='utf-8', index_col='Sdate', date_parser=dateparse)
    points2 = []
    points3 = []

    for h in points1['class']:
        points2.append(h)
        if h == 1:
            count.append(h)

    kmeans1 = KMeans(n_clusters=2, random_state=0).fit(points)

    # try:
    for k in range(len(points1)):

        for j in range(len(matches)):
            if points1.index[k] == matches.index[j]:
                if matches['place'][j] == ' H ':
                    kmeans = KMeans(n_clusters=2, random_state=0).fit(points)
                    points3.append(kmeans.labels_)

    if len(points3):
        for d in range(len(points3[0])):
            if points3[0][d] == np.argmin(np.bincount(points3[0])):
                if points2[d] == 1:
                    print(1)
                    outliers.append(d)

    markers = ['^', 'x']
    for i in range(2):
        members = kmeans1.labels_ == i
        plt.scatter(points[members, 1], points[members, 0], s=60, marker=markers[i], c='b', alpha=0.5)
    label_added = False
    for k in outliers:
        if not label_added:
            plt.scatter(points[k][1], points[k][0], color='red', label='Outliers by football match')
            label_added = True
        else:
            plt.scatter(points[k][1], points[k][0], color='red')

    plt.legend()
    plt.title(' ')
    plt.savefig(path+'kmeansfig/'+str(q) + str(w) + 'hour.png')
    plt.savefig(pathproject+'static/'+str(q) + str(w) + 'hour.png')
    plt.gcf().clear()
    plt.clf()
    return

@app.route('/Kmeans',methods=['POST'])
def kmeans1():
    if request.method=='POST':

        return render_template('Kmeans.html')

@app.route('/Kmeans1',methods=['POST'])
def kmeans():
    if request.method=='POST':
        count = []
        outliers = []
        points4 = []
        total = []
        for i in [2, 3, 4, 6, 7]:
            for j in [12, 14, 15, 16, 17, 18, 19, 20]:
                print('training'+str(i)+str(j))
                Kmeans(count, outliers, i, j,points4,total)
        normal=(len(points4))
        totalnor=(len(total))
        out=(len(outliers))
        totalout=(len(count))
        reuslt = len(outliers) / len(count)
        return render_template('Kmeans.html',result= reuslt, normal= normal,totalnor=totalnor,
                               out= out, totalout= totalout)


def return_img_stream(img_local_path):
    img_stream = ''
    with open(img_local_path, 'rb') as img_f:
        img_stream = img_f.read()
        img_stream = base64.b64encode(img_stream)
    return img_stream

@app.route('/Kmeans2',methods=['POST'])
def kmeans2():
    if request.method=='POST':

        weekday = int(request.form['weekday'])
        time = int(request.form['time'])
        Kmeans1(weekday,time)
        img_path =str(weekday)+str(time)+"hour.png"
        img_stream = return_img_stream(img_path)
        w=weekday
        t=time
        return render_template('Kmeans.html')


@app.route('/')
def index():
    # show the user profile for that user

    return render_template('index.html')


@app.route('/datalake',methods=['GET', 'POST'], strict_slashes=False)
def datalake():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            print(request.url)
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], str(filename)))
    return render_template('Datalake.html')
    # if request.method=='POST1':
    #     return render_template('index.html')

@app.route('/datalake1',methods=['GET', 'POST'], strict_slashes=False)
def datalake1():
    if request.method == 'POST':
        return redirect(url_for('file://'+path))


@app.route('/datascrap1',methods=['POST'])
def datascrap1():
    return render_template('Datascrap.html')


date1=[]

@app.route('/datascrap',methods=['POST'])
def datascrap():

    if request.method=='POST':
        date = request.form['date']
        date1.append(date)

        url=matches(date)
        return render_template('Datascrap.html',url=url)

    if request.method=='POST1':
        return render_template('index.html')
    return date1


@app.route('/footballmatch',methods=['POST','GET'])
def footballmatch():
    # if request.method=='POST':
    if request.method=='GET':

        date = datascrap()[-1]

        return render_template(str(int(date))+'footballmatch.html')

@app.route('/Arima1',methods=['POST'])
def Arimamatches1():
    return render_template('Arima.html')


@app.route('/Arima',methods=['POST'])
def Arimamatches():
    if request.method=='POST':
        result = []
        mse=[]
        list1 = []
        total=[]

        # app.run(host="0.0.0.0", port=8386, debug=True)
        # for time in ['2015-12-05','2015-12-19','2016-11-19','2016-11-27','2016-12-11','2016-12-26',
        #             '2016-12-31','2017-11-18','2017-11-25','2017-12-10','2017-12-13','2017-12-26','2017-12-30']:
        times1 = pd.read_csv(path+'alldata3.csv')
        times2 = times1[3867:]
        times3 = times2['date']
        times4 = {}.fromkeys(times3).keys()  # del redundancy

        for times in ['2015-12-05', '2015-12-19', '2016-11-19', '2016-11-27', '2016-12-11', '2016-12-26',
                      '2016-12-31', '2017-11-18', '2017-11-25', '2017-12-10', '2017-12-13', '2017-12-26', '2017-12-30']:

            try:
                for i in [12, 14, 15, 16, 17, 19, 20]:
                    r,m = main(i, times, list1,total)
                    result.append(r)
                    mse.append(m)

            except:
                pass
        count = []
        points1 = pd.read_csv(path+'Arimavalidation2.csv')

        result1 = {}.fromkeys(result[len(result) - 1]).keys()  # del redundancy
        mse1 = mse[len(mse) - 1]
        mse3 = numpy.array(mse1)
        avgmse = np.mean(mse3)
        for h in points1['class']:
            if h == 1:
                count.append(h)
        result = len(result1) / len(count)
        r1 = (len(result1))
        c1 = (len(count))
        print(len(times1[3867:]))
        return render_template('Arima.html', result= result,avgmse=avgmse,r1=r1,c1=c1)

@app.route('/Arima2', methods=['POST'])
def Arimamatches2():

    if request.method == 'POST':
        result = []
        mse = []
        list1 = []
        times1 = pd.read_csv(path+'alldata3.csv')
        times2 = times1[3867:]
        times3 = times2['date']
        times4 = {}.fromkeys(times3).keys()  # del redundancy

        date = request.form['date']
        hour = request.form['hour']

        r, m = main(int(hour), date, list1)
        result.append(r)
        mse.append(m)

        count = [1]
        points1 = pd.read_csv(path+'Arimavalidation2.csv')
        if len(result):
            result1 = {}.fromkeys(result[len(result) - 1]).keys()  # del redundancy
            mse1=mse[len(mse) - 1]
            mse3=numpy.array(mse1)
            avgmse=np.mean(mse3)
            result = len(result1) / len(count)
            r1=(len(result1))
            c1=(len(count))
            print(len(times1[3867:]))
            return render_template('Arima.html',result=result, avgmse=avgmse, r1=r1, c1=c1)
        else:
            result1='no outliers in this period'

            return render_template('Arima.html', result1= result1)




def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS



if __name__ == '__main__':
    app.secret_key = 'super secret key'
    app.config['SESSION_TYPE'] = 'filesystem'
    app.run(host="0.0.0.0", port=8382, debug=True)
