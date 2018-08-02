
# coding: utf-8

# In[26]:


##Import libraries
import math                           ##Math functions
import numpy as np                    ##Number functions
import pandas as pd                   ##Handling data sets
import datetime                       ##Used to formate date and time
import stockstats

global stock_df

#15 Indicators in this file more to come


# In[27]:


Indicatornames = ['RSI(data, value=14)','WILLIAMR(data, value=14)','CCI(data, value=14)','BB(data)','ADX(data)',
                  'MACD(data)','PCT_Change(data)',"Rate_of_Change(data,column='close',shift=1)",'RSV(data,value)',
                  "TRIX(data,column='close',value=12)",'DMA(data)',"MVAR(data,column='close',value=12)",
                  "MSTD(data,column='close',value=12)","SMA(data,column='close',value=14)",
                  "EMA(data,column='close',value=14)"]

funnames = ['ReduceDataFrame(data,timeFrame,Days2Consider=30,date_index=False)',
            'AddAttributesLabels(data,Candles2Consider=30,tradeLenght=5,date_index=False)',
           'Test_prediction(df_test, prediction)']


# In[28]:


def IndicatorNames():
    for i in range (len(Indicatornames)):
        print(Indicatornames[i])
def FunctionNames():
    for i in range (len(funnames)):
        print(funnames[i])


# In[29]:


def split(x,y,train_split=0.8):
    if len(x) != len(y):
        print('ERROR: the size of x lenght is not equal to y lenght: x:('+len(x)+') and y:('+len(y)+')')
    lenght=len(x)
    num_train = int(train_split*lenght)
    num_test = lenght - num_train
    
    x_train = x[0:num_train]
    y_train = y[0:num_train]
    x_test = x[num_train:]
    y_test = y[num_train:]
    
    return x_train, y_train, x_test, y_test


# In[30]:


def RSI(data, value=14):
    df = data.copy()
    df['RSI '+str(value)] = stock_df['rsi_'+str(value)][value+1:]
    return df


# In[31]:


def WILLIAMR(data, value=14):
    df = data.copy()
    df['WILLR '+str(value)] = stock_df['wr_'+str(value)][value+1:]
    return df


# In[32]:


def CCI(data, value=14):
    df = data.copy()
    df['CCI '+str(value)] = stock_df['cci_'+str(value)][value+1:]
    return df


# In[33]:


def BB(data):
    df = data.copy()
    df['BB Lb'] = stock_df['boll_lb'][21:]
    df['BB'] = stock_df['boll'][21:]
    df['BB Ub'] = stock_df['boll_ub'][21:]
    return df


# In[34]:


def ADX(data):
    df = data.copy()
    df['+DI'] = stock_df['pdi'][17:]
    df['-DI'] = stock_df['mdi'] [17:]
    df['DX'] = stock_df['dx'] [17:]
    df['ADX'] = stock_df['adx']  [17:]
    df['ADXr'] = stock_df['adxr'] [17:]   
    return df


# In[35]:


def MACD(data):
    df = data.copy()
    df['MACD Fast'] = stock_df['macd'][26:]
    df['MACD Slow'] = stock_df['macds'] [26:]
    df['MACD'] = stock_df['macdh'] [26:]
    return df


# In[36]:


def PCT_Change(data):
    df = data.copy()
    df['PCT Change'] = stock_df['change'][5:]
    return df


# In[37]:


def Rate_of_Change(data,column='close',shift=1):
    df = data.copy()
    df['Rate Of Change '+str(column)] = stock_df[str(column)+'_'+str(shift)+'_r'][shift+2:]
    return df


# In[38]:


def RSV(data,value): #Raw Stochastic Value
    df = data.copy()
    df['RSV '+str(value)] = stock_df['rsv_'+str(value)][value+1:]
    return df


# In[39]:


def TRIX(data,column='close',value=12): #triple exponentially smoothed moving average
    df = data.copy()
    df['TRIX '+str(column)+' '+str(value)] = stock_df[str(column)+'_'+str(value)+'_trix'][value+1:]
    return df


# In[40]:


def DMA(data): #diffrence of moving average
    df = data.copy()
    df['DMA'] = stock_df['dma'][50+1:]
    return df


# In[41]:


def MVAR(data,column='close',value=12): #moving variance
    df = data.copy()
    df['MVAR '+str(column)+' '+str(value)] = stock_df[str(column)+'_'+str(value)+'_mvar'][value+1:]
    return df


# In[42]:


def MSTD(data,column='close',value=12): #moving standard deviation
    df = data.copy()
    df['MSTD '+str(column)+' '+str(value)] = stock_df[str(column)+'_'+str(value)+'_mstd'][value+1:]
    return df


# In[43]:


def SMA(data,column='close',value=14): #simple moving average
    df = data.copy()
    df['SMA '+str(column)+' '+str(value)] = stock_df[str(column)+'_'+str(value)+'_sma'][value+1:]
    return df


# In[44]:


def EMA(data,column='close',value=14): #simple moving average
    df = data.copy()
    df['EMA '+str(column)+' '+str(value)] = stock_df[str(column)+'_'+str(value)+'_ema'][value+1:]
    return df


# In[45]:


###                           ###
###   Indicators Complete     ###
###                           ###


# In[46]:


##Get Data from csv files compressed with gzip format
def getData(ForexPair,year,timeFrame, date_index=False): #TF is either 1H or 1M
    data = pd.read_csv('Compressed_Forex_'+timeFrame+'_(gzip)/'+str(year)+'/Compressed_'+ForexPair+'_'+str(year)+'_'+timeFrame+'_.csv', compression='gzip')
    
    ##Rename columns
    if date_index == True:
        data['Date and Time'] = pd.to_datetime(data['Gmt time'],format='%d.%m.%Y %H:%M:%S.000')
        data.set_index('Date and Time', inplace=True)
    else:
        pass
    data = data.loc[:,['Open','High', 'Low', 'Close','Volume']]
    global stock_df
    stock_df = stockstats.StockDataFrame.retype(data)
    return data


# In[47]:


##This allows you to choose how many days to consider in new data frame
def ReduceDataFrame(data,timeFrame,Days2Consider=30,date_index=False): ##Original DF, TimeFrame of data, How many days to consider   
    if timeFrame == '1M':                             
        Minutes2Consider = (Days2Consider*24*60)      ##How many minutes in x amoint of days
    elif timeFrame == '1H':
        Minutes2Consider = (Days2Consider*24)         ##How many hours in x amoint of days
    else:
        pass
    ##Resize the data to get most recent values of data for x amount of days
    if Days2Consider=='all':
        df=data.copy()
        pass
    else:
        data = data.tail(round(Minutes2Consider))
        df=data.copy()
    ##Re-index the data frame
    if date_index==True:
        pass
    else:
        df = df.reset_index()
        del df['index']
    return df


# In[48]:


##This allows you to choose how many days to consider in new data frame
def SliceData(data,timeFrame,begining,ending,date_index=False): ##Original DF, TimeFrame of data, How many days to consider   
    ##slice the data to get most recent values of data for x amount of days
    data = data.tail(round(Minutes2Consider))
    df=data.copy()
    
    
    ##Re-index the data frame
    if date_index==True:
        pass
    else:
        df = df.reset_index()
        del df['index']
    return df


# In[49]:


##This is the for the NN will be able to understand each row will contain many colums 
##colums will give values of stock and indicators from that instance to x amount time before that instance.
## so that each row is an independent input for the Neaural Network
def AddAttributesLabels(data,Candles2Consider=30,tradeLenght=5,date_index=False): 
    whatToPredict = 'close'     ##this will be what i will use as the forcast value aka the Label
    df=data.copy()              ##Make a copy not as to change original
    head=list(df)
    count=len(head)
    for i in range (Candles2Consider):   ##How many candle stick back for the NN to consider
        for j in range(count):           ##Add new colums with each feature but with past market values
            df[head[j]+'_'+str(i+1)] = df[head[j]].shift(i+1)
        
        
    lenghtTradePrediction = tradeLenght             ##lenght of trade
    
    #Create a label that is 1 when up and 0 when down
    #df['1(up) or 0(down)'] = np.where(df[whatToPredict].shift(-lenghtTradePrediction) > df[whatToPredict], 1, 0) 
    
    ##Create a label that tells us what the output lenghtPerUnitTime Later
    df['Label'] = df[whatToPredict].shift(-lenghtTradePrediction)
    
    
    ##Get rid of rows that have empyt cells 
    df.dropna(inplace = True)
    
    ##Re-index the data frame
    if date_index==True:
        pass
    else:
        df = df.reset_index()
        del df['index']
    
    return df


# In[52]:


def AddLabels(data,whatToPredict = ['close'], tradeLenght=5, date_index=False):
    df=data.copy()              ##Make a copy not as to change original
    
    ##Create a label that tells us what the output lenghtPerUnitTime Later
    for i in range (len(whatToPredict)):    
        df['Label_'+whatToPredict[i]] = df[whatToPredict[i]].shift(-tradeLenght)
    
    
    ##Get rid of rows that have empyt cells 
    df.dropna(inplace = True)
    
    ##Re-index the data frame
    if date_index==True:
        pass
    else:
        df = df.reset_index()
        del df['index']
    
    return df


# In[51]:


def Test_prediction(x_test,y_test, prediction,close_position):
    good=[]
    bad=[]
    if len(prediction)==len(y_test):
        for i in range (len(prediction)):
            if y_test[i] > x_test[i,close_position]: #Up Trade
                if prediction[i] > x_test[i,close_position]:
                    good.append(0)
                else:
                    bad.append(0)
            elif y_test[i] < x_test[i,close_position]: #Down Trade
                if prediction[i] < x_test[i,close_position]:
                    good.append(0)
                else:
                    bad.append(0)
            else:
                bad.append(0)
                    
        ITM=len(good)
        OTM=len(bad)
        Trade_total=ITM+OTM
        win_PCT=round((ITM/Trade_total)*100)
        #print (str(ITM) + ' Trades ITM')
        #print (str(OTM) + ' Trades OTM')
        #print (str(win_PCT) + '% Accurate' )
        
        return win_PCT
    
    elif len(prediction)>len(y_test):
        print('Error Number of predicitions is greater than Answers')
    elif len(prediction)<len(y_test):
        print('Error Number of predicitions is less than Answers')

