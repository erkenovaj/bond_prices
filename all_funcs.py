import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels
import math
import scipy.stats
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy.stats import t
from scipy.stats import f
from scipy.stats import kurtosis, skew
from scipy.stats import mannwhitneyu
import statsmodels.api as sm
from scipy.stats import shapiro
from scipy.stats import norm
from statsmodels.iolib.table import SimpleTable
import statistics
from numpy import asarray
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import chi2
from sklearn.metrics import mean_squared_error
import datetime

def mod_df(name):
  df = pd.read_excel(name, skiprows=1)
  df = df[['Дата', 'Indicative', 'YTM Indicative', 'G-spread', 'ISIN']]
  isin = df['ISIN'][1]
  df['Дата'] = pd.to_datetime(df['Дата'])
  df = df.resample('D', on='Дата').median().ffill()
  df['t'] = [i+1 for i in range(len(df))]
  df['ISIN'] = isin
  df['name'] = name
  return df

def merge_dfs(dfs):
  df = pd.concat(dfs)
  df = pd.get_dummies(df)
  df = df.sort_values(by = 'Дата')
  return df

def add_factor(df, name):
  new_df = pd.read_excel(name)
  new_df['Дата'] = pd.to_datetime(new_df['Дата'])
  new_df = new_df.resample('D', on='Дата').median().bfill()
  return df.merge(new_df, left_on = df.index, right_on = new_df.index)

def graphs(data):
  data.plot()
  plt.show()
  sns.displot(data, kind = 'kde')
  plt.show()

def student_test (y1, y2, n1, n2, var1, var2, a):
  if (var1 != var2):
    St = ((y1-y2)/math.sqrt(var1/(n1-1)+var2/(n2-1)))
  else:
    St = ((y1-y2)/var1 * math.sqrt(((n1)*(n2))/(n1+n2)))
  return St

def _get_updown_sub_serias(series):
    subseries = [None, ]
    for i in range(len(series)):
        if i == 1:
            pass
        else:
            if (series[i] - series[i-1]) > 0:
                subseries.append('+')
            else:
                subseries.append('-')
    return subseries

def _get_numeric_updown_criteria(subseries):
    number_subseries = 0
    numeric_subseries = [None, ]
    for i in range(len(subseries)):
        if i == 1:
            pass
        else:
            if subseries[i] != subseries[i-1]:
                number_subseries += 1
                numeric_subseries.append(number_subseries)
            else:
                numeric_subseries.append(number_subseries)
                
    return numeric_subseries
    

def updown_criteria(series):
    subseries = _get_updown_sub_serias(series)
    numeric_subseries = _get_numeric_updown_criteria(subseries)
    sub_df = pd.DataFrame({'series': series, 'subseries': subseries, 'numeric_subs': numeric_subseries})
    val_count_sub_df = sub_df['numeric_subs'].value_counts()
    total_subdf = pd.merge(sub_df, val_count_sub_df, how='left', left_on='numeric_subs', right_index=True)
    
    v_n = total_subdf['numeric_subs_x'].max()
    t_n = total_subdf['numeric_subs_y'].max()
    
    u_t = 1.96
    
    n = total_subdf.shape[0]-1
    
    criteria_x = 1/3 * (2*n-1) - u_t* np.sqrt((16*n-29)/90)
    
    if n <= 26:
        criteria_y = 5
    elif n>26 and n <= 153:
        criteria_y = 6
    elif n>153 and n <= 1170:
        criteria_y = 7
    else:
        criteria_y = 8    
    
    print(f'Число серий v_n = {v_n} больше критического значения равного {criteria_x}: {v_n > criteria_x}')
    print(f'Длина максимальной серии t_n = {t_n} меньше критического значения равного {criteria_y}: {t_n < criteria_y}')
    print('При нарушении хотя бы одного из условий ряд нестационарный')


def test_mannwhitneyu(series):
    stat, p = mannwhitneyu(series[:int(len(series)/2)], series[int(len(series)/2):])
    print('stat={0:.3g}, p={0:.3g}'.format(stat, p))
    if p > 0.05:
        print('Probably the same distribution')
    else:
        print('Probably different distributions')

def _get_ranked(series):
    series_max_index = len(series)
    first_sublist = list(range(2,series_max_index,4))
    first_sublist = sorted(first_sublist + list(range(min(first_sublist)+1, max(first_sublist)+2, 4)))
    second_sublist = [col for col in list(range(1,series_max_index+1)) if col not in first_sublist]
    first_sublist.reverse()
    return second_sublist + first_sublist  

def siegel_tukey(series):
    first_subsample, second_subsample = series[:int(len(series)/2)], series[int(len(series)/2):]
    first_subsample, second_subsample = pd.DataFrame({'values': first_subsample}), pd.DataFrame({'values': second_subsample})
    first_subsample['class'] = 'first'
    second_subsample['class'] = 'second'
    final_df = pd.concat([first_subsample,second_subsample]).sort_values(by='values')
    
    final_df['ranked'] = _get_ranked(final_df['values'])
    _ = pd.pivot_table(final_df, index='class', aggfunc=['sum'])['sum']['ranked']
    mean_w = _['first']-len(first_subsample)*(len(first_subsample)+len(second_subsample)+1)/2 
    disp_w = len(first_subsample)*len(second_subsample)*(len(first_subsample)+len(second_subsample)+1)/12
    if disp_w != 0:
        z_value = mean_w/np.sqrt(disp_w)
    else:
        print(f'disp_w = 0')
        
    if z_value < 0:
        z_value = (mean_w + 0.5) / np.sqrt(disp_w)
    else:
        z_value = (mean_w - 0.5) / np.sqrt(disp_w)
    print(f'z value: {z_value:.3f}')
    if z_value < 1.97 and z_value > -1.97:
        print('Probably the same distribution')
    else:
        print('Probably different distributions')


def test_normal_dist(series):
    print('================')
    print('Тест Jarque Bera')
    row =  [u'JB', u'p-value', u'skew', u'kurtosis']
    jb_test = sm.stats.stattools.jarque_bera(series)
    a = np.vstack([jb_test])
    itog = SimpleTable(a, row)
    print(itog)
    alpha = 0.05
    # if itog['p-value'] > alpha:
    #     print('Принять гипотезу о нормальности')
    # else:
    #     print('Отклонить гипотезу о нормальности')       
    print('')
    print('================')
    print('Тест Шапиро-Уилк')
    stat, p = shapiro(series) # тест Шапиро-Уилк 
    print('Statistics=%.3f, p-value=%.3f' % (stat, p))
    alpha = 0.05
    if p > alpha:
        print('Принять гипотезу о нормальности')
    else:
        print('Отклонить гипотезу о нормальности')        
            
def get_distribution(series):
    print(test_normal_dist(series))
    
    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(4, 1)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[2, 0])
    ax3 = fig.add_subplot(gs[3, 0])
    sns.kdeplot(norm(loc=series.mean(), scale=np.std(series)).rvs(size=700), ax=ax0, color='#b3e8c9', fill=True, alpha=1)
    sns.kdeplot(ax=ax0, x=series)     
    sns.histplot(ax=ax1, x=series, fill=False)     
    sns.boxplot(series, ax=ax2)
    sm.qqplot(series, ax=ax3, line="s")

def kurt_skew(data):
  print( 'excess kurtosis of normal distribution (should be 0): {}'.format( kurtosis(data) ))
  print( 'skewness of normal distribution (should be 0): {}'.format( skew(data) ))

def cap_outliers(data, i, j): 
  data = data.sort_values()
  if i!=0:
    data.iloc[0:i] = data.iloc[i]
  if j!=0:
    data.iloc[(len(data)-j):len(data)] = data.iloc[len(data) - j]
  data = data.sort_index()
  return(data)

def t_f_criterion(data):
  X = data.values
  split = round(len(X) / 2)
  X1, X2 = X[0:split], X[split:]
  mean1, mean2 = X1.mean(), X2.mean()
  var1, var2 = X1.var(), X2.var()
  print('Тестирование средних. Статистика Стьюдента: {}. Критическое значение: {}'.format(((mean1-mean2)/math.sqrt(var1+var2)), t.ppf(0.95, len(X1)-1)))
  print('Тестирование дисперсий. Статистика Фишера: {}. Критическое значение: {}'.format(var2/var1, f.ppf(0.95, len(X1)-1, len(X2)-1)))


def median_criterion (data):
  result = []
  for i in range(len(data)):
    if data[i] < statistics.median(data):
      result.append('-')
    elif data[i] == statistics.median(data): result.append(' ')
    else: result.append('+')
  vt = 1
  tmax = 1
  t = 1
  for i in range(1, len(result)):
    if result[i] == result[i-1]: 
      vt = vt
      t += 1
      if t > tmax: tmax = t
    else:  
      vt +=1
      t = 0
  print('Число серий = {} больше рассчитанного {}: {}'.format(vt, (0.5*(len(data) + 2) - 1.96*math.sqrt(len(data)-1)), vt > (0.5*(len(data) + 2) - 1.96*math.sqrt(len(data)-1))))
  print('Максимальная длина серии = {} меньше рассчитанного {}: {}'.format(tmax, (1.43 * math.log(len(data + 1))), tmax < (1.43 * math.log(len(data + 1)))))
  print('При нарушении хоть одного условия, ряд не стационарный')
  print(statistics.median(data))
  print(result)

def make_autocorr_plot(series, lags=24):
    background_color = '#f6f5f5'
    fig = plt.figure(figsize=(10, 8), facecolor=background_color)
    gs = fig.add_gridspec(2, 1)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[1, 0])

    # ax0.set_facecolor(background_color)
    # ax1.set_facecolor(background_color)
    
    
    # background_color = '#f6f5f5'
    # ax0.grid(which='major', axis='x', zorder=0, color='#EEEEEE', linewidth=0.4)
    # ax0.grid(which='major', axis='y', zorder=0, color='#EEEEEE', linewidth=0.4)
    # ax0.axes.xaxis.set_visible(False)
    # ax1.grid(which='major', axis='x', zorder=0, color='#EEEEEE', linewidth=0.4)
    # ax1.grid(which='major', axis='y', zorder=0, color='#EEEEEE', linewidth=0.4)
    
    plot_acf(series, lags=lags, zero=False, ax=ax0)
    plot_pacf(series, lags=lags, zero=False, ax=ax1)

numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

def get_corr(data):
  sns.heatmap(data.select_dtypes(include=numerics).corr(), vmin = -1, annot = True, vmax = 1)
  plt.show()

def boxplots(df):
  for column in df.select_dtypes(include=numerics):
        plt.figure(figsize=(10,1))
        sns.boxplot(data=df, x=column)

def find_outliers_IQR(df):
   q1=df.quantile(0.25)
   q3=df.quantile(0.75)
   IQR=q3-q1
   outliers = df[((df<(q1-1.5*IQR)) | (df>(q3+1.5*IQR)))]
   return outliers

def ult_analysis(data):
  print(data.info())
  print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ - 1')
  print(data.isna().sum())
  print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ - 2')
  print(data.duplicated().sum())
  print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ - 3')
  get_corr(data)
  plt.show()
  print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ - 4')
  print(find_outliers_IQR(data['Indicative']))
  print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ - 5')
  boxplots(data)
  plt.show()
  print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ - 6')
  graphs(data[['Indicative']])
  plt.show()
  print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ - 7')
  test_normal_dist(data['Indicative'])
  print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ - 8')
  get_distribution(data['Indicative'])
  plt.show()
  print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ - 9')
  make_autocorr_plot(data['Indicative'])
  plt.show()
  print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ - 10')
  t_f_criterion(data['Indicative'])
  print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ - 11')
  kurt_skew(data['Indicative'])

def get_corr(data):
  sns.heatmap(data.select_dtypes(include=numerics).corr(), cmap="YlGnBu", vmin = -1, annot = True, vmax = 1)
  plt.show()

def three_graphs_one_plot (dfs):
  fig, ax = plt.subplots(figsize=(12, 6), layout='constrained')
  for df in dfs:
   ax.plot(df['Indicative'], label=df['name'][0])  
  plt.ylabel("Цена,  RUB")
  plt.xlabel("Дата")
  ax.legend()

def fac_plots(df):
  fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(14, 7))
  ll = df.columns[:6]
  for i in range(len(ll)):
    if i > 2: 
      j = 1 
    else: j = 0
    ax[j, i%3].plot(df[ll[i]], 'k-', label=ll[i])
    ax[j, i%3].legend()
    plt.ylabel("Значение показателя,  RUB")
    plt.xlabel("Дата")
  # for name in ['YTM Indicative', 'G-spread', 'RUB Yield Curve 10Y', 'RUCBCP3A3YNS', 'USD/RUB (FX)', 'Ставка RUONIA']:
  #   dfs[0][name].plot()
  #   plt.show()
  fig.tight_layout()

def cap_outliers_by_list(data): 
  outliers = find_outliers_IQR(data)
  if len(outliers) > 0:
    if outliers[0] > data.median():
      sort_df = data.sort_values(ascending = False)
    else: 
      sort_df = data.sort_values(ascending = True)
    sort_df.iloc[0:len(outliers)] = sort_df.iloc[len(outliers)+1]
    data = sort_df.sort_index()
  return(data)

def calculateMahalanobis(y=None, data=None, cov=None):
    y_mu = y - np.mean(data)
    if not cov:
        cov = np.cov(data.values.T)
    inv_covmat = np.linalg.inv(cov)
    left = np.dot(y_mu, inv_covmat)
    mahal = np.dot(left, y_mu.T)
    return mahal.diagonal()

# fit an random forest model and make a one step prediction
def random_forest_forecast(train, testX):
 # transform list into array
 train = asarray(train)
 # split into input and output columns
 trainX, trainy = train[:, :-1], train[:, -1]
 # fit model
 model = RandomForestRegressor(n_estimators=1000)
 model.fit(trainX, trainy)
 # make a one-step prediction
 yhat = model.predict([testX])
 return yhat[0]

def train_test_split(X, y, n_test):
 return X[:-n_test], X[-n_test:], y[:-n_test], y[-n_test:]

def plot_pred(predictions, y_test):
  fig, ax = plt.subplots(figsize=(10, 5), layout='constrained')
  plt.plot(pd.date_range(start='2022-12-01',end='2023-03-17'), predictions, 'k-', label = 'Предсказание прироста цен')
  plt.plot(pd.date_range(start='2022-12-01',end='2023-03-17'), y_test, 'k--', label = 'Реальный прирост цен')
  plt.ylabel("Прирост цены,  RUB")
  plt.xlabel("Дата")
  plt.legend()
  plt.show()

def pred_val(y_test, prediction):
  print('MSE = %.4g' % mean_squared_error(y_test, prediction))
  plot_pred(prediction, y_test)     

# convert time series into supervised learning problem
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
 n_vars = 1 if type(data) is list else data.shape[1]
 df = pd.DataFrame(data)
 cols, names = list(), list()
 # input sequence (t-n, ... t-1)
 for i in range(n_in, 0, -1):
  cols.append(df.shift(i))
 names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
 # forecast sequence (t, t+1, ... t+n)
 for i in range(0, n_out):
  cols.append(df.shift(-i))
 if i == 0:
  names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
 else:
  names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
 # put it all together
 agg = pd.concat(cols, axis=1)
 agg.columns = names
 # drop rows with NaN values
 if dropnan:
  agg.dropna(inplace=True)
 return agg

def persistence(last_ob, n_seq):
 return [last_ob for i in range(n_seq)]

# evaluate the persistence model
def make_forecasts(train, test, n_lag, n_seq):
	forecasts = list()
	for i in range(len(test)):
		X, y = test[i, 0:n_lag], test[i, n_lag:]
		# make forecast
		forecast = persistence(X[-1], n_seq)
		# store the forecast
		forecasts.append(forecast)
	return forecasts

# evaluate the RMSE for each forecast time step
def evaluate_forecasts(test, forecasts, n_lag, n_seq):
	for i in range(n_seq):
		actual = test[:,(n_lag+i)]
		predicted = [forecast[i] for forecast in forecasts]
		rmse = np.sqrt(mean_squared_error(actual, predicted))
		print('t+%d RMSE: %f' % ((i+1), rmse))
                
# plot the forecasts in the context of the original dataset
def plot_forecasts(series, forecasts, n_test):
	# plot the entire dataset in blue
	plt.plot(series.values)
	# plot the forecasts in red
	for i in range(len(forecasts)):
		off_s = len(series) - n_test + i
		off_e = off_s + len(forecasts[i])
		xaxis = [x for x in range(off_s, off_e)]
		plt.plot(xaxis, forecasts[i], color='red')
	# show the plot
	plt.show()
        
def split_into_sequences(data, seq_len):
    n_seq = len(data) - seq_len + 1
    return np.array([data[i:(i+seq_len)] for i in range(n_seq)])

def get_train_test_sets(data, seq_len, train_frac):
    sequences = split_into_sequences(data, seq_len)
    n_train = int(sequences.shape[0] * train_frac)
    X_train = sequences[:n_train, :-1, :]
    y_train = sequences[:n_train, -1, :]
    X_test = sequences[n_train:, :-1, :]
    y_test = sequences[n_train:, -1, :]
    return X_train, y_train, X_test, y_test