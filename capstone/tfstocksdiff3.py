import time
import math
import os
from datetime import date
from datetime import timedelta
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import pandas_datareader as pdr
from pandas_datareader import data, wb
from six.moves import Pickle as pickle
from yahoo_finance import Share

# Choose amount of historical data to use NHistData
NHistData = 30
TrainDataSetSize = 3000

# Load the Dow 30 stocks from Yahoo into a Pandas datasheet

dow30 = ['AXP', 'AAPL', 'BA', 'CAT', 'CSCO', 'CVX', 'DD', 'XOM',
         'GE', 'GS', 'HD', 'IBM', 'INTC', 'JNJ', 'KO', 'JPM',
         'MCD', 'MMM', 'MRK', 'MSFT', 'NKE', 'PFE', 'PG',
         'TRV', 'UNH', 'UTX', 'VZ', 'V', 'WMT', 'DIS']

num_stocks = len(dow30)

trainData = None
loadNew = False

# If stocks.pickle exists then this contains saved stock data, so use this,
# else use the Pandas DataReader to get the stock data and then pickle it.
stock_filename = 'stocks.pickle'
if os.path.exists(stock_filename):
    try:
        with open(stock_filename, 'rb') as f:
            save = pickle.load(f)
            trainDataPrices = save['trainDataPrices']
            trainDataVolume = save['trainDataVolume']
            del save  # hint to help gc free up memory 
    except Exception as e:
      print('Unable to process data from', stock_filename, ':', e)
      raise            
    print('%s already present - Skipping requesting/pickling.' % stock_filename)
else:

    # Get the historical data. Make the date range quite a bit bigger than
    # TrainDataSetSize since there are no quotes for weekends and holidays. This
    # ensures we have enough data.
    
    f = pdr.data.DataReader(dow30, 'yahoo', date.today()-timedelta(days=TrainDataSetSize*2+5), date.today())
    cleanDataPrices = f.ix['Adj Close']
    trainDataPrices = pd.DataFrame(cleanDataPrices)
    trainDataPrices.fillna(method='backfill', inplace=True)
    cleanDataVolume = f.ix['Volume']
    trainDataVolume = pd.DataFrame(cleanDataVolume)
    trainDataVolume.fillna(method='backfill', inplace=True)
    loadNew = True
    print('Pickling %s.' % stock_filename)
    try:
        with open(stock_filename, 'wb') as f:
          save = {
              'trainDataPrices': trainDataPrices,
              'trainDataVolume': trainDataVolume,      
              }            
          pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print('Unable to save data to', stock_filename, ':', e)

# Normalize the data by dividing each price by the first price for a stock.
# This way all the prices start together at 1.
# Remember the normalizing factors so we can go back to real stock prices
# for our final predictions.
factorsPrices = np.ndarray(shape=( num_stocks ), dtype=np.float32)
factorsVolume = np.ndarray(shape=( num_stocks ), dtype=np.float32)
i = 0

for symbol in dow30:
    factorsPrices[i] = trainDataPrices[symbol][0]
    factorsVolume[i] = trainDataVolume[symbol][0]
    trainDataPrices[symbol] = trainDataPrices[symbol]/trainDataPrices[symbol][0]
    trainDataVolume[symbol] = trainDataVolume[symbol]/trainDataVolume[symbol][0]
    i = i + 1    

# Configure how much of the data to use for training, testing and validation.

usableData = len(trainDataPrices.index) - NHistData + 1
numTrainData =  int(0.6 * usableData)
numValidData =  int(0.2 * usableData)
numTestData = usableData - numTrainData - numValidData - 1
# numTrainData = usableData - 1
# numValidData = 0
# numTestData = 0

train_dataset = np.ndarray(shape=(numTrainData - 1, num_stocks * NHistData * 2), dtype=np.float32)
train_labels = np.ndarray(shape=(numTrainData - 1, num_stocks), dtype=np.float32)
valid_dataset = np.ndarray(shape=(max(0, numValidData - 1), num_stocks * NHistData * 2), dtype=np.float32)
valid_labels = np.ndarray(shape=(max(0, numValidData - 1), num_stocks), dtype=np.float32)
test_dataset = np.ndarray(shape=(max(0, numTestData - 1), num_stocks * NHistData * 2), dtype=np.float32)
test_labels = np.ndarray(shape=(max(0, numTestData - 1), num_stocks), dtype=np.float32)
final_row = np.ndarray(shape=(1, num_stocks * NHistData * 2), dtype=np.float32)
final_row_prices = np.ndarray(shape=(1, num_stocks * NHistData), dtype=np.float32)

# Build the taining datasets in the correct format with the matching labels. So if calculate based on last 30 stock prices then the desired
# result is the 31st. So note that the first 29 data points can't be used.
# Rather than use the stock price, use the pricing deltas.

pickle_file = "traindata.pickle"

if loadNew == True or not os.path.exists(pickle_file):
    for i in range(1, numTrainData):
        for j in range(num_stocks):
            for k in range(NHistData):
                train_dataset[i-1][j * NHistData * 2 + 2 * k] = trainDataPrices[dow30[j]][i + k] - trainDataPrices[dow30[j]][i + k - 1]
                train_dataset[i-1][j * NHistData * 2 + 2 * k + 1] = trainDataVolume[dow30[j]][i + k]
            train_labels[i-1][j] = trainDataPrices[dow30[j]][i + NHistData] - trainDataPrices[dow30[j]][i + NHistData - 1]     

    for i in range(1, numValidData):
        for j in range(num_stocks):
            for k in range(NHistData):
                valid_dataset[i-1][j * NHistData * 2 + 2 * k] = trainDataPrices[dow30[j]][i + k + numTrainData] - trainDataPrices[dow30[j]][i + k + numTrainData - 1]
                valid_dataset[i-1][j * NHistData * 2 + 2 * k + 1] = trainDataVolume[dow30[j]][i + k + numTrainData]
            valid_labels[i-1][j] = trainDataPrices[dow30[j]][i + NHistData + numTrainData] - trainDataPrices[dow30[j]][i + NHistData + numTrainData - 1]
            
    for i in range(1, numTestData):
        for j in range(num_stocks):
            for k in range(NHistData):
                test_dataset[i-1][j * NHistData * 2 + 2 * k] = trainDataPrices[dow30[j]][i + k + numTrainData + numValidData] - trainDataPrices[dow30[j]][i + k + numTrainData + numValidData - 1]
                test_dataset[i-1][j * NHistData * 2 + 2 * k + 1] = trainDataVolume[dow30[j]][i + k + numTrainData + numValidData]
            test_labels[i-1][j] = trainDataPrices[dow30[j]][i + NHistData + numTrainData + numValidData] - trainDataPrices[dow30[j]][i + NHistData + numTrainData + numValidData - 1]

    try:
      f = open(pickle_file, 'wb')
      save = {
        'train_dataset': train_dataset,
        'train_labels': train_labels,
        'valid_dataset': valid_dataset,
        'valid_labels': valid_labels,
        'test_dataset': test_dataset,
        'test_labels': test_labels,
        }
      pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
      f.close()
    except Exception as e:
      print('Unable to save data to', pickle_file, ':', e)
      raise

else:
    with open(pickle_file, 'rb') as f:
      save = pickle.load(f)
      train_dataset = save['train_dataset']
      train_labels = save['train_labels']
      valid_dataset = save['valid_dataset']
      valid_labels = save['valid_labels']
      test_dataset = save['test_dataset']
      test_labels = save['test_labels']
      del save  # hint to help gc free up memory    

for j in range(num_stocks):
    for k in range(NHistData):
            final_row_prices[0][j * NHistData + k] = trainDataPrices[dow30[j]][k + len(trainDataPrices.index) - NHistData]
            final_row[0][j * NHistData * 2 + 2 * k] = trainDataPrices[dow30[j]][k + len(trainDataPrices.index) - NHistData] - trainDataPrices[dow30[j]][k + len(trainDataPrices.index) - NHistData - 1]
            final_row[0][j * NHistData * 2 + 2 * k + 1] = trainDataVolume[dow30[j]][k + len(trainDataVolume.index) - NHistData]

print('Training set', train_dataset.shape, train_labels.shape)

print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)   

for row in train_dataset:
    for x in row:
        assert( not math.isnan(x) )
for row in train_labels:
    for x in row:
        assert( not math.isnan(x) )     
for row in test_dataset:
    for x in row:
        assert( not math.isnan(x) )
for row in test_labels:
    for x in row:
        assert( not math.isnan(x) )       
for row in valid_dataset:
    for x in row:
        assert( not math.isnan(x) )
for row in valid_labels:
    for x in row:
        assert( not math.isnan(x) )       

# This accuracy function is used for reporting progress during training, it isn't actually
# used for training.
def accuracy(predictions, labels):
  err = np.sum( np.isclose(predictions, labels, 0.0, 0.005) ) / (predictions.shape[0] * predictions.shape[1])
  return (100.0 * err)

def variable_summaries(var, name):
  """Attach a lot of summaries to a Tensor."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.scalar_summary('mean/' + name, mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.scalar_summary('stddev/' + name, stddev)
    tf.scalar_summary('max/' + name, tf.reduce_max(var))
    tf.scalar_summary('min/' + name, tf.reduce_min(var))
    tf.histogram_summary(name, var)
    
batch_size = 100
num_hidden = 64
num_labels = num_stocks

graph = tf.Graph()

# input is 30 days of dow 30 prices normalized to be between 0 and 1.
# output is 30 values for normalized next day price change of dow stocks
# use a 4 level neural network to compute this.

with graph.as_default():

  # Input data.
  tf_train_dataset = tf.placeholder(
    tf.float32, shape=(batch_size, num_stocks * NHistData * 2))
  tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
  tf_valid_dataset = tf.constant(valid_dataset)
  tf_test_dataset = tf.constant(test_dataset)
  tf_final_dataset = tf.constant(final_row)
 
  # Variables.
  with tf.name_scope('Layer1'):
      with tf.name_scope('weights'):
          layer1_weights = tf.Variable(tf.truncated_normal(
              [NHistData * num_stocks * 2, num_hidden], stddev=0.1))
          variable_summaries(layer1_weights, 'Layer1' + '/weights')
      with tf.name_scope('biases'):    
          layer1_biases = tf.Variable(tf.zeros([num_hidden]))
          variable_summaries(layer1_biases, 'Layer1' + '/biases')
  with tf.name_scope('Layer2'):
      with tf.name_scope('weights'):
          layer2_weights = tf.Variable(tf.truncated_normal(
              [num_hidden, int(num_hidden / 2)], stddev=0.1))
          variable_summaries(layer2_weights, 'Layer2' + '/weights')
      with tf.name_scope('biases'):
          layer2_biases = tf.Variable(tf.constant(1.0, shape=[int(num_hidden / 2)]))
          variable_summaries(layer2_biases, 'Layer2' + '/biases')
  with tf.name_scope('Layer3'):
      with tf.name_scope('weights'):
          layer3_weights = tf.Variable(tf.truncated_normal(
              [int(num_hidden / 2), int(num_hidden / 4)], stddev=0.1))
          variable_summaries(layer3_weights, 'Layer3' + '/weights')
      with tf.name_scope('biases'):    
          layer3_biases = tf.Variable(tf.constant(1.0, shape=[int(num_hidden  /4)]))
          variable_summaries(layer3_biases, 'Layer3' + '/biases')
  with tf.name_scope('Layer4'):
      with tf.name_scope('weights'):
          layer4_weights = tf.Variable(tf.truncated_normal(
              [int(num_hidden / 4), num_labels], stddev=0.1))
          variable_summaries(layer4_weights, 'Layer4' + '/weights')
      with tf.name_scope('biases'):    
          layer4_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))
          variable_summaries(layer4_biases, 'Layer4' + '/biases')
  
  # Model.
  def model(data, name):
    with tf.name_scope(name + 'model'):
        hidden = tf.tanh(tf.matmul(data, layer1_weights) + layer1_biases)
        tf.histogram_summary(name + 'model' + '/layer1activations', hidden)                         
        hidden = tf.tanh(tf.matmul(hidden, layer2_weights) + layer2_biases)
        tf.histogram_summary(name + 'model' + '/layer2activations', hidden)                                             
        hidden = tf.tanh(tf.matmul(hidden, layer3_weights) + layer3_biases)
        tf.histogram_summary(name + 'model' + '/layer3activations', hidden)                     
        output = tf.matmul(hidden, layer4_weights) + layer4_biases
        tf.histogram_summary(name + 'model' + '/layer4activations', output)                     
    return output
  
  # Training computation.
  logits = model(tf_train_dataset, 'training')
  beta = 0.0
  #loss = (tf.reduce_mean(
  #  tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))
  #  + tf.nn.l2_loss(layer1_weights)*beta
  #  + tf.nn.l2_loss(layer2_weights)*beta
  #  + tf.nn.l2_loss(layer3_weights)*beta
  #  + tf.nn.l2_loss(layer4_weights)*beta)
  with tf.name_scope('loss'):
     loss = (tf.nn.l2_loss( tf.sub(logits, tf_train_labels))
         + tf.nn.l2_loss(layer1_weights)*beta
         + tf.nn.l2_loss(layer2_weights)*beta
         + tf.nn.l2_loss(layer3_weights)*beta
         + tf.nn.l2_loss(layer4_weights)*beta)
     tf.scalar_summary('loss', loss)
  # loss = tf.reduce_max( tf.abs(tf.sub(logits, tf_train_labels)))

  # Setup Learning Rate Decay                          
  global_step = tf.Variable(0, trainable=False)                      
  starter_learning_rate = 0.01
  with tf.name_scope('learning_rate'):
      learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                    5000, 0.96, staircase=True)
      tf.scalar_summary('learning_rate', learning_rate)
       
  # Optimizer.
  with tf.name_scope('train'):
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    # optimizer = tf.train.AdadeltaOptimizer(starter_learning_rate).minimize(loss)
    # optimizer = tf.train.AdagradOptimizer(starter_learning_rate).minimize(loss)     # promising
    # optimizer = tf.train.AdamOptimizer(starter_learning_rate).minimize(loss)     # promising
    # optimizer = tf.train.MomentumOptimizer(starter_learning_rate, 0.99).minimize(loss) # diverges
    optimizer = tf.train.FtrlOptimizer(starter_learning_rate).minimize(loss)    # promising
    # optimizer = tf.train.RMSPropOptimizer(starter_learning_rate).minimize(loss)   # promising
  
  # Predictions for the training, validation, and test data.
  train_prediction = logits
  valid_prediction = model(tf_valid_dataset, 'validation')
  test_prediction = model(tf_test_dataset, 'test')
  next_prices = model(tf_final_dataset, 'final')
  
num_steps = 10001

with tf.Session(graph=graph) as session:
    
  # Merge all the summaries for TensorBoard and write them out to /tmp/mnist_logs (by default)
  merged = tf.merge_all_summaries()
  test_writer = tf.train.SummaryWriter('/tmp/tf/test',
                                      session.graph)

  tf.initialize_all_variables().run()

  tracelevel = tf.RunOptions.NO_TRACE      # should be NO_TRACE or FULL_TRACE
  run_options = tf.RunOptions(trace_level=tracelevel)
  
  print('Initialized')
  for step in range(num_steps):
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    batch_data = train_dataset[offset:(offset + batch_size), :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
    run_metadata = tf.RunMetadata()
    summary, _, l, predictions = session.run(
      [merged, optimizer, loss, train_prediction], feed_dict=feed_dict,
      options=run_options, run_metadata=run_metadata)
    if tracelevel != tf.RunOptions.NO_TRACE:
        # test_writer.add_run_metadata(run_metadata, 'step%d' % step)
        test_writer.add_summary(summary, i)    
    acc = accuracy(predictions, batch_labels)
#    if (acc > 45):    #if sufficiently accurate then stop.
#      print('Minibatch loss at step %d: %f' % (step, l))
#      print('Minibatch accuracy: %.1f%%' % acc)        
#      break;
    if (step % 1000 == 0):
      print('Minibatch loss at step %d: %f' % (step, l))
      print('Minibatch accuracy: %.1f%%' % acc)
      if numValidData > 0:
          print('Validation accuracy: %.1f%%' % accuracy(
              valid_prediction.eval(), valid_labels))
  if numTestData > 0:        
      print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))

  predictions = next_prices.eval() * factorsPrices

  print("Stock    Last Close  Predict Chg   Predict Next      Current     Current Chg       Error")
  i = 0
  total_error = 0.0
  for x in dow30:
      yhfeed = Share(x)
      currentPrice = float(yhfeed.get_price())
      error = abs(predictions[0][i] - (currentPrice - final_row_prices[0][i * NHistData + NHistData - 1] * factorsPrices[i]))
      print( "%-6s  %9.2f  %9.2f       %9.2f       %9.2f     %9.2f     %9.2f" % (x,
             final_row_prices[0][i * NHistData + NHistData - 1] * factorsPrices[i],
             predictions[0][i],
             final_row_prices[0][i * NHistData + NHistData - 1] * factorsPrices[i] + predictions[0][i],
             currentPrice,
             currentPrice - final_row_prices[0][i * NHistData + NHistData - 1] * factorsPrices[i],
             error) )
      total_error += error
      i = i + 1
  
  print("Total Error: %9.2f" % (total_error))
