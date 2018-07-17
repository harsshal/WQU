__author__ = 'harsshal'

import sqlite3
import pandas as pd
import data
import tensorflow as tf
import numpy as np

def simple_combo(signals):
    '''
    regular simple addition combo
    which simply adds all the signals

    :type signals: list of dataframes of scenarios
    :return: optimal combination of scenarios
    '''

    df = signals[0]
    for i in range(1,len(signals)):
        df += signals[i]
    return signals[1]
    return df

def ann_combo(signals):
    '''
    STILL IN WORKS...

    This funtion uses tensorflow to create a
    artificial neural network which can combine scenarios optimally.

    :type signals: list of dataframes of scenarios
    :return: optimal combination of scenarios
    '''

    y_true = (signals[0]>signals[0].shift(1)).fillna(0).astype(int).replace(0,-1)

    input_dim = len(signals[0].columns) * len(signals) + 1 # for close
    output_dim = 2 # buy or sell

    x = tf.placeholder('float', [None, input_dim])
    y = tf.placeholder(tf.float32, [None, output_dim])
    weights = tf.Variable(tf.zeros([input_dim, output_dim]))
    biases = tf.Variable(tf.zeros([output_dim]))

    logits = tf.matmul(x, weights) + biases
    y_pred = tf.nn.softmax(logits)

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                        labels=y)
    cost = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(cost)

    session = tf.Session()
    session.run(tf.global_variables_initializer())

    for i in range(10):
        # Get a batch of training examples.
        # x_batch now holds a batch of images and
        # y_true_batch are the true labels for those images.
        x_batch = np.vstack([signals[0].iloc[i],signals[1].iloc[i],signals[2].iloc[i]])\
                        .reshape(1,len(signals[0].columns) * len(signals))[0]
        y_true_batch = y_true.iloc[i].as_matrix()

        # Put the batch into a dict with the proper names
        # for placeholder variables in the TensorFlow graph.
        # Note that the placeholder for y_true_cls is not set
        # because it is not used during training.
        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch}

        # Run the optimizer using this batch of training data.
        # TensorFlow assigns the variables in feed_dict_train
        # to the placeholder variables and then runs the optimizer.
        session.run(optimizer, feed_dict=feed_dict_train)

    # Use TensorFlow to compute the accuracy.
    correct_prediction = tf.equal(y_pred, y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    feed_dict_test = {x: signals[0].merge(signals[1],on='date').merge(signals[2],on='date').as_matrix(),
                  y_true: y}
    acc = session.run(accuracy, feed_dict=feed_dict_test)

    # Print the accuracy.
    print("Accuracy on test-set: {0:.1%}".format(acc))

    return signals[0]


def main():
    with sqlite3.connect('alpha.db') as conn:
        conn = sqlite3.connect('alpha.db')
        c = conn.cursor()

        query = 'select * from close'
        close = pd.read_sql_query(query,conn,index_col='date',
                                  parse_dates={'date': {'format': '%Y-%m-%d %H:%M:%S'}})

        query = 'select * from dip'
        dip = pd.read_sql_query(query,conn,index_col='date',
                                parse_dates={'date': {'format': '%Y-%m-%d %H:%M:%S'}})

        query = 'select * from bol_finger'
        bol_finger = pd.read_sql_query(query,conn,index_col='date',
                                       parse_dates={'date': {'format': '%Y-%m-%d %H:%M:%S'}})

        #position = simple_combo([dip,bol_finger])
        position = ann_combo([close,dip,bol_finger])

        portfolio = pd.Panel({'price':close,'pos':position})

        data.find_kpi(portfolio)


if __name__ == '__main__':
    main()