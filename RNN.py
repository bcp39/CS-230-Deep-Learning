# Using python 3.5.2 to run


import os
import csv
import numpy as np
import pandas as pd
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import math

currentDirectory = os.getcwd() + "\\"
number_of_columns = 45

"""RUN THE COMMENTED CODE BELOW IF NEED TO FIND OUT THE NUMBER OF COLUMNS IN THE DATA"""
# print ("Checking number of columns in dataset...")
# with open(currentDirectory + "Dataset.csv", 'rU') as csvfile:
# 		reader = csv.reader(csvfile, delimiter=',' )
# 		for row in reader:
# 			number_of_columns = len(row)
# 			break
# print ("Number of columns = ", number_of_columns)


def rnn():

    learning_rate = 0.01
    batch_size = 400000
    iter_num = 50
    display_step = 1
    beta = 0.1
    threshold = 0.5

    seed = 5
    np.random.seed(seed)
    tf.set_random_seed(seed)

    print ("Loading training data...")
    train_data = pd.read_csv(currentDirectory + "train\\" + "CombinedTrainData.csv", header = None)
    print ("Shuffling training data...")
    train_data = train_data.iloc[np.random.permutation(len(train_data))]
    train_X = train_data.iloc[:,0:number_of_columns-1] #selecting the feature columns
    train_X = np.array(train_X)

    train_Y = np.array(train_data.iloc[:,number_of_columns-1:]) # selecting the label column
    train_Y = np.array(train_Y)


    # print ("Loading validation data...")
    # dev_data = pd.read_csv(currentDirectory + "dev\\" + "CombinedDevData.csv")
    # dev_X = dev_data.iloc[:,0:number_of_columns-1] #selecting the feature columns
    # #dev_X = min_max_normalized(np.array(dev_X))
    # dev_Y = np.array(dev_data.iloc[:,number_of_columns-1:]) # selecting the label column
    
    print ("Loading test data...")
    test_data = pd.read_csv(currentDirectory + "test\\" + "CombinedTestData.csv")
    test_X = test_data.iloc[:,0:number_of_columns-1] #selecting the feature columns
    test_X = np.array(test_X)
    #dev_X = min_max_normalized(np.array(dev_X))
    test_Y = np.array(test_data.iloc[:,number_of_columns-1:]) # selecting the label column
    test_Y = np.array(test_Y)



    #Defining the model framework
    w1 =tf.get_variable(shape = [44,5], name = "weights_layer_1",  initializer=tf.contrib.layers.xavier_initializer())
    b1 = tf.Variable(tf.zeros([1]), name = "bias_layer_1")

    w2 =tf.get_variable(shape = [5,5], name = "weights_layer_2",  initializer=tf.contrib.layers.xavier_initializer())
    u2 = tf.get_variable(shape =[44,5], name = "2nd_weights_layer_2",  initializer=tf.contrib.layers.xavier_initializer())
    b2 = tf.Variable(tf.zeros([1]), name = "bias_layer_2")

    w3 =tf.get_variable(shape = [5,1], name = "weights_layer_3",  initializer=tf.contrib.layers.xavier_initializer())
    u3 = tf.get_variable(shape = [44,1], name = "2nd_weights_layer_3",  initializer=tf.contrib.layers.xavier_initializer())
    b3 = tf.Variable(tf.zeros([1]), name = "bias_layer_3")

    X = tf.placeholder(tf.float32,[None, 44], name = 'data')
    Y = tf.placeholder(tf.float32, [None, 1], name = 'target')

    

    h1 = tf.nn.relu(tf.matmul(X, w1) + b1) #layer_1
    h2 = tf.nn.relu(tf.matmul(h1, w2) + tf.matmul(X,u2) + b2) #layer_2
    #pred = tf.nn.softmax(tf.matmul(h2, w3) + tf.matmul(X,u3) + b3) #layer_3
    pred = tf.sigmoid(tf.matmul(h2, w3) + tf.matmul(X,u3) + b3) #layer_3
    # Minimize error using cross entropy (1st cost function)
    #cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=Y))



    # (2nd cost function used)
    cost = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=pred, targets=Y, pos_weight = beta)) 

    #To calculate precision, which is the main metric
    rounded_pred = tf.cast(pred>=threshold, dtype = tf.float32)
    total_ones  = tf.reduce_sum(rounded_pred)
    true_positive = tf.reduce_sum(tf.multiply(rounded_pred, Y)) 

    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)



    #correct = tf.cast(tf.equal(tf.round(pred), Y), dtype=tf.float32)
    #accuracy = tf.metrics.precision(labels = Y, predictions = tf.round(pred))
    writer = tf.summary.FileWriter('./graphs', tf.get_default_graph())
    train_precision = []
    test_precision = []
    epoch_array = []


    init = tf.global_variables_initializer()
    init_l = tf.local_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        sess.run(init_l)
        for epoch in range(iter_num):
            avg_cost = 0.0
            tp = 0
            tp_fp = 0.000000000001 # to prevent division by 0
            test_tp = 0
            test_tp_fp = 0.000000000001 # to prevent division by 0
            total_batch = math.ceil(train_X.shape[0]/batch_size)
            for i in range(total_batch - 1):
                batch_xs = train_X[i*batch_size:batch_size*i+batch_size,:]
                batch_ys = train_Y[i*batch_size:batch_size*i+batch_size,:]

                test_xs = test_X[i*batch_size:batch_size*i+batch_size,:]
                test_ys = test_Y[i*batch_size:batch_size*i+batch_size,:]

            #     # convert into a matrix, and the shape of the placeholder to correspond
            #     temp_train_acc = sess.run(accuracy, feed_dict={X: batch_xs, Y: batch_ys})
            #     temp_test_acc = sess.run(accuracy, feed_dict={X: dev_X, Y: dev_Y})
            #     # recode the result
            #     loss_trace.append(temp_loss)
            #     train_acc.append(temp_train_acc)
            #     validation_acc.append(temp_test_acc)
            # output

            # Run optimization op (backprop) and cost op (to get loss value)
                #_, c, p, temp_train_acc = sess.run([optimizer, cost, pred, accuracy], feed_dict={X: batch_xs, Y: batch_ys})
                _, c, p, ones, correct_ones = sess.run([optimizer, cost, pred, total_ones, true_positive], feed_dict={X: batch_xs, Y: batch_ys})
                test_ones, test_correct_ones = sess.run([total_ones, true_positive], feed_dict={X: test_xs, Y: test_ys})
                avg_cost += c 
                tp += correct_ones
                tp_fp += ones
                test_tp +=  test_correct_ones
                test_tp_fp += test_ones

                


            

            #including the last batch
            batch_xs = train_X[batch_size*(total_batch - 1):,:]
            batch_ys = train_Y[batch_size*(total_batch - 1):,:]
            #_, c, p = sess.run([optimizer, cost, pred], feed_dict={X: batch_xs, Y: batch_ys})
            #_, c, p, temp_train_acc = sess.run([optimizer, cost, pred, accuracy], feed_dict={X: batch_xs, Y: batch_ys})
            _, c, p, ones, correct_ones = sess.run([optimizer, cost, pred, total_ones, true_positive], feed_dict={X: batch_xs, Y: batch_ys})
            test_ones, test_correct_ones = sess.run([total_ones, true_positive], feed_dict={X: test_xs, Y: test_ys})
            avg_cost += c 
            tp += correct_ones
            tp_fp += ones
            test_tp +=  test_correct_ones
            test_tp_fp += test_ones
            #temp_train_acc = sess.run(accuracy, feed_dict={X: batch_xs, Y: batch_ys})
            #train_acc.append(temp_train_acc[0])

            #print ("Tensorflow precision: ", temp_train_acc[0])

            # Display logs per epoch step
            if (epoch+1) % display_step == 0:
                #print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))
                #print('epoch: {:4d} cost = : {:.9f} train_precision: {:.9f} '.format(epoch + 1, avg_cost, temp_train_acc[0]))
                print ("Number of true positives for test: ", test_tp)
                print ("Number of ones predicted for test: ", test_tp_fp)
                print('epoch: {:4d} cost = : {:.9f} train_precision: {:.9f} test_precision: {:.9f}'.format(epoch + 1, avg_cost, float(tp)/float(tp_fp), float(test_tp)/float(test_tp_fp)))
                epoch_array.append(epoch+1)
                train_precision.append(float(tp)/float(tp_fp))
                test_precision.append(float(test_tp)/float(test_tp_fp))
        print("Optimization Finished!")
    
    with open("results.csv", "w") as csv_file:
        results = csv.writer(csv_file, delimiter=',')
        results.writerow(epoch_array)
        results.writerow(train_precision)
        results.writerow(test_precision)
    writer.close()



    

if __name__ == "__main__":
	rnn()

            