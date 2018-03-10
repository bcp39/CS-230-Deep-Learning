# Using python 3.5.2 to run

import tensorflow as tf
import os
import csv
import numpy as np


currentDirectory = os.getcwd() + "\\"
number_of_columns = 0

with open(currentDirectory + "Dataset.csv", 'rU') as csvfile:
		reader = csv.reader(csvfile, delimiter=',' )
		for row in reader:
			number_of_columns = len(row)
			break
print (number_of_columns)

filename_queue = tf.train.string_input_producer([currentDirectory + "train\\" + "CombinedTrainData.csv"],shuffle=True)
line_reader = tf.TextLineReader(skip_header_lines=False)
_, csv_row = line_reader.read(filename_queue)
record_defaults = []
for x in range(number_of_columns-1):
    record_defaults.append([0.0])
record_defaults.append([0])
all_columns = tf.decode_csv(csv_row, record_defaults=record_defaults)

covariates = []
for x in range(0, len(all_columns) - 1):
    covariates.append(all_columns[x])
features = tf.stack(covariates)
labels = tf.stack(all_columns[len(all_columns)-1])


learning_rate = 0.01
batch_size = 128
n_epochs = 25

X = tf.placeholder(tf.float32, [batch_size, 44])
Y = tf.placeholder(tf.int32, [batch_size, 1])



with tf.Session() as sess:
    tf.global_variables_initializer().run()

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    for iteration in range(1, 11):
        
        example, label = sess.run([features, all_columns[number_of_columns-1]])
        print(example, label)
        print (type(example), type(label))
        print (example.shape, label.shape)
    coord.request_stop()
    coord.join(threads)

if __name__ == "__main__":
	pass