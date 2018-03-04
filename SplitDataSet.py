import os
import csv
import random

random.seed(10)
currentDirectory = os.getcwd() + "\\"

# 7103560 training examples
# 26970
def splitData(csvfilename, train_frac, dev_frac, test_frac):  
	"""
	Takes a CSV file e.g. Dataset.csv (file must be in current directory) as an input and outputs the train, dev and test sets in their respective directories
	"""
	total_frac = train_frac + dev_frac + test_frac
	if total_frac != 1:
		print "Your data split does not sum to 1 !"
		quit()

	print "This will take a few minutes to run..."
	if not os.path.exists(currentDirectory + "train\\"): #make the train folder if it does not exist
		os.makedirs(currentDirectory+ "train\\")
	if not os.path.exists(currentDirectory + "dev\\"): #make the dev folder if it does not exist
		os.makedirs(currentDirectory+ "dev\\")
	if not os.path.exists(currentDirectory + "test\\"): #make the test folder if it does not exist
		os.makedirs(currentDirectory+ "test\\")

	#Seperate the dataset file into 2 classes because of the uneven distribution between the 2 classes (1s consists of only 0.367% of the entire dataset)
	onesFile = open(currentDirectory + "LabeledOnes.csv",  'wb')
	zerosFile = open(currentDirectory + "LabeledZeros.csv", 'wb')
	total_ones = 0
	total_zeros = 0

	with open(csvfilename, 'rU') as csvfile:
		reader = csv.reader(csvfile, delimiter="," )
		
		with onesFile:
			writerOnes = csv.writer(onesFile)
			with zerosFile:
				writerZeros = csv.writer(zerosFile)
				for row in reader:
					if row[-1] == "1":
						writerOnes.writerows([row])
						total_ones = total_ones + 1
					else:
						writerZeros.writerows([row])
						total_zeros = total_zeros + 1

	print str(total_ones) + " and " + str(total_zeros) + " rows of data split according to labels 1 and 0 respectively !"

	list_of_ones = range(total_ones)
	list_of_zeros = range(total_zeros)
	random.shuffle(list_of_ones)
	random.shuffle(list_of_zeros)

	#Spliting for the distribution with labels 1
	split_1 = int(train_frac * len(list_of_ones))
	split_2 = int((train_frac + dev_frac) * len(list_of_ones))
	train_ones = list_of_ones[:split_1]
	dev_ones = list_of_ones[split_1:split_2]
	test_ones = list_of_ones[split_2:]
	train_ones.sort()
	dev_ones.sort()
	test_ones.sort()

	#Spliting for the distribution with labels 0
	split_3 = int(train_frac * len(list_of_zeros))
	split_4 = int((train_frac + dev_frac) * len(list_of_zeros))
	train_zeros = list_of_zeros[:split_3]
	dev_zeros = list_of_zeros[split_3:split_4]
	test_zeros = list_of_zeros[split_4:]
	train_zeros.sort()
	dev_zeros.sort()
	test_zeros.sort()

	with open("LabeledOnes.csv", 'rU') as csvfile:
		reader = csv.reader(csvfile, delimiter="," )
		count = 0
		iterator1 = 0
		iterator2 = 0
		iterator3 = 0
		onesFileTrain = open(currentDirectory + "train\\" "LabeledOnesTrain.csv",  'wb')
		onesFileDev = open(currentDirectory + "dev\\" "LabeledOnesDev.csv",  'wb')
		onesFileTest = open(currentDirectory + "test\\" "LabeledOnesTest.csv",  'wb')
		with onesFileTrain:
			writerOnesTrain = csv.writer(onesFileTrain)
			with onesFileDev:
				writerOnesDev = csv.writer(onesFileDev)
				with onesFileTest:
					writerOnesTest = csv.writer(onesFileTest)
					for row in reader:
						if count == train_ones[iterator1]:
							writerOnesTrain.writerows([row])
							if iterator1 < split_1:
								iterator1 = iterator1 + 1
						elif count == dev_ones[iterator2]:
							writerOnesDev.writerows([row])
							if iterator2 < split_2 - split_1:
								iterator2 = iterator2 + 1
						elif count == test_ones[iterator3]:
							writerOnesTest.writerows([row])
							if iterator3 < len(list_of_ones) - split_2:
								iterator3 = iterator3 + 1
						count = count + 1
		print str(iterator1+iterator2+iterator3), " LabeledOnes have been split into Training, Dev and Test!"


	with open("LabeledZeros.csv", 'rU') as csvfile:
		reader = csv.reader(csvfile, delimiter="," )
		count = 0
		iterator1 = 0
		iterator2 = 0
		iterator3 = 0
		zerosFileTrain = open(currentDirectory + "train\\" "LabeledZerosTrain.csv",  'wb')
		zerosFileDev = open(currentDirectory + "dev\\" "LabeledZerosDev.csv",  'wb')
		zerosFileTest = open(currentDirectory + "test\\" "LabeledZerosTest.csv",  'wb')
		with zerosFileTrain:
			writerZerosTrain = csv.writer(zerosFileTrain)
			with zerosFileDev:
				writerZerosDev = csv.writer(zerosFileDev)
				with zerosFileTest:
					writerZerosTest = csv.writer(zerosFileTest)
					for row in reader:
						if count == train_zeros[iterator1]:
							writerZerosTrain.writerows([row])
							if iterator1 < split_3:
								iterator1 = iterator1 + 1
						elif count == dev_zeros[iterator2]:
							writerZerosDev.writerows([row])
							if iterator2 < split_4 - split_3:
								iterator2 = iterator2 + 1
						elif count == test_zeros[iterator3]:
							writerZerosTest.writerows([row])
							if iterator3 < len(list_of_zeros) - split_4:
								iterator3 = iterator3 + 1
						count = count + 1
		print str(iterator1+iterator2+iterator3), " LabeledZeros have been split into Training, Dev and Test!"

	#Combining the data in each of the test,dev and train folders respectively
	train_folder_ones = currentDirectory + "train\\" "LabeledOnesTrain.csv"
	train_folder_zeros = currentDirectory + "train\\" "LabeledZerosTrain.csv"
	combinedTrainData = open(currentDirectory + "train\\" "CombinedTrainData.csv",  'wb')
	writer = csv.writer(combinedTrainData)
	count = 0
	with open(train_folder_ones, 'rU') as csvfile:
		reader1 = csv.reader(csvfile, delimiter="," )
		for row in reader1:
			writer.writerows([row])
			count = count + 1
	with open(train_folder_zeros, 'rU') as csvfile:
		reader2 = csv.reader(csvfile, delimiter="," )
		for row in reader2:
			writer.writerows([row])
			count = count + 1
	print count, " of rows of training data combined"


	dev_folder_ones = currentDirectory + "dev\\" "LabeledOnesDev.csv"
	dev_folder_zeros = currentDirectory + "dev\\" "LabeledZerosDev.csv"
	combinedDevData = open(currentDirectory + "dev\\" "CombinedDevData.csv",  'wb')
	writer = csv.writer(combinedDevData)
	count = 0
	with open(dev_folder_ones, 'rU') as csvfile:
		reader1 = csv.reader(csvfile, delimiter="," )
		for row in reader1:
			writer.writerows([row])
			count = count + 1
	with open(dev_folder_zeros, 'rU') as csvfile:
		reader2 = csv.reader(csvfile, delimiter="," )
		for row in reader2:
			writer.writerows([row])
			count = count + 1
	print count, " of rows of dev data combined"

	test_folder_ones = currentDirectory + "test\\" "LabeledOnesTest.csv"
	test_folder_zeros = currentDirectory + "test\\" "LabeledZerosTest.csv"
	combinedTestData = open(currentDirectory + "test\\" "CombinedTestData.csv",  'wb')
	writer = csv.writer(combinedTestData)
	count = 0
	with open(test_folder_ones, 'rU') as csvfile:
		reader1 = csv.reader(csvfile, delimiter="," )
		for row in reader1:
			writer.writerows([row])
			count = count + 1
	with open(test_folder_zeros, 'rU') as csvfile:
		reader2 = csv.reader(csvfile, delimiter="," )
		for row in reader2:
			writer.writerows([row])
			count = count + 1
	print count, " of rows of test data combined"





if __name__ == "__main__":
	splitData("Dataset.csv", 0.8, 0.1, 0.1)
