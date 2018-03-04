import os
import requests
import json
import urllib
import pandas as pd
import csv

Storage_directory = os.getcwd() +"\StockData\\" 


def cleanKeyRatiosData(ticker):
		
	"""
	Processing the data from the key ratios CSV file
	"""
	ticker_directory = ticker + "\\"
	#Data to be fetched, add more lists as needed
	year_categories = []
	earningsPerShareUSD = []
	sharesMil = []
	netIncomeUSDMil = []

	with open(Storage_directory + ticker_directory + ticker + "Fundamentals.csv", 'rU') as csvfile:
		reader = csv.reader(csvfile, delimiter=',' )
		csv_data = list(reader)

		try: 
			year_categories = csv_data[2] #the categories of the years are stored in the 3rd row in the data fetched
		except: 
			year_categories = [] # sometimes, the fundamental data is missing

		#Extracting Earnings Per Share data
		for row in csv_data:
			if len(row)!= 0 and row[0] == "Earnings Per Share USD":
				earningsPerShareUSD = row
				break

		#Extracting Shares in Million data
		for row in csv_data:
			if len(row)!= 0 and row[0] == "Shares Mil":
				sharesMil = row
				break

		#Extracting Net Income USD Mil data
		for row in csv_data:
			if len(row)!= 0 and row[0] == "Net Income USD Mil":
				netIncomeUSDMil = row
				break

	#Writing to a new CSV file for processing
	cleanedData = []
	cleanedData.append(year_categories)
	cleanedData.append(earningsPerShareUSD)
	cleanedData.append(sharesMil)
	cleanedData.append(netIncomeUSDMil)

	newfile = open(Storage_directory + ticker_directory + ticker + "CleanedFundamentals.csv", 'wb')
	with newfile:
		writer = csv.writer(newfile)
		writer.writerows(cleanedData)

	print("Cleaned " + ticker + " Fundamentals Data!")



def for230(ticker, threshold):
	"""
	Putting the Data Together for 1 stock for deep learning
	"""
	ticker_directory = ticker + "\\"
	data = []
	with open(Storage_directory + ticker_directory + ticker + "DailyPrice.csv", 'rU') as csvfile:
		with open(Storage_directory + ticker_directory + ticker + "CleanedFundamentals.csv", 'rU') as csvfile2:
			data = []
			reader = csv.reader(csvfile, delimiter=',' )
			reader2 = csv.reader(csvfile2, delimiter=',')
			csv_data = list(reader)
			ratios_data = list(reader2)
			csv_data[:] = [item for item in csv_data if len(item) != 0] #remove empty ]rows

			for rowNumber in range(6,len(csv_data)-19): #skip the first 5 rows and stop at the last 20 rows
				temprow = []
				for i in reversed(range(20)):
					temprow.append(csv_data[rowNumber+i][4]) #column 5 refers to closing price for 20 consecutive days
				for i in reversed(range(20)):
					temprow.append(csv_data[rowNumber+i][5]) #column 6 refers to volume for 20 consecutive days

				#Extracting Shares in Mil
				for row in ratios_data:
					if len(row)!= 0 and row[0] == "Shares Mil":
						for x in range(1,len(ratios_data[0])): #looping through the dates
							if ratios_data[0][x][0:4] in csv_data[rowNumber][0]: #checking the latest date to add in the earnings per share
								if row[x] != '':
									temprow.append(row[x])
								else:
									temprow.append('')
								break
						break
				if len(temprow) == 40:
					temprow.append('NIL')

				#Extracting Net Income USD Mil data
				for row in ratios_data:
					if len(row)!= 0 and row[0] == "Net Income USD Mil":
						for x in range(1,len(ratios_data[0])): #looping through the dates
							if ratios_data[0][x][0:4] in csv_data[rowNumber][0]: #checking the latest date to add in the earnings per share
								if row[x] != '':
									temprow.append(row[x])
								else:
									temprow.append('')
								break
						break
				if len(temprow) == 41:
					temprow.append('NIL')

				percentGainOver5days = 0
				if float(csv_data[rowNumber][4]) == 0.000: #sometimes, the price can be recorded as 0
					percentGainOver5days = ((float(csv_data[rowNumber-5][4]) - float(csv_data[rowNumber][4]))/0.0009) 
				else: 
					percentGainOver5days = ((float(csv_data[rowNumber-5][4]) - float(csv_data[rowNumber][4]))/float(csv_data[rowNumber][4])) 
				if percentGainOver5days >= threshold:
					temprow.append(1)
				else:
					temprow.append(0)
				data.append(temprow)
	newfile = open(Storage_directory + ticker_directory + ticker + "Completed.csv", 'wb')
	with newfile:
		writer = csv.writer(newfile)
		writer.writerows(data)
	print "Data Processed for " + ticker

if __name__ == "__main__":
	list_of_stocks = []
	for filename in os.listdir(Storage_directory):
		list_of_stocks.append(filename)
	count = len(list_of_stocks)
	for ticker in list_of_stocks:
		cleanKeyRatiosData(ticker)
		for230(ticker,0.5)
		count = count - 1
		print str(count) + " number of stocks left to build!" 
