import os
import requests
import json
import urllib
import pandas as pd
import csv

AlphaVantage_API_Key = "MM494M5Q1QQJCXRX"
Storage_directory = "C:\Users\Brandon\Desktop\Winter 2018\CS 230\Project\Stock Data\\" 



def getDailyPriceHistory(ticker):
	"""
	Downloads a CSV file that contains the day-to-day price history of a stock from alpha vantage
	Input: String containing ticker symbol
	Output: CSV file of daily price history saved to current working directory
	"""
	ticker_directory = ticker + "\\"
	directory = Storage_directory + ticker_directory
	if not os.path.exists(directory): #checking if directory already exists, otherwise create a new one
		os.makedirs(directory)

	
	time_series_url = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=" + ticker + "&outputsize=full&apikey= " + AlphaVantage_API_Key+ "&datatype=csv"
	testfile = urllib.URLopener()
	testfile.retrieve(time_series_url, directory + ticker + "DailyPrice.csv")
	print ticker + " price history obtained!"


def getKeyRatios(ticker):
	"""
	Downloads a CSV file that contains the key ratios for a stock for the past 10 years from morningstar
	Input: String containing ticker symbol
	Output: CSV file of key ratios 10-year history saved to current working directory
	"""
	ticker_directory = ticker + "\\"
	directory = Storage_directory + ticker_directory
	if not os.path.exists(directory): #checking if directory already exists, otherwise create a new one
		os.makedirs(directory)

	url = "http://financials.morningstar.com/ajax/exportKR2CSV.html?t=" + ticker
	testfile = urllib.URLopener()
	testfile.retrieve(url, directory + ticker + "Fundamentals.csv")
	print ticker + " fundamentals history obtained!"



def updateData(ticker):
	"""
	Update financial data by redownloading the csv files
	Input: String containing ticker symbol
	Output: Latest CSV files saved to current working directory
	"""
	getDailyPriceHistory(ticker)
	getKeyRatios(ticker)



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
	update = raw_input("Do you want to re-fetch data? Enter 'y' to refetch, otherwise enter any other key\n")
	if update == "y":
		with open("StockList.csv", 'rU') as csvfile:
		 	reader = csv.reader(csvfile, delimiter=',' )
		 	csv_data = list(reader)
		 	for rowNumber in range(1497, len(csv_data)):
		 		ticker = csv_data[rowNumber][0]
		 		updateData(ticker) #updating the price for each ticker
		 		cleanKeyRatiosData(ticker)
		 		for230(ticker,0.5)
		 		print str(len(csv_data)-rowNumber) + " stocks left to process!"

	