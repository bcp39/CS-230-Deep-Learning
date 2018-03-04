import os
import requests
import json
import urllib
import pandas as pd
import csv

Storage_directory = os.getcwd() +"\StockData\\" 

def extractCleanedData(ticker):
	#Returns a list of rows in the CSV file "tickerCOmpleted.csv"
	listOfData = []
	ticker_directory = ticker + "\\"
	with open(Storage_directory + ticker_directory + ticker + "Completed.csv", 'rU') as csvfile:
		reader = csv.reader(csvfile, delimiter=',' )
		csv_data = list(reader)
		listOfData = csv_data
	return listOfData



if __name__ == "__main__":

	newfile = open(os.getcwd() + "\\" + "Dataset.csv", 'wb')
	list_of_stocks = []
	with newfile:
		writer = csv.writer(newfile)
		for filename in os.listdir(Storage_directory):
			list_of_stocks.append(filename)
		count = len(list_of_stocks)
		for ticker in list_of_stocks:
			templist = extractCleanedData(ticker)
			writer.writerows(templist)
			count = count - 1
			print str(count) + " number of stocks left to compile!" 

