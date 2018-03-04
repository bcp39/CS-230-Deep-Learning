import os
import requests
import json
import urllib
import pandas as pd
import csv

# A CSV list of stocks named "StockList.csv" in the same directory should exist for the data to pulled 

AlphaVantage_API_Key = "MM494M5Q1QQJCXRX"
currentDirectory = os.getcwd() +"\StockData\\" 

def getDailyPriceHistory(ticker):
	"""
	Downloads a CSV file that contains the day-to-day price history of a stock from alpha vantage
	Input: String containing ticker symbol
	Output: CSV file of daily price history saved to current working directory
	"""
	ticker_directory = ticker + "\\"
	directory = currentDirectory + ticker_directory
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
	directory = currentDirectory + ticker_directory
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



if __name__ == "__main__":
	if not os.path.exists(currentDirectory): #make the stockdata folder if it does not exist
		os.makedirs(currentDirectory)
	with open("StockList.csv", 'rU') as csvfile:
		reader = csv.reader(csvfile, delimiter=',' )
		csv_data = list(reader)
		for rowNumber in range(0, len(csv_data)):
			while True:
				try:
		 			ticker = csv_data[rowNumber][0]
		 			updateData(ticker) #updating the price for each ticker
		 			print str(len(csv_data)-rowNumber) + " stocks left to process!"
		 		except: #retry if connection error occurs
		 			print "Retrying..."
		 			continue
		 		break