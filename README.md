# Meme-Stock & r/WallStreetBets Portfolio Analysis
In this project, we will use a combination of descriptive statistics with financial theory (along with a little bit of mathematics) to determine whether the r/WallStreetBets subreddit can generate profitable investment strategies.

## Goal

1. Scrape through the post of the r/WallStreetBets subreddit and classify the most discussed stocks/assets on the platform (completed in another project)
2. Build a portfolio of popular meme-stocks and other assets that are popular among the r/WallStreetBets community
3. Import the financial data of the assets in our portfolio while using the data to construct important financial statistics (returns, volatility, etc.)
4. Using descriptive statistics to find alpha, beta, covariance, sharpe ratio, and the correlation matrix
5. Use an algorithm to generate the optimal portfolio for our asset class

## Data

The data for this project primary originates from Yahoo Finance, which provides historical financial data for free.

## Enviroment and Tools

The following are the modules we will use in this notebook. However, the program relies on many more dependencies than what is shown here. Please be sure to set up a virtual enviroment and install the [requirements.txt](https://github.com/KidQuant/Meme-Stock-r-WallStreetBets-Portfolio-Analysis/blob/main/requirements.txt) file before running this programming on your own.

We will also be using functions and methods available from different projects, using the [MPT Functions](https://github.com/KidQuant/Meme-Stock-r-WallStreetBets-Portfolio-Analysis/blob/main/MPT_Functions.ipynb)

1. Numpy
2. Pandas
3. Datetime
4. Matplotlib
5. Seaborn
6. Pandas_datareader
7. Functools
8. Tabulate

## Other Related Finance Project

1. [Pairs Trading With Python](https://github.com/KidQuant/Pairs-Trading-With-Python)
2. [Modeling Vanilla Interest Rate Swap](https://github.com/KidQuant/Modeling-Vanilla-Interest-Rate-Swaps)
3. [401K Optimization Using Modern Portfolio Theory](https://github.com/KidQuant/401K-Optimization-Using-Modern-Portfolio-Theory)
