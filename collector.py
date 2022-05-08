# Web scraper in Python
# Tutorial from - https://www.topcoder.com/thrive/articles/web-crawler-in-python

########################################### IMPORTS ##############################################

# Necessary imports
from sys import argv
from utils import *

########################################### COLLECTOR ################################################

# Parameters
request = ["fat", "saturated fat", "sodium", "sugars"]

def main(argv) :
    # We check if the number of parameters is correct
    if (len(argv) != 4) :
        print("You need to pass 3 parameters to the script.\n$file counterStart counterRange iterNumber")
    else :
        # We start collecting data
        print(argv[0])
        cStart = int(argv[1]) # Counter start
        cRange = int(argv[2]) # Counter range
        lp = int(argv[3]) # Number of iterations

        # We collect the data and write it into a csv file
        for i in range(lp) :
            fileP = "data/recipedata_" + str(cStart + i * cRange) + "_" + str(cStart + (i + 1) * cRange - 1) + ".csv"
            recipeCollector(cStart + i * cRange, cRange, request, toCSV=True, filepath=fileP)

if __name__ == "__main__" :
    main(argv)