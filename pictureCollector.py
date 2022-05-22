# Web scraper in Python

# Github Repository Link
# https://github.com/almehdiKrisni/Projet_RI

########################################### IMPORTS ##############################################

# Necessary imports
from ctypes import resize
from sys import argv
from utils import *

########################################### COLLECTOR ################################################

# Parameters
request = ["fat", "saturated fat", "sodium", "sugars"]
advRequest = ["fat", "saturated fat", "sodium", "sugars", "carbohydrates", "cholesterol", "calcium", "iron", "magnesium", "potassium", "dietary fiber"]

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
            download_pictures(cStart + i * cRange, cRange)
            reshape_pictures(cStart + i * cRange, cRange, (300,300))
        

if __name__ == "__main__" :
    main(argv)