#################################### Utilitaries #######################################

###################################### Imports #########################################
import requests as rq
import lxml
from bs4 import BeautifulSoup as bs
import string
import pandas as pd
import pprint
import re
import time
import csv

################################## Scraping functions ###################################

# Function to scrape website data from "allrecipes.com"with a number of pages to visit and the researched values
# Returns a dictionnary or a pandas database (request with parameter) of the scraping
# The counter starts from the adress
def recipeCollector(startCounter, counter, research, toCSV=False, filepath="data/default_file.csv") :
    # We create the dictionnary and root adress
    nutr = []
    root = "https://www.allrecipes.com/recipe/"

    # Note - It is possible some addresses won't return any recipe
    # The basic data scraped is 

    # Header creation (to avoid any problem when accessing the website)
    headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36 QIHU 360SE'
    }

    # Data extraction
    for c in range(startCounter, startCounter + counter) :
        # We print the progress
        print(c - startCounter, "/", counter, end="\r")

        # We get the website data then parse it
        adr = root + str(c)
        f = rq.get(adr)
        recipe = bs(f.content, 'lxml')

        # We save the recipe name
        recipeName = recipe.find('h1', {"class" : "headline heading-content elementFont__display"})

        # We check if the website contains a recipe or an error with the title
        # If it contains a title, it means there is a recipe and values
        if (recipeName != None) :
            
            # We create a sub-dictionnary that will contain the data and we save the recipe's ID
            recipeInfo = dict()
            recipeInfo["name"] = recipeName.text
            recipeInfo["id"] = c

            # We collect the calories per serving
            tmp1 = recipe.find("div", {"class" : "recipeNutritionSectionBlock"})
            tmp2 = tmp1.find("div", {"class" : "section-body"}).text.split(";")[0].split()[0]
            recipeInfo["calories"] = int(tmp2)

            # We collect the requested data
            recipe_nutr_info = recipe.find_all('div', {"class" : "nutrition-row"})
            for i in recipe_nutr_info :
                res = i.find_all('span', {"class" : "nutrient-name"})

                for j in res :
                    d = j.find('span', {"class" : "elementFont__details--bold elementFont__transformCapitalize"}).text.strip()
                    v = j.find('span', {"class" : "nutrient-value"}).text.strip()

                    # Looking for the values asked for in the 'research' array
                    if d in research :
                        if "mg" in v :
                            recipeInfo[d] = float(re.sub("[^0-9.]", "", v)) / 1000
                        else :
                            recipeInfo[d] = float(re.sub("[^0-9.]", "", v))

            # We save the data
            nutr.append(recipeInfo)

    # We create a CSV file if asked
    if (toCSV) :
        df = pd.DataFrame(nutr)
        df.fillna(0, inplace=True)
        df.to_csv(filepath)

    # We return the collected data
    print("Number of recipes collected :", len(nutr))
    return nutr


# File check and recipe counter
# Function checks if a specific collection of recipes has been collected
# We give the counterStart, the counterRange and the counterEnd
def checkAndCount(cStart, cRange, cEnd) :
    # We create the file counter and the line counter
    c = cStart
    l = 0

    # We iterate on the files
    while (c < cEnd) :
        # We show the state of c
        print(c, "/", cEnd, end="\r")
        # We create the file name to read
        fn = "data/recipedata_" + str(c) + "_" + str(c + cRange - 1) + ".csv"
        # We try to read the data file and read its number of lines
        try :
            f = open(fn)
            r = csv.reader(f)
            l += len(list(r))
        # If it doesn't exist, we print a message with the ID of the missing data file
        except :
            print("Missing or corrupted file :", fn, end="\n")

        # We add the range to the counter
        c += cRange

    # We print the number of lines
    print("Number of recipes collected between ID [", cStart, ",", cEnd,"] =", l)
