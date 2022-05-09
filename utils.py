#################################### Utilitaries #######################################

###################################### Imports #########################################

# Needed for data collection
from pyrsistent import v
import requests as rq
import lxml
from bs4 import BeautifulSoup as bs

# Needed for picture collection
import os
import urllib
from PIL import Image as im

# Needed for cosine similarity
from scipy import spatial

# Other usage
import string
import pandas as pd
import pprint
import re
import time
import csv
import seaborn as sns
import numpy as np
import copy
import matplotlib.pyplot as plt

################################## Scraping functions ###################################

# Research parameters
classicResearch = ["fat", "saturated fat", "sodium", "sugars", "carbohydrates", "cholesterol", "calcium", "iron", "magnesium", "potassium", "dietary fiber"]

# Function to scrape website data from "allrecipes.com"with a number of pages to visit and the researched values
# Returns a dictionnary or a pandas database (request with parameter) of the scraping
# The counter starts from the adress
def recipeCollector(startCounter, counter, research, toCSV=False, filepath="data/default_file.csv") :
    # We create the dictionnary
    nutr = []

    # Note - It is possible some addresses won't return any recipe
    # The basic data scraped is 

    # Header creation (to avoid any problem when accessing the website)
    headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36 QIHU 360SE'
    }

    # Data extraction
    for id in range(startCounter, startCounter + counter) :
        # Status
        print(id, end="\r")

        # We try to collect the recipe
        # There are some cases where there won't be a picture or a description, so we take out those
        try :
            # We collect the recipe data
            data = recipeGrab(id, research)

            # We add the new recipe data to the dictionnary
            nutr.append(data)
        except Exception :
            pass

    # We create a CSV file if asked
    if (toCSV) :
        df = pd.DataFrame(nutr)
        df.fillna(0, inplace=True)
        df.to_csv(filepath)

    # We return the collected data
    print("Number of recipes collected :", len(nutr))
    return nutr

# Recipe collector
def recipeGrab(recipeID, research) :

    # We get the website data then parse it
    adr = "https://www.allrecipes.com/recipe/" + str(recipeID)
    f = rq.get(adr)
    recipe = bs(f.content, 'lxml')
    data = dict()


    # Recipe name
    recipe_name = recipe.find('h1', {"class" : "headline heading-content elementFont__display"}).text
    data["name"] = recipe_name
    data["id"] = recipeID


    # Recipe description
    tmp = recipe.find('div', {"class" : "recipe-summary elementFont__dek--within"})
    recipe_desc = tmp.find('p').text
    data["description"] = recipe_desc


    # Recipe picture
    data["picture"] = 0
    try :
        tmp1 = recipe.find('div', {"class" : "image-container"})
        tmp2 = tmp1.find('div')
        tmp3 = str(tmp2.find('noscript'))
        recipe_picture_url = tmp3.split("src")[1].split('"')[1]
        data["picture"] = recipe_picture_url
    except Exception :
        # Maybe the first picture is a video, we have to check the secondary pictures
        try :
            tmp1 = recipe.find('div', {"class" : "recipe-review-image-wrapper"})
            tmp2 = tmp1.find('div')
            recipe_picture_url = str(tmp2).split('data-src="')[1].split('"')[0]
            data["picture"] = recipe_picture_url
        except Exception :
            # Nothing was found
            print("No picture was found for recipe", recipeID)

    # Recipe servings
    tmp = str(recipe.find('div', {"class" : "nutrition-top light-underline elementFont__subtitle"}))
    recipe_servings = tmp.split("Servings Per Recipe: ")[1].split("<")[0]
    data["servings"] = recipe_servings

    # Recipe calories
    tmp = str(recipe.find('div', {"class" : "nutrition-top light-underline elementFont__subtitle"}))
    recipe_calories = tmp.split("Calories:</span> ")[1].split("<")[0]
    data["calories"] = recipe_calories


    # Recipe number of reviews
    data["reviews"] = recipe.find('span', {"class" : "feedback__total"}).text


    # Recipe mean user rating
    tmp1 = recipe.find_all('span', {"review-star-text visually-hidden"})
    rating = 0.
    nbRatings = 0.
    # We iterate on the reviews
    for r in tmp1 :
        rating += float(r.text.split()[1])
        nbRatings += 1
    # If there's at least one review, we save the mean rating
    if (nbRatings != 0) :
        data["rating"] = round(rating / nbRatings, 1)


    # Nutrition info
    recipe_nutr_info = recipe.find_all('div', {"class" : "nutrition-row"})
    for i in recipe_nutr_info :
        res = i.find_all('span', {"class" : "nutrient-name"})
        for j in res :
            d = j.find('span', {"class" : "elementFont__details--bold elementFont__transformCapitalize"}).text.strip()
            v = j.find('span', {"class" : "nutrient-value"}).text.strip()

            # Looking for the values asked for in the 'research' array
            if d in research :
                if "mg" in v :
                    data[d] = float(re.sub("[^0-9.]", "", v)) / 1000
                else :
                    data[d] = float(re.sub("[^0-9.]", "", v))


    # We collect all the recipe's ingredients
    tmp = recipe.find_all('span', {"class" : "ingredients-item-name elementFont__body"})
    recipe_ingredients = []
    for i in tmp :
        recipe_ingredients.append(i.text)
    data["ingredients"] = "#".join(recipe_ingredients)


    # We return the result of the research
    return data

# Function downloading all the pictures in a specific range (all ranges are equal to a 100)
def download_pictures(startIndex, rangeIndex) :
    # We only pass as parameter the starting index (10000 + (100 * index))

    # We check if the directory already exists
    dir_name = "pictures/recipes_" + str(startIndex) + "_" + str(startIndex + rangeIndex - 1) + "/"
    if (os.path.isdir(dir_name)) :
        pass
    else :
        try :
            os.mkdir(dir_name)
        except Exception as e:
            print("Directory creation error -", dir_name, "could not be created.")
            print(str(e))

    # We create the panda database
    df = pdCreator(startIndex, rangeIndex, startIndex + rangeIndex)

    # We iterate on the "picture" and "id" values
    for p in range(rangeIndex) :
        # There are cases where we don't have rangeIndex in the dataframe
        if (p < len(df)) :
            rID = df["id"][p]
            adr = str(df["picture"][p])

            print("Downloading picture for recipe", rID, end="\r")

            # If the recipe has a picture, we download it
            if (len(adr) > 10) :
                filename = dir_name + "recipe_" + rID + ".png"
                urllib.request.urlretrieve(adr, filename)

    print("Finished downloading the pictures (" + str(startIndex) + "_" + str(startIndex + rangeIndex - 1) + str(")"))

# Function resizing all the pictures into a specific shape
def reshape_pictures(startIndex, rangeIndex, shape) :

    # We first check if the directory exists
    dir_name = "pictures/recipes_" + str(startIndex) + "_" + str(startIndex + rangeIndex - 1) + "/"
    if (os.path.isdir(dir_name)) :
        # We create the panda database
        df = pdCreator(startIndex, rangeIndex, startIndex + rangeIndex)

        # We iterate on the "picture" and "id" values
        for p in range(rangeIndex) :
            # There are cases where we don't have rangeIndex in the dataframe
            if (p < len(df)) :
                rID = df["id"][p]
                adr = str(df["picture"][p])

                print("Resizing picture for recipe", rID, end="\r")

                # If the recipe has a picture, we download it
                if (len(adr) > 10) :
                    filename = dir_name + "recipe_" + rID + ".png"
                    image = im.open(filename)
                    image = image.resize(shape, im.ANTIALIAS)
                    image.save(fp=filename)

    else :
        print("Directory creation error -", dir_name, "could not be found.")
    
    print("Finished resizing the pictures (" + str(startIndex) + "_" + str(startIndex + rangeIndex - 1) + str(")"))
        

################################################# RECIPES UTILS FUNCTIONS ############################################

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


############################################# DATAFRAME MANAGEMENT ####################################################


# Function used to read all the data collected and create a global pandas DataFrame
def pdCreator(cStart, cRange, cEnd) :
    # We get the right columns order. The right order is the one of the first .csv file (recipedata_10000_10099.csv)
    tmp = pd.read_csv("data/recipedata_10000_10099.csv")
    tmp = tmp.loc[:, ~tmp.columns.str.contains('^Unnamed')]
    colMod = tmp.columns.tolist()

    v = 0

    # We iterate on every data file to create a general pandas dataframe
    c = cStart
    # We create a list because it is faster to create a df from a list than to append to an
    # existing df
    # ( https://stackoverflow.com/questions/13784192/creating-an-empty-pandas-dataframe-then-filling-it )
    valueslist = []

    while (c < cEnd) :
        # We get the filename
        fn = "data/recipedata_" + str(c) + "_" + str(c + cRange - 1) + ".csv"
        # We try get the file and transform it into a list
        try :
            # We create a dataframe containing all the data
            tmp = pd.read_csv(fn)
            tmp = tmp.loc[:, ~tmp.columns.str.contains('^Unnamed')]
            tmp = tmp[colMod]
            valueslist = valueslist + tmp.values.tolist()
            v += (len(tmp.values.tolist()))
        
        # If we encounter an error, we skip the file
        except Exception as e:
            print(str(e))

        # We update the counter
        c += cRange

    # We return the global dataframe after a few modifications (first column, column name)
    df = pd.DataFrame(data=valueslist, columns=colMod)
    df = df.dropna()
    return df

# Function removing all the exemples without a picture
def removeNoPictures(df) :
    # We return a new df
    newdf = copy.deepcopy(df)

    # We remove all the exemples without a picture
    # Condition - Length of address
    newdf.drop(newdf[newdf["picture"] == "0"].index, inplace=True)
    newdf.drop(newdf[newdf["picture"] == ""].index, inplace=True)
    return newdf