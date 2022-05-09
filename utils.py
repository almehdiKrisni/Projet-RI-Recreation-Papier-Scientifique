#################################### Utilitaries #######################################

###################################### Imports #########################################

# Needed for data collection
from turtle import color
from matplotlib import colors
from matplotlib.style import use
from pyrsistent import v
import requests as rq
import lxml
from bs4 import BeautifulSoup as bs

# Needed for picture collection
import os
import urllib
from PIL import Image as im, ImageStat
import argparse
import imutils
import cv2

# Needed for cosine similarity
from numpy.linalg import norm

# Other usage
import pandas as pd
import re
import time
import csv
import seaborn as sns
import numpy as np
import copy
import matplotlib.pyplot as plt
import random

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
            rID = str(df["id"][p])
            adr = str(df["picture"][p])

            print("Downloading picture for recipe", rID, end="\r")

            # If the recipe has a picture, we download it
            if (len(adr) > 10) :
                filename = dir_name + "recipe_" + rID + ".png"
                urllib.request.urlretrieve(adr, filename)

    print("Finished downloading the pictures (" + str(startIndex) + "_" + str(startIndex + rangeIndex - 1) + str(")"))

#################################################### PICTURE MANAGEMENT ###########################################################

# Function computing the brightness, sharpness, entropy, colorfulness and contrast of a picture
def picStats(id) :
    # We get the picture
    filename = recipePicPath(id)

    # We try to open the picture file
    try :
        # We convert the picture to grayscale
        pic = im.open(filename).convert('L')

        # We compute all the values
        # Brightness
        brightstat = ImageStat.Stat(pic).mean[0]
        print("Brigtness :\t", brightstat)

        # Sharpness
        array = np.asarray(pic, dtype=np.int32)
        gy, gx = np.gradient(array)
        gnorm = np.sqrt(gx**2 + gy**2)
        sharpstat = np.average(gnorm)
        print("Sharpness :\t", sharpstat)

        # Entropy
        # print(image_entropy(filename))

        # Colorfulness
        colorstat = image_colorfulness(filename)
        print("Colorfulness :\t", colorstat)

        # Contrast
        contrstat = image_contrast(filename)
        print("Contrast :\t", contrstat)

        # We return the values
        return brightstat, sharpstat, colorstat, contrstat

    # In case there is a problem with the picture
    except Exception as e :
        print("Could not find the picture with id", id, ".")
        print(str(e))


# Function returning a picture colorfulness with a filepath passed as a parameter
# (https://pyimagesearch.com/2017/06/05/computing-image-colorfulness-with-opencv-and-python/)
def image_colorfulness(filepath):
    # We get the image
    image = cv2.imread(filepath)

	# split the image into its respective RGB components
    (B, G, R) = cv2.split(image.astype("float"))

    # compute rg = R - G
    rg = np.absolute(R - G)

	# compute yb = 0.5 * (R + G) - B
    yb = np.absolute(0.5 * (R + G) - B)

	# compute the mean and standard deviation of both `rg` and `yb`
    (rbMean, rbStd) = (np.mean(rg), np.std(rg))
    (ybMean, ybStd) = (np.mean(yb), np.std(yb))

	# combine the mean and standard deviations
    stdRoot = np.sqrt((rbStd ** 2) + (ybStd ** 2))
    meanRoot = np.sqrt((rbMean ** 2) + (ybMean ** 2))

	# derive the "colorfulness" metric and return it
    return stdRoot + (0.3 * meanRoot)

# Function returning the contrast of a picture with a filepath passed as a parameter (Michelson contrast)
# (https://stackoverflow.com/questions/58821130/how-to-calculate-the-contrast-of-an-image)
def image_contrast(filepath) :
    # We get the image
    img = cv2.imread(filepath)
    Y = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)[:,:,0]

    # compute min and max of Y
    min = np.min(Y)
    max = np.max(Y)

    # compute contrast
    contrast = (max-min)/(max+min)
    return contrast

# Function returning the shannon entropy of a picture with a filepath passed as a parameter
# (https://stackoverflow.com/questions/50313114/what-is-the-entropy-of-an-image-and-how-is-it-calculated)
# def image_entropy(filepath) :
#     img = cv2.imread(filepath)
#     return shannon_entropy(img[:,:,0])

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
                rID = str(df["id"][p])
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


# Get the list of ingredients (need a recipe dataframe and an ID)
def getIngredientsList(df, id) :
    # We retrieve the corresponding recipe
    recipe = df.loc[df["id"] == id]

    # We get the ingredients
    ing = recipe["ingredients"]

    # We transfrom the values and return the split
    ing = ing.values[0].replace('\u2009', ' and ')
    return ing.split('#')


# Compute the cosine similarity for two recipes
def cosineSimCalc(tdf, idA, idB) :
    # We need to pass a transformed dataframe (the one without any 'object' values)
    recipeA = tdf.loc[tdf["id"] == idA].drop(columns=['id']).values[0]
    recipeB = tdf.loc[tdf["id"] == idB].drop(columns=['id']).values[0]

    # We then compute the cosine similarity
    return np.dot(recipeA,recipeB)/(norm(recipeA)*norm(recipeB))


# Compute the cosine similarity between a specific recipe and the entier dataframe
def allCosSim(tdf, id) :
    # We iterate on all of the recipes except the studied recipe itself
    return [(c, cosineSimCalc(tdf, id, c)) for c in tdf["id"] if c != id]

# Compute the list of recipe ids with a cosine similarity to the model repice greater than a specific value
def findSimRecipes(tdf, id, val) :
    # We return the list of ids
    return [c for c in tdf["id"] if (c != id) and (cosineSimCalc(tdf, id, c) >= val)]


# Get the statistics of cosine similarity for a specific recipe
# We use the following reference values : [0.8, 0.6, 0.4, 0.2]
def recipeSimStats(cossim) :
    # Probability of finding a recipe with more than 0.2 cosine similarity in the database
    odds_2 = sum([1 for v in cossim if v[1] >= 0.2]) / len(cossim)

    # Probability of finding a recipe with more than 0.4 cosine similarity in the database
    odds_4 = sum([1 for v in cossim if v[1] >= 0.4]) / len(cossim)

    # Probability of finding a recipe with more than 0.6 cosine similarity in the database
    odds_6 = sum([1 for v in cossim if v[1] >= 0.6]) / len(cossim)

    # Probability of finding a recipe with more than 0.8 cosine similarity in the database
    odds_8 = sum([1 for v in cossim if v[1] >= 0.8]) / len(cossim)

    return odds_2, odds_4, odds_6, odds_8


# Function returning the path to a recipe picture based on the id of the recipe
def recipePicPath(id) :
    dirid = int(id / 100)
    dir_name = "pictures/recipes_" + str(dirid * 100) + "_" + str(dirid * 100 + 99) + "/"
    return dir_name + "recipe_" + str(id) + ".png"

############################################# DATAFRAME MANAGEMENT ####################################################

# Function used to read all the data collected and create a global pandas DataFrame
def pdCreator(cStart, cRange, cEnd) :
    # We get the right columns order. The right order is the one of the first .csv file (recipedata_10000_10099.csv)
    tmp = pd.read_csv("data/recipedata_10000_10099.csv")
    tmp = tmp.loc[:, ~tmp.columns.str.contains('^Unnamed')]
    colMod = tmp.columns.tolist()

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

############################################################ USER EXPERIENCE ##################################################

# Function generating a user experience based on a dataframe, a number of choices, an experience id and a similarity value
# User experiences are represented by a .csv file containing for each choice :
# it's id, the recipe A id, the recipe B id, the recipe A picture path, the recipe B picture path, the correct answer (1 for A, 2 for B)
def generate_user_experience(df, nbQuestions, expId, simval) :
    # File header
    header = ["question_id", "recipe_A_id", "recipe_B_id", "recipe_A_picture", "recipe_B_picture", "correct_answer"]

    # We use a list to save all the used ids in the created user experience
    usedIds = []

    # We create the file
    filename = "userExperiencesModels/model_" + str(expId) + ".csv"
    with open(filename, 'w', encoding='utf8') as f :
        # We create a writer
        writer = csv.writer(f)

        # We write the header
        writer.writerow(header)

        # We iterate on the number of questions
        for i in range(nbQuestions) :
            # Status print
            print("Generating question " + str(i + 1) + " for user experience model n." + str(expId) + " ...", end="\r")

            # Correct run variable and recipe ids variables
            runAgain = True
            recipeAindex = 0
            recipeBindex = 0

            while (runAgain) :
                # We take a random id from the dataframe 
                recipeAindex = random.choice(df["id"].tolist())

                # We now generate the list of similar recipes and pick a random id
                simRecipesId = findSimRecipes(df, recipeAindex, simval)

                # We check if there are similar recipes
                if (len(simRecipesId) > 0) :
                    # We choose a random recipe B from the list
                    recipeBindex = random.choice(simRecipesId)
                    runAgain = False

                    # We check if any of the collected ids are in the list in case of a correct run
                    if ((recipeAindex in usedIds) or (recipeBindex in usedIds)) :
                        runAgain = True

            # We save the collected ids in the appropriate list
            usedIds.append(recipeAindex)
            usedIds.append(recipeBindex)

            # We find the expected answer
            expectedAns = 0
            fatAvalue = df.loc[df["id"] == recipeAindex]["fat"].values[0]
            fatBvalue = df.loc[df["id"] == recipeBindex]["fat"].values[0]

            if (fatAvalue >= fatBvalue) :
                expectedAns = 1
            else :
                expectedAns = 2

            # We save the recipes pictures paths
            picPathA = recipePicPath(recipeAindex)
            picPathB = recipePicPath(recipeBindex)

            # We save the data in the model file
            writer.writerow([i + 1, recipeAindex, recipeBindex, picPathA, picPathB, expectedAns])

    # Message to signal the user experience model has been creater
    print("The experience user model id." + str(expId) + " has been created.")


########################################################### FSA SCORE CALCULATOR ##############################################