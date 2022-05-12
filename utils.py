#################################### Utilitaries #######################################

###################################### Imports #########################################

# Needed for data collection
from tracemalloc import stop
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
import string
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
stopwords = stopwords.words('english')

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

# Get the list of ingredients (need a recipe dataframe and an ID)
def getIngredientsList(df, id) :
    # We retrieve the corresponding recipe
    recipe = df.loc[df["id"] == id]

    # We get the ingredients
    ing = recipe["ingredients"]

    # We transfrom the values and return the split
    ing = ing.values[0].replace('\u2009', ' and ')
    return ing.split('#')


# Compute the cosine similarity for two recipes without using the recipes names
def cosSimCalc(df, idA, idB, mode=1, coefVal=0.5, coefName=0.5, coefIng=1/3) :
    # We need to use a transformed dataframe (the one without any 'object' values)
    tdf = df.select_dtypes(exclude=['object'])
    recipeA = tdf.loc[tdf["id"] == idA].drop(columns=['id']).values[0]
    recipeB = tdf.loc[tdf["id"] == idB].drop(columns=['id']).values[0]

    # Mode 1 - We only consider the values
    if (mode == 1) :
        # We then compute the cosine similarity
        return np.dot(recipeA,recipeB)/(norm(recipeA)*norm(recipeB))

    # Mode 2 - We consider the values and the recipe names
    elif (mode == 2) :
        # We check if the sum of the coef values is equal to 1
        # If not, we return an error
        if (sum([coefVal, coefName]) != 1.) :
            print("The sum of the coef values should be equal to 1.")
            return

        # We have to save the names of the recipes in order their cosine similarity
        nameA = df.loc[df["id"] == idA]["name"].values[0]
        nameB = df.loc[df["id"] == idB]["name"].values[0]

        # We then compute the cosine similarity
        return np.dot(recipeA,recipeB)/(norm(recipeA)*norm(recipeB)) * coefVal + stringCosSim(nameA, nameB) * coefName

    # Mode 3 - We consider the values, the ingredients and the recipe names
    elif (mode == 3) :
        # We check if the sum of the coef values is equal to 1
        # If not, we return an error
        if (sum([coefVal, coefName, coefIng]) != 1.) :
            print("The sum of the coef values should be equal to 1.")
            return 

        # We have to save the names of the recipes in order to compute their cosine similarity
        nameA = df.loc[df["id"] == idA]["name"].values[0]
        nameB = df.loc[df["id"] == idB]["name"].values[0]

        # We save the ingredients of the recipes in order to compute their cosine similarity
        ingsA = df.loc[df["id"] == idA]["ingredients"].values[0]
        ingsB = df.loc[df["id"] == idB]["ingredients"].values[0]

        # We then compute the cosine similarity
        return np.dot(recipeA,recipeB)/(norm(recipeA)*norm(recipeB)) * coefVal + stringCosSim(nameA, nameB) * coefName + stringCosSim(ingsA, ingsB) * coefIng

 
# Compute the cosine similarity between a specific recipe and the entier dataframe
def allCosSim(df, id, mode=1, coefVal=0.5, coefName=0.5, coefIng=1/3) :
    # We iterate on all of the recipes except the studied recipe itself
    return [(c, cosSimCalc(df, id, c, mode=mode, coefVal=coefVal, coefName=coefName, coefIng=coefIng)) for c in df["id"] if c != id]

# Compute the list of recipe ids with a cosine similarity to the model repice greater than a specific value
def findSimRecipes(df, id, val, mode=1, coefVal=0.5, coefName=0.5, coefIng=1/3) :
    # We return the list of ids
    return [c for c in df["id"] if (c != id) and (cosSimCalc(df, id, c, mode=mode, coefVal=coefVal, coefName=coefName) >= val)]

# Get the statistics of cosine similarity for a specific recipe
# We use the following reference values : [0.8, 0.6, 0.4, 0.2]
def recipeSimStats(df, id, mode=1, coefVal=0.5, coefName=0.5, coefIng=1/3) :
    # We compute the cosine similiraties
    cossim = allCosSim(df, id, mode=mode, coefVal=coefVal, coefName=coefName, coefIng=coefIng)

    # Probability of finding a recipe with more than 0.2 cosine similarity in the database
    odds_2 = sum([1 for v in cossim if v[1] >= 0.2]) / len(cossim)

    # Probability of finding a recipe with more than 0.4 cosine similarity in the database
    odds_4 = sum([1 for v in cossim if v[1] >= 0.4]) / len(cossim)

    # Probability of finding a recipe with more than 0.6 cosine similarity in the database
    odds_6 = sum([1 for v in cossim if v[1] >= 0.6]) / len(cossim)

    # Probability of finding a recipe with more than 0.8 cosine similarity in the database
    odds_8 = sum([1 for v in cossim if v[1] >= 0.8]) / len(cossim)

    # We return the probabilities
    return odds_2, odds_4, odds_6, odds_8


# Function returning the path to a recipe picture based on the id of the recipe
def recipePicPath(id) :
    dirid = int(id / 100)
    dir_name = "pictures/recipes_" + str(dirid * 100) + "_" + str(dirid * 100 + 99) + "/"
    return dir_name + "recipe_" + str(id) + ".png"

# (https://towardsdatascience.com/calculating-string-similarity-in-python-276e18a7d33a)
# Functions returning a clean string (punctuation removed, lowercase string, removed stopwords)
def clean_string(text) :
    text = ''.join([word for word in text if word not in string.punctuation])
    text = text.lower()
    text = ' '.join([word for word in text.split() if word not in stopwords])
    return text

# Function returning the cosine similarity between two strings
def cosine_sim_vectors(vec1, vec2) :
    vec1 = vec1.reshape(1, -1)
    vec2 = vec2.reshape(1, -1)
    return cosine_similarity(vec1, vec2)[0][0]

def stringCosSim(string1, string2) :
    # We clean the strings
    cleaned = list(map(clean_string, [string1, string2]))

    # We vectorize the strings
    vectorizer = CountVectorizer()
    vectors = vectorizer.fit_transform(cleaned).toarray()

    # We return the cosine similarity
    return cosine_sim_vectors(vectors[0], vectors[1])


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

            # It's possible that the data file contains 
            if not tmp.empty :
                tmp = tmp.loc[:, ~tmp.columns.str.contains('^Unnamed')]
                tmp = tmp[colMod]
                valueslist = valueslist + tmp.values.tolist()
        
        # If we encounter an error, we skip the file
        except Exception as e:
            print(c, str(e))

        # We update the counter
        c += cRange

    # We return the global dataframe after a few modifications (first column, column name)
    df = pd.DataFrame(data=valueslist, columns=colMod)
    df = df.dropna()
    return df

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
# it's id, the recipe A id, the recipe A name, the recipe A picture path, the recipe B id, the recipe B name, 
# the recipe B picture path, the correct answer (1 for A, 2 for B)
def generate_user_experience(df, nbQuestions, expId, simval, mode=1, coefVal=0.5, coefName=0.5, coefIng=1/3) :
    # File header
    header = ["question_id", "recipe_A_id", "recipe_A_name", "recipe_A_picture", "recipe_B_id", "recipe_B_name", "recipe_B_picture", "correct_answer"]

    # We use a list to save all the used ids in the created user experience
    usedIds = []

    # We have to make sure all the recipes in the dataframe have a picture
    df = removeNoPictures(df)

    # We create the file
    filename = "userExperiencesModels/model_" + expId + ".csv"
    with open(filename, 'w', encoding='utf8') as f :
        # We create a writer
        writer = csv.writer(f)

        # We write the header
        writer.writerow(header)

        # We iterate on the number of questions
        for i in range(nbQuestions) :
            # Status print
            print("Generating question " + str(i + 1) + " for user experience model n." + expId + " ...", end="\r")

            # Correct run variable and recipe ids variables
            runAgain = True
            recipeAindex = 0
            recipeBindex = 0

            while (runAgain) :
                # We take a random id from the dataframe 
                recipeAindex = random.choice(df["id"].tolist())

                # We now generate the list of similar recipes and pick a random id
                simRecipesId = findSimRecipes(df, recipeAindex, simval, mode=mode, coefVal=coefVal, coefName=coefName, coefIng=coefIng)

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

            # We find the names of the recipes
            recipeAname = df.loc[df["id"] == recipeAindex]["name"].values[0]
            recipeBname = df.loc[df["id"] == recipeBindex]["name"].values[0]

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
            writer.writerow([i + 1, recipeAindex, recipeAname, picPathA, recipeBindex, recipeBname, picPathB, expectedAns])

    # Message to signal the user experience model has been creater
    print("The experience user model id." + expId + " has been created.")


########################################################### FSA SCORE CALCULATOR ##############################################

# Function returning the quantiles (1st and 3rd)
def getQuantiles(dfS) :
    # We need to pass as a parameter a pandas Series (a column for example)
    [q1, q2] = dfS.quantile([0.25, 0.75]).values
    return q1, q2

# Function returning the score for a single recipe
def getFSArecipe(rec, q1fat, q3fat, q1satfat, q3satfat, q1sod, q3sod, q1sug, q3sug) :
    # Variable to save the score
    score = 0

    # Fat score
    if (rec["fat"] <= q1fat) :
        score += 1
    else :
        if (rec["fat"] <= q3fat) :
            score += 2
        else :
            score += 3
    
    # Saturated fat score
    if (rec["saturated fat"] <= q1satfat) :
        score += 1
    else :
        if (rec["saturated fat"] <= q3satfat) :
            score += 2
        else :
            score += 3

    # Sodium score
    if (rec["sodium"] <= q1sod) :
        score += 1
    else :
        if (rec["sodium"] <= q3sod) :
            score += 2
        else :
            score += 3

    # Sugars score
    if (rec["sugars"] <= q1sug) :
        score += 1
    else :
        if (rec["sugars"] <= q3sug) :
            score += 2
        else :
            score += 3

    # We return the score
    return score

# Function adding a FSA Health score value to every example in the dataframe
def computeFSAscore(df) :
    # We need to pass as a parameter an unchanged recipe database (like the one created with pdCreator)
    # The FSA score is calculated with the 'fat', 'saturated fat', 'sodium' and 'sugars'

    # We create the empty FSA score column
    df["FSA_score"] = np.nan

    # We get the quantiles for the score calculation
    q1fat, q3fat = getQuantiles(df["fat"])
    q1satfat, q3satfat = getQuantiles(df["saturated fat"])
    q1sod, q3sod = getQuantiles(df["sodium"])
    q1sug, q3sug = getQuantiles(df["sugars"])

    # We create an empty list to save the results
    scoreList = []

    # We iterate on the ids and calculate the scores
    for i in range(len(df)) :
        # We pass the recipe to the score calculator
        scoreList.append(getFSArecipe(df.iloc[i], q1fat, q3fat, q1satfat, q3satfat, q1sod, q3sod, q1sug, q3sug))

    # We save the new FSA score column with the list
    df["FSA_score"] = scoreList

#################################################### VALUES MANAGEMENT ###########################################

# Function adding the number of ingredients to the recipe dataframe
def computeNumberIngredients(df) :
    # We need to pass as parameter an unchanged recipe database
    # We split using the '#' value in order to get the number of ingredients

    # We create the empty number of ingredients column
    df["number_ingredients"] = np.nan

    # We save the number of ingredients
    df["number_ingredients"] = [len(df.iloc[i]["ingredients"].split('#')) for i in range(len(df))]

############################################### VIZUALISATION FUNCTIONS ##########################################

# Function showing a bar plot (histogram) of the values of an array
# The values should be continued for better usage
def showBar(values, odds=False) :
    labels, counts = np.unique(values, return_counts=True)
    if (odds) :
        counts = counts / len(values)
    plt.bar(labels, counts, align='center')
    plt.gca().set_xticks(labels)
    plt.show()