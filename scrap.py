# Web scraper in Python
# Tutorial from - https://www.topcoder.com/thrive/articles/web-crawler-in-python

###################################################################### IMPORTS ################################################################################

# Necessary imports
from pyrsistent import v
import requests as rq
import lxml
from bs4 import BeautifulSoup as bs
import string

from torch import nuclear_norm

######################################################### CREATION, ACCESS, PARSING, EXTRACTION ###############################################################

# URL creation and access
url = "https://www.allrecipes.com/recipe/109291/"
headers = {
  'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36 QIHU 360SE'
}
f = rq.get(url, headers=headers)

# Parse webpage
recipe = bs(f.content, 'lxml')

# Researched values
research = ["fat", "saturated fat", "sodium", "sugars"]

# Data extraction
# Dict creation
nutrVal = dict()

# Recipe name
recipe_name = recipe.find('h1', {"class" : "headline heading-content elementFont__display"}).text
nutrVal["recipe name"] = recipe_name

# Nutritional info
recipe_nutr_info = recipe.find_all('div', {"class" : "nutrition-row"})


for i in recipe_nutr_info :
    res = i.find_all('span', {"class" : "nutrient-name"})

    for j in res :
        d = j.find('span', {"class" : "elementFont__details--bold elementFont__transformCapitalize"}).text.strip()
        v = j.find('span', {"class" : "nutrient-value"}).text.strip()

        if d in research :
            nutrVal[d] = v


print(nutrVal)