# Web scraper in Python
# Tutorial from - https://www.topcoder.com/thrive/articles/web-crawler-in-python

########################################### IMPORTS ##############################################

# Necessary imports
from utils import *

########################################### TESTS ################################################

request = ["fat", "saturated fat", "sodium", "sugars"]
pprint.pprint(recipeCollector(10000, 10, request))
