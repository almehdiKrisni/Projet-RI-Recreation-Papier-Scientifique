# Web scraper in Python
# Tutorial from - https://www.topcoder.com/thrive/articles/web-crawler-in-python

###################################################################### IMPORTS ################################################################################

# Necessary imports
import requests as rq
import lxml
from bs4 import BeautifulSoup as bs

############################################################## CREATION, ACCESS AND PARSING ####################################################################

# URL creation and access
url = "https://www.allrecipes.com"
f = rq.get(url)

# Parse webpage
dt = bs(f.content, 'lxml')

###################################################################### TEST PART ################################################################################
# Information extraction
test = dt.find('table')

# Print test
dt[0]