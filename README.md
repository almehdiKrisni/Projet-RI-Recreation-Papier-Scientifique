# Projet_RI

### Members
- KRISNI Almehdi
- ARICHANDRA Santhos

Recréation de papiers scientifiques - Exploiting Food Choices Biases for Healthier Recipe Recommendation

Source - https://dl.acm.org/doi/pdf/10.1145/3077136.3080826

### First step
Web crawler creation. Data scraping on the *https://www.allrecipes.com* website.
We only consider recipes with nutritional contents available. <br/>We only collect a recipe if all of the following elements are available on the website :
- recipe description
- recipe picture (if there isn't one, the recipe won't be used for the user interface)
- number of servings
- calories
- number of reviews
- mean user rating
- fat
- saturated fat
- sodium
- sugars
- list of ingredients

Research the FSA (**Food Standards Agency**) and the procedure to obtain the nutritional content per portion.
