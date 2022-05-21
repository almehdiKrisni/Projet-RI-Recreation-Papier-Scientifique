# Projet_RI

## **Members**
- KRISNI Almehdi
- ARICHANDRA Santhos

Recréation de papiers scientifiques - Exploiting Food Choices Biases for Healthier Recipe Recommendation

Source - https://dl.acm.org/doi/pdf/10.1145/3077136.3080826

## **First step**
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

## **User Experience Models - How to use**

First, you have to use the following command :
> pip install -r requirements.txt

It will allow you to install all the libraries used in the project. The file was generated thanks to the **pipreqs** library.
A few libraries are not included in the file, so you will to download them when executing the file.

### **Start**

Use the command :
> python userExperience.py

It will open an interface in the corresponding Terminal. You will be asked to input a name, a study choice and a model.

**Study choice** :
- 0 - You will have to choose between the recipes only using the names or the pictures
- 1 - You will have to choose between the recipes only using the names, the pictures or the ingredients

**Models** :
<br>There are 20 different models for each study. You only need to input the model ID in the Terminal. A list with all of the IDs will appear when asked to choose one.

**Selection process** :
<br>
