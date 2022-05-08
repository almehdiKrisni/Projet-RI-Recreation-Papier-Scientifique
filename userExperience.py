# File containing all the necessary functions for the user experience

# Imports
from PIL import ImageTk, Image as im, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from utils import *
from tkinter import *

##################################################### WINDOW CREATION ########################################

# declare the window
window = Tk()

# set window title
window.title("User Experience")
window.configure(bg='lightgray')
window.geometry("300x200")

################################################### GLOBAL PARAMETERS ########################################

# Questionnary values
currentQuestion = 0
answerList = dict()

# Choices index


##################################################### UTILS FUNCTIONS ########################################

# Function - Close window
def closeWindow() :
    window.quit()

# Function - Clear window
def clearWindow() :
    for i in window.winfo_children() :
        i.destroy()

##################################################### INIT SEQUENCE ##########################################

# User Info
usernameValue = "None"

# Functions
# First window setup
def initWindow() :
    usernameEntry.pack(side=TOP, pady=50)
    submit_button.pack(side=TOP)
    start_button.pack(side=BOTTOM)

# Submit username
def submitUsername() :
    # We verify if the user has given his name
    name = usernameEntry.get()
    if (name != "") :
        usernameValue = name
        print(usernameValue)

# Start questionnary
def startQuestionnary() :
    # We clear the window
    clearWindow()

    # We prepare the first question
    questionSequence(0)

# Entries and buttons
usernameEntry = Entry(window)
submit_button = Button(window, text="Submit username", command=submitUsername)
start_button = Button(window, text="Start", command=startQuestionnary)

################################################## QUESTION SEQUENCE ###########################################

# Adding pictures to the window
recipePicture_A = ImageTk.PhotoImage(im.open("pictures/recipes_10000_10099/recipe_10000.png"))

# Functions
# Show the choice situation number questionID
def questionSequence(questionID) :
    # We get the recipes IDs
    Label(window, image=recipePicture_A).pack(side=BOTTOM)



################################################# QUESTIONNARY END #############################################
# User Info

################################################## START COMMAND ###############################################

# Showing the window
initWindow()
window.mainloop()
