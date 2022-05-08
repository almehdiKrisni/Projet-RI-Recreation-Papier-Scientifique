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
window.configure(bg='#7C7A7A')
window.geometry("300x200")
window.resizable(False, False)

################################################### GLOBAL PARAMETERS ########################################

# Questionnary values
currentQuestion = 1
answerList = list()

# List of recipe names

# List of recipe pictures
recipeA_picture_list = []
for i in range(10) :
    id = 10000 + i * 10
    recipeA_picture_list.append(ImageTk.PhotoImage(im.open("pictures/recipes_10000_10099/recipe_" + str(id) + ".png")))
recipeB_picture_list = []

##################################################### UTILS FUNCTIONS ########################################

# Function - Close window
def closeWindow() :
    window.quit()

# Function - Clear window
def clearWindow() :
    for i in window.winfo_children() :
        i.destroy()

##################################################### INIT SEQUENCE ##########################################

# Functions
# First window setup
def initWindow() :
    Label(window, text="User Experience - Food Questions", justify=CENTER).place(relx=0.5, rely=0.3, anchor=CENTER)
    start_button.place(relx=0.5, rely=0.7, anchor=CENTER)

# Start questionnary
def startQuestionnary() :
    # We clear the window
    clearWindow()

    # We prepare the first question
    window.geometry("1000x800")
    questionSequence(0)

# Entries and buttons
start_button = Button(window, text="Start", command=startQuestionnary)

################################################## QUESTION SEQUENCE ###########################################

# Functions
# Show the choice situation number questionID
def questionSequence(questionID) :
    # We clear the window
    clearWindow()

    # We create the checkboxes and the IntVar variables
    choiceA = IntVar() # Represents the value 1
    choiceABox = Checkbutton(window, text="Recipe A", variable=choiceA)

    choiceB = IntVar() # Represents the value 2
    choiceBBox = Checkbutton(window, text="Recipe B", variable=choiceB)

    pictureChoice = IntVar() # Represents the value 1
    pictureChoiceBox = Checkbutton(window, text="The pictures", variable=pictureChoice)

    nameChoice = IntVar() # Represents the value 2
    nameChoiceBox = Checkbutton(window, text="The names", variable=nameChoice)

    # Function to check if the necessary boxes have been checked
    def validSelection() :
        return (choiceA.get() + choiceB.get() == 1) and (pictureChoice.get() + nameChoice.get() == 1)

    # Function returning a tuple containing the choices the user made
    def saveChoices() :
        # Creating the variables
        rC = 0
        rI = 0

        # Saving the recipe choice
        if (choiceA.get() > choiceB.get()) :
            rC = 1
        else :
            rC = 2

        # Saving the info choice
        if (pictureChoice.get() > nameChoice.get()) :
            rI = 1
        else :
            rI = 2

        # Return the choice tuple
        return rC, rI

    # Function to go to the next
    def goToNextQuestion() :
        if (validSelection()) :
            # Saving the selection*
            tmpC, tmpI = saveChoices()
            answerList.append([questionID, tmpC, tmpI])
            print(answerList[questionID])

            # Ending the questionnary
            if (questionID + 1 >= 10) :
                endWindow()

            # Moving to the next question
            else :
                endWindow()
                # questionSequence(questionID + 1)

    # Question title
    Label(window, text="Choice " + str(currentQuestion), justify=CENTER).pack(side=TOP)
    Label(window, text="In your opinion, which recipe contains the most fat ?", justify=CENTER).pack(side=TOP)

    # We show the recipes' pictures
    Label(window, image=recipeA_picture_list[questionID]).place(x=100, y=200)
    Label(window, image=recipeA_picture_list[questionID]).place(x=600, y=200)

    # We show the recipes' names
    choiceABox.place(x=100, y=510)
    choiceBBox.place(x=600, y=510)

    # We ask about what helped the user decide and add the next button
    Button(window, text="Next", command=goToNextQuestion).pack(side=BOTTOM)
    nameChoiceBox.pack(side=BOTTOM)
    pictureChoiceBox.pack(side=BOTTOM)
    Label(window, text="Which information helped you choose ?").pack(side=BOTTOM)
    

################################################# QUESTIONNARY END #############################################

# Final window
def endWindow() :
    # We clear the window
    clearWindow()

    # Saving the results in the results file
    resultsfn = "results_" + str(np.random.randint(0, 1000000)) + ".csv"

    # We open the .csv file
    with open(resultsfn, 'w', encoding='utf8') as f :
        # We create a writer
        writer = csv.writer(f)

        # We write the header
        writer.writerow(["question_id", "recipe_choice", "information_choice"])

        # We write the result of the experience
        writer.writerows(answerList)

    # We resize the window
    window.geometry("300x200")

    # Thanks for participating
    Label(window, text="Thanks for participating !\nYou can find the results of the questionnary\nin the " + resultsfn + " file.", justify=CENTER).place(relx=0.5, rely=0.3, anchor=CENTER)
    Button(window, text="Exit", command=closeWindow).place(relx=0.5, rely=0.8, anchor=CENTER)

################################################## START COMMAND ###############################################

# Showing the window
initWindow()
window.mainloop()
