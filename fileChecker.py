# This file checks the data collected and the number of usable items

########################################### IMPORTS ##############################################

# Necessary imports
from sys import argv
from utils import *

####################################### CHECK AND COUNT ##########################################

# Main function
def main(argv) :
    # We check if the number of parameters is correct
    if (len(argv) != 4) :
        print("You need to pass 3 parameters to the script.\n$file counterStart counterRange counterEnd")
        input('Press ENTER to exit')
    else :
        # We create the variables
        cS = int(argv[1])
        cR = int(argv[2])
        cE = int(argv[3])

        # We check if the counters are correct
        if (cS > cE) :
            print("counterEnd cannot be inferior to counterStart")
            input('Press ENTER to exit')
        if (cR < 1) :
            print("counterRange cannot be inferior to 1")
            input('Press ENTER to exit')
            
        # We check the files
        checkAndCount(cS, cR, cE)
        input('Press ENTER to exit')

# Exec
if __name__ == "__main__" :
    main(argv)