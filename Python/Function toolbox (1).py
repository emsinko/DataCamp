
## Data science toolbox - part1

## Funkcie

## Sytnax pre tvorbu funkcie:

def nazov_funkcie(input_var):
    ....
    ....
    ....
    
# Dôležité: neexistuje {} na začiatok a ukončenie tela funkcie. Vyhodnocuje sa to podla zarovnania kodu (pod menom funkcie)


# Príklad na definíciu druhej mocniny len s printom:
def square(x):
    print(x**2)

square(11)    

temp = square(11)
temp - 121          # nevie čo stým, nakoľko v temp je ulozeny print(121) a nie realna hodnota... Opravime to return-om:


# Príklad na definíciu druhej mocniny s returnom:
def square(x): 
    """ Return the square of value """      # toto je tzv. docstring - akási dokumentácia funkcie
    return x**2

temp = square(11)
temp - 121 

################################################

### INTRO DO TUPLES - variable type:

even_nums = (2,4,6)
print(type(even_nums))     # type = tuple

a,b,c = even_nums          # teraz sa ulozia do premennych a <- 2 , b <- 4 a c <-6
print(a)

b = even_nums[2]           # takto sa dostaneme tiež napríklad k tretej hodnote 



### MULTIPLE OUTPUT

def raise_both(var1, var2):
    """ Raise var1 to the power of var2 and vice versa""" 
    new_tuple = (var1 ** var2, var2 ** var1)
    return new_tuple

mocniny = raise_both(2,3)    
print(mocniny)

a,b = raise_both(2,3)
print(a)
print(b)

#############################################

# Import pandas
import pandas as pd

# Import Twitter data as DataFrame: df
df = pd.read_csv("tweets.csv")

# Initialize an empty dictionary: langs_count
langs_count = {}

# Extract column from DataFrame: col
col = df['lang']

# Iterate over lang column in DataFrame
for entry in col:
    
    # If the language is in langs_count, add 1
    if entry in langs_count.keys():
        langs_count[entry] = langs_count[entry] + 1 
    # Else add the language to langs_count, set the value to 1
    else:
        langs_count[entry] = 1

# Print the populated dictionary
print(langs_count)

###############################################

# Define count_entries()
def count_entries(df, col_name):
    """Return a dictionary with counts of 
    occurrences as value for each key."""

    # Initialize an empty dictionary: langs_count
    langs_count = {}
    
    # Extract column from DataFrame: col
    col = df[col_name]
    
    # Iterate over lang column in DataFrame
    for entry in col:

        # If the language is in langs_count, add 1
        if entry in langs_count.keys():
            langs_count[entry] = langs_count[entry] + 1 
        # Else add the language to langs_count, set the value to 1
        else:
            langs_count[entry] = 1 

    # Return the langs_count dictionary
    return langs_count

# Call count_entries(): result
result = count_entries(tweets_df, "lang")

# Print the result
print(result)

###############################################

# Ak chceme definovať globálnu premennú vo vnútri funkcie:

def square(x):
    global value    # syntax na globálnu premennú
    value = x ** 2 
    return value

print(value)        # print ude fungovať keďže je to globalna premenná 

###############################################

### NESTED FUNCTIONS 

# Define three_shouts
def three_shouts(word1, word2, word3):
    """Returns a tuple of strings
    concatenated with '!!!'."""

    # Define inner
    def inner(word):
        """Returns a string concatenated with '!!!'."""
        return word + '!!!'

    # Return a tuple of strings
    return (inner(word1), inner(word2), inner(word3))

# Call three_shouts() and print
print(three_shouts('a', 'b', 'c'))

######
## POMOCOU VNORENEJ FUNKCIE VYTVARAT NOVE FUNKCIE PODLA PARAMETRA VONKAJSEJ FUNKCIE

# Define echo
def echo(n):
    """Return the inner_echo function."""

    # Define inner_echo
    def inner_echo(word1):
        """Concatenate n copies of word1."""
        echo_word = word1 * n
        return echo_word

    # Return inner_echo
    return inner_echo

# Call echo: twice
twice = echo(2)                  ### Vytvori funkciu inner_echo s n=2 a vloží ju do premennej twice (novej funkcie)

# Call echo: thrice
thrice = echo(3)

# Call twice() and thrice() then print
print(twice('hello'), thrice('hello'))


##############################################

# Define echo_shout()
def echo_shout(word):
    """Change the value of a nonlocal variable"""
    
    # Concatenate word with itself: echo_word
    echo_word = word + word
    
    #Print echo_word
    print(echo_word)
    
    # Define inner function shout()
    def shout():
        """Alter a variable in the enclosing scope"""    
        #Use echo_word in nonlocal scope
        nonlocal echo_word                       # nonlocal je o triedu vyssie ako local, ale nie je to global este  
        
        #Change echo_word to echo_word concatenated with '!!!'
        echo_word = echo_word + "!!!"            # keby nebolo to nonlocal tak by nevedeko najst echo_word     
    
    # Call function shout()
    shout()
    
    #Print echo_word
    print(echo_word)

#Call function echo_shout() with argument 'hello'    
echo_shout("hello")


################################################

## FLEXIBLE ARGUMENTS 

# Define gibberish
def gibberish(*args):                                     # Rozumiem tomu tak, že ak dáme *args, tak tá hviezdička zoberie všetky argumenty
    """Concatenate strings in *args together."""

    # Initialize an empty string: hodgepodge
    hodgepodge = ""

    # Concatenate the strings in args
    for word in args:                                     # for loopujeme cez všetky zadané argumenty
        hodgepodge += word

    # Return hodgepodge
    return hodgepodge 

# Call gibberish() with one string: one_word
one_word = gibberish("luke")

# Call gibberish() with five strings: many_words
many_words = gibberish("luke", "leia", "han", "obi", "darth")

# Print one_word and many_words
print(one_word)
print(many_words)


#Alebo:
def sum_all(*args):
    print(sum(args))
sum_all(7,8,9)




##############################################################

#### KWARGS

# Define report_status
def report_status(**kwargs):
    """Print out the status of a movie character."""

    print("\nBEGIN: REPORT\n")

    # Iterate over the key-value pairs of kwargs
    for key, value in kwargs.items():
        # Print out the keys and values, separated by a colon ':'
        print(key + ": " + value)

    print("\nEND REPORT")

# First call to report_status()
report_status(name = "luke", affiliation = "jedi", status = "missing")

# Second call to report_status()
report_status(name = "anakin", affiliation = "sith lord", status = "deceased")

# Ide o akúsi viacargumentovu funkciu, ale s dvojicou vstupov (dictionaries). Cez kwargs.items() prehladávame key a value(s)


##############################################################

# Praktický príklad ešte na *args

# Define count_entries()
def count_entries(df, *args):
    """Return a dictionary with counts of
    occurrences as value for each key."""
    
    #Initialize an empty dictionary: cols_count
    cols_count = {}
    
    # Iterate over column names in args
    for col_name in args:
    
        # Extract column from DataFrame: col
        col = df[col_name]
    
        # Iterate over the column in DataFrame
        for entry in col:
    
            # If entry is in cols_count, add 1
            if entry in cols_count.keys():
                cols_count[entry] += 1
    
            # Else add the entry to cols_count, set the value to 1
            else:
                cols_count[entry] = 1

    # Return the cols_count dictionary
    return cols_count

# Call count_entries(): result1
result1 = count_entries(tweets_df, "lang")

# Call count_entries(): result2
result2 = count_entries(tweets_df, "lang", "source")

# Print result1 and result2
print(result1)
print(result2)

############################################################
###### FAST & DIRTY FUNCTION :    LAMBDA FUNCTION <3 <3 ####
############################################################

def power(x,y):
    return x**y

print(power(7,2))

# Toto sa dá napísať ako one-liner pomocou lambda funkcie:
# Syntax:   function_name = lamva var1, var2, var3,.... varN:  telo_funkcie čo sa má return-núť 

power = lambda x,y: x**y   # vyžitie najmä pri rýchlych funkciách (anonýmne funkcie):
map(func, seq)             # príklad

nums = [48,6,9,21,1]
square_all = map(lambda num: num ** 2, nums)
print(square_all)                   # je to map objekt
print(list(square_all))             # vráti už hodnoty


################
### FILTER 

# Create a list of strings: fellowship
fellowship = ['frodo', 'samwise', 'merry', 'aragorn', 'legolas', 'boromir', 'gimli']

# Use filter() to apply a lambda function over fellowship: result
result = filter(lambda member: len(member) > 6 , fellowship)

# Convert result to a list: result_list
result_list = list(result)

# Convert result into a list and print it
print(result_list)


################
## REDUCE

from functools import reduce
?reduce

#Apply a function of two arguments cumulatively to the items of a sequence,
#from left to right, so as to reduce the sequence to a single value.
#For example, reduce(lambda x, y: x+y, [1, 2, 3, 4, 5]) calculates
#((((1+2)+3)+4)+5).  If initial is present, it is placed before the items
#of the sequence in the calculation, and serves as a default when the
#sequence is empty.

#Spajanie textu:

# Import reduce from functools
from functools import reduce 

# Create a list of strings: stark
stark = ['robb', 'sansa', 'arya', 'eddard', 'jon']

# Use reduce() to apply a lambda function over stark: result
result = reduce(lambda item1, item2: item1+item2, stark)

# Print the result
print(result)





