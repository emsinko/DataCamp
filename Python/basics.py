# Example, do not modify!

print(5 / 8)



# Put code below here

print(7+10)

10 // 7 #floor division 
10 ** 2 #mocnenie 
type(variable) # typ premennej

"text" alebo 'text' robí to isté
"ab"+"bc" -> "abbc"	
int(),bool(),str() # zmena typu


### List 

# area variables (in square meters)

hall = 11.25

kit = 18.0

liv = 20.0

bed = 10.75

bath = 9.50


# house information as list of lists

house = [["hallway", hall],
         
	["kitchen", kit],
         
	["living room", liv],
         
	["bedroom", bed],
         
	["bathroom",bath]]


# Print out house

print(house)



# Print out the type of house

print(type(house))


### SUBSETTING 

[0,1,2,3,4] 		# aj zero index je, t.j. nie ako v Rku od jednotky
[-5,-4,-3,-2,-1] 	# minusovy index znamena ze chceme zo zadu nejaku hodnotu, t.j. neznamena to ze chceme "vsetko okrem" ako je v Rku... AVŠAK POZOR, TU UŽ JE NORMALNY INDEX A NEEXISTUJE x[-0]
[strat:end]             # pozor vrati to iba [start ..... end-1] 
[:4]           	        # vráti indexy 0 až 3 	
[5:]           	        # vráti indexy 5 až posledne 	

x = [["a", "b", "c"],
     ["d", "e", "f"],
     ["g", "h", "i"]]
x[2][0]                #vráti 3 list a prvý prvok v nom, t.j. g
x[2][:2]


### MANIPULATION

#klasicke nahradzovanie v liste/vektore   	x[nieco] = nove
#pridanie na koniec   				novy_list = stary_list + ["nove",cislo]
#vymazanie 3tieho elementu :			del(x[2])	

## Bacha na jednu zaludnost : 

x = [1,2,3]
y = x   # vytvori dvojicku, ale takú že ak niečo zmenis explicitne v y.. napr. y[1] = 8  tak aj x aj y sa zmenia na [1,8,3]

Osetruje sa to napriklad : y = x[:] , alebo y = list(x)		

### FUNKCIE

len() #length
max()
type()
help(), alebo ?
full_sorted = sorted(full,reverse = True)   # sortovanie Descendingove... defualtne je reverse = False

### METHODS

kazdy typ premennej ma svoje "funkcie" ktore vieme aplikovat. 

Napríklad na object s typeom list použijeme funkciu append nasledovne:   x = [1,2,3] ,  x.append(4) , t.j. syntax je ako keby variable.method(arg1,arg2,...)
Iný príklad : x.index(2) ti vráti index pre číslo 2 v liste x, t.j. v tomto prípade 3

room = "poolhouse"
room.upper()     ----> "POOLHOUSE"
room.capitalize()  --> "Poolhouse"

# Print out the number of o's in room

print(room.count("o"))  		## Pozor funkcia sa aplikuje na object room v ktorom je uložené poolhouse

# Tieto metódy sa použijú podobne: var.append(co_chcem_pridat), 
•append(), that adds an element to the list it is called on,
•remove(), that removes the first element of a list that matches the input, and
•reverse(), that reverses the order of the elements in the list it is called on.


### PACKAGES

Numpy
Matplotlib 	# vizualizacia
Scikit-learn    # ML


# Import the math package  .... syntax: import nazov_packageu
import math

  
#Pouzitie nejakej funkcie alebo objectu z balika vyzaduje stále použitia syntaxe:  meno_balika.meno_funkcie

#Calculate C
C = 2*math.pi*r

     #konštanta pi sa nachádza v balíku math

# Calculate A

A = math.pi*r**2



# Build printout

print("Circumference: " + str(C))

print("Area: " + str(A))


### Import presnej funkcie z balika:
from math import pi    				## Teraz uz nemusime pisat math.pi, ale stací písať pi 
from scipy.linalg import inv as my_inv    	## Z balíka scipy nacitaj podbalík linalg a v ňom funkciu inv s aliasom my_inv. Teraz mozeme funkciu inv pouzit aj ako my_inv


### BALICEK NUMPY (numeric python)

Python nevie narabat s listami napr:   height = [1.5, 1.8, 1.7] , weight = [65,78,80] ... ak by sme chceli BMI :  weight/height**2  tak Python nevie ako narabat s vektorom (resp. vektorové operácie, ktoré Rko vie)
Opravit to vieme pomocou "vektorizacie" listu pomocou funkcie numpy.array
Numpy array nemoze obsahovat viac ako jeden type a podobne ako v Rku sa snazi to convertnut na jeden type ... vacsinou textove su univerzalne

#Inštalácia : pip3 install numpy    #pip3 tá trojka udáva verziu pythonu
import numpy as np

## numpy array funguje tak ako Rko ... ak máme python_list = [1,2,3] a dáme python_list + python_list dostaneme [1,2,3,1,2,3] ..ak máme numpy_array = np.array[1,2,3] a dáme numpy_array + numpy_array tak dostaneme [2,4,6]


### NUMPY SUBSETTING:
## numpy je FASA, lebo mozes konecne pracovat ako s Rkom:   x[x>23] fachá 

# numpy vytvára svoj vlastny objekt numpy.ndarray  --- N-dimensional array	

## 2D matice 
# vytvori z povodnej pythonovskej sublisty v liste a dostanes maticu ktoru subsetujes rovnako ako v Rku
# variable.shape   vráti rozmery matice... t.j. dim() v Rku 

## NUMPY STATISTIC:

# Funkcie (nie metódy):
np.mean()
np.median()
np.corrcoef()
np.sum()
np.sort()
np.random.normal(mean,std,number_samples)
np.column_stack()    #niečo ako cbind
np.std()

# Príklad:

# Import numpy 

import numpy as np

# Print mean height (first column)
avg = np.mean(np_baseball[:,0])
print("Average: " + str(avg))

# Print median height. Replace 'None'
med = np.median(np_baseball[:,0])
print("Median: " + str(med))

# Print out the standard deviation on height. Replace 'None'
stddev = np.std(np_baseball[:,0])
print("Standard Deviation: " + str(stddev))

# Print out correlation between first and second column. Replace 'None'
corr = np.corrcoef(np_baseball[:,0],np_baseball[:,1])
print("Correlation: " + str(corr))


################################
#### INTERMEDIATE PYTHON #######
################################


###########
### Vizualizácia (balíček Matplotlib - pyplot  ... subbalík)
###########

import matplotlib.pyplot as plt

#otravná vec je tá, že ak aj dáš plot lubovolný, tak musíš explixitne zadať aby sa to aj objavilo cez plt.show()

plt.plot(year,pop)   	 	 # line_chart
plt.scatter(year,pop, s=n_pop)	 # scatter_plot   ... argument s= ... nam hovorí, že aké veľké majú byť jednotlivé bodky (size)
plt.xlabel("year")
plt.ylabel("pop")
plt.yticks([0,2,4,6,8,10])       # aké hodnoty sa majú zobraziť na y-osi.... Môžeme pridať aj názov : plt.yticks([0,2,4,6,8,10], ["0B","2B","4B","6B","8B","10B"])    
plt.title("World population)
plt.show()			 # vykreslenie 
plt.clf()            		 # vymazanie plot pamate aby si mohol zacat odznova

plot(x,y,color="green",marker = "o",linestyle = "dashed", linewidth=2,markersize = 12)
plt.xscale("log")  #logaritmická škála

plt.hist(x,bins = 10, range=None, normed = False...) .... histogram


### Plot tuning:
plt.scatter(x = gdp_cap, y = life_exp, s = np.array(pop) * 2, c = col, alpha = 0.8)
plt.xscale('log') 

plt.xlabel('GDP per Capita [in USD]')

plt.ylabel('Life Expectancy [in years]')

plt.title('World Development in 2007')

plt.xticks([1000,10000,100000], ['1k','10k','100k'])
plt.text(1550, 71, 'India')
plt.grid(True)
plt.show()

###########
### Dictionaries
###########

world = {"slovakia":5.6, "cesko":10.1}
world["slovakia"]  ---> output je 5.6

# všeobecne :  dict = {key1:value1, key2:value2}

###Príklad

# Definition of dictionary
europe = {'spain':'madrid', 'france':'paris', 'germany':'berlin', 'norway':'oslo' }

# Print out the keys in europe
print(europe.keys())                               ### Metóda : dict.keys()  vráti hodnoty 

# Print out value that belongs to key 'norway'
print(europe["norway"])

del(europe["norway"])            # vymaže nórsko z dictionary


##########
###Dictionary of dictionaries
##########

europe = { 'spain': { 'capital':'madrid', 'population':46.77 },
           'france': { 'capital':'paris', 'population':66.03 },
           'germany': { 'capital':'berlin', 'population':80.62 },
           'norway': { 'capital':'oslo', 'population':5.084 } }

# Print out the capital of France
print(europe["france"]["capital"])

# Create sub-dictionary data
data ={"capital":"rome","population":59.83}

# Add data to europe under key 'italy'
europe["italy"] = data

# Print europe
print(europe)


#################################
#### PANDA - data manipulation
#################################


#Praca s dataframe, ktoré obsahujú rôzny typ premenných. Numpy 2D array mohla mať len jeden typ (niečo ako matrix v Rku)

# Iný spôsob tvorby dictionary:  
dict = { "country":["brazil","russia",india"], 
	 "capital":["brasilia","moscow","new delhi"],
	 "area":[8.51, 17.1, 3.286],
	 "population":[200.4, 143.5, 1252]
       }
#.... vytvorenie keys (column labels) a values (data, column by column)

import pandas as pd
pd.DataFrame(dict)  
	


####priklad

import pandas as pd

# Build cars DataFrame
names = ['United States', 'Australia', 'Japan', 'India', 'Russia', 'Morocco', 'Egypt']
dr =  [True, False, False, False, True, True, True]
cpc = [809, 731, 588, 18, 200, 70, 45]
dict = { 'country':names, 'drives_right':dr, 'cars_per_cap':cpc }
cars = pd.DataFrame(dict)
print(cars)

# Definition of row_labels
row_labels = ['US', 'AUS', 'JAP', 'IN', 'RU', 'MOR', 'EG']

# Specify row labels of cars
cars.index = row_labels                                          #LABEL NAME 

# Print cars again
print(cars)


## Načítanie dát :  cars = pd.read_csv("cars.csv",index_col = 0)   ## Index_col = 0 priradí prvý stlpec od rownames


## Subsetting: 

data["column1"]    
data[["column1","column2"]] 		# vráti stĺpce s názvom column1 a column2

data[1:4]   						# vráti riadky 2 až 5 

## POZOR !! data[1] nevráti 2. riadok... nerozumiem prečo zatiaľ, ale riadkové indexovanie funguje len ako slice-ovanie... ak dáme data[1:2] tak to vráti 2.riadok
data.loc[1]    # funguje a vráti len 2. riadok, avšak row_label by sa mal volat "1" ak som dobre pochopil

#loc : label-based
#iloc : integer position based
	
data.loc["RU"]   									# vráti riadok s Ruskom, a je to "Row as Pandas Series" ----- zobrazuje sa to ako vektor stlpcovy v konzole.. nerozumiem este tomu
data.loc[["RU"]] 									# to isté, ale vráti to ako data_frame  
data.loc[["RU","SVK","CH"]] 						# vráti tri riadky
data.loc[["RU","SVK","CH"],  ["country","capital"] 	# vráti tri riadky, ale len dva stĺpce country a capital
data.loc[:,  ["country","capital"] 					# vráti všetky riadky, ale len dva stĺpce country a capital
	


data.iloc[[1]] 							# vráti druhý riadok
data.iloc[[1,2,3]]	 					# vráti tri riadky
data.iloc[[1,2,3],  [0,1] ]				# vráti tri riadky, ale len dva stĺpce country a capital
data.iloc[:,  [0,1]]					# vráti všetky riadky, ale len dva stĺpce country a capital

# Kombinovanie iloc a loc sa dá cez metódu ix  ... treba kuknut help keď tak


### AND OR NOT ...

# ak chceme porovnať vektorové and/or/not tak nam nestaci expr1 and expr2 .... potrebujeme vektorové verzie pre numpy array:

logical_and(vec1,vec2)
logical_or(vec1,vec2)
logical_not(vec1,vec2)   ... všetko z balíka numpy teda : np.logical_..

# Both my_house and your_house smaller than 11

print(np.logical_and(my_house < 11, your_house < 11))   #premenne house su arraye 

##########
### IF, IF ELSE
##########

if condition : 
  expression1
  expression2 .....
elif condition2 :
  expresion3
else :
  expression_ifnot
  

while condition :
  expression

### ak sa podarí zacykliť python, stačí stlačit Control + C na ukončenie proces
### koniec if/elif/else je závislé od zarovnanie kódu   (4 medzery resp. tabulator), t. j. neexistuje { }


### FOR CYKLUS

for var in data :
	print(var)
	
for index,a in enumerate(areas) :
    print("room "+ str(index) + ": " + str(a) )    # enumerate urobí :  (0,areas[0]), (1,areas[1]) ....

	
for i in range(1,10):    # range funguje tak ako seq v Rku ... t.j. range(from=1, to=9, step = 1) , ale opäť platí, že to = 10 dáva čísla len po 9 
		print(i)

### FOR CYKLY CEZ DICTIONARY: 

# Definition of dictionary
europe = {'spain':'madrid', 'france':'paris', 'germany':'berlin',
          'norway':'oslo', 'italy':'rome', 'poland':'warsaw', 'austria':'vienna' }
          
# Iterate over europe
for key,value in europe.items():                             ####TREA POUZIT metódu   .items()
    print("the capital of " + key + " is " + value)


# For loop over np_baseball ... ak máme 2D array a chceme cez všetky iterovať, potrebujeme funkcie numpy.nditer(np_array)
for var in np.nditer(np_baseball):
    print(var)	
	
	
##### For loop cez panda dataframe

# Import cars data
import pandas as pd
cars = pd.read_csv('cars.csv', index_col = 0)

# Iterate over rows of cars
for lab,row in cars.iterrows():
    print(lab)
    print(row)	                             # Toto vypíše najskôr rowname/rowlabel a potom postupne vypisuje páry: "meno stlpca + hodnota v tom riadku"

	
for lab, row in cars.iterrows() :
    print(str(lab) + ": " + str(row["cars_per_cap"]))    ##Toto vypise row_label a k tomu len hod
    	
		
		
##########################################################
# Import cars data
import pandas as pd
cars = pd.read_csv('cars.csv', index_col = 0)

# Code for loop that adds COUNTRY column

for lab, row in cars.iterrows():
    cars.loc[lab,"COUNTRY"] = row["country"].upper()
		
# Print cars
print(cars)
###########################################################
### APPLY :

cars["COUNTRY"] = cars["country"].apply(str.upper)   # robí to isté ako predošlý for loop


		
	
