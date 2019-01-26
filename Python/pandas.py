############################################################################
########## POUŽITIE BALÍKA PANDAS NA MANIPULÁCIU DÁT    ####################
############################################################################

df.head(n)       # vráti prvých n riadkov
df.tail(n)       # vráti posledných n riadkov
df.info()        # vráti ako keby summary tabulku so základnými informáciami pre datafram df + aj hpočet non-NULL hodnôt


############################
# Import numpy
import numpy as np


#Poznámky: df.values vráti hodnoty z data frame ako array

# Create array of DataFrame values: np_vals
np_vals = np.array(df.values)

# Create new array of base 10 logarithm values: np_vals_log10
np_vals_log10 = np.log10(np_vals)

# Create array of new DataFrame by passing df to np.log10(): df_log10
df_log10 = np.log10(df)

# Print original and new data containers
[print(x, 'has type', type(eval(x))) for x in ['np_vals', 'np_vals_log10', 'df', 'df_log10']]
#############################################



############################################
#### Building DataFrames from scratch ######
############################################

import pandas as pd
pd.read.csv("path",index_col = 0)

#Vytvorenie:
data = {"weekday": ["MON","TUE","WED", ....],
        "city": ["Austin", "Dallas", ...],
        "signups":[7,12,3,5]
        }
users = pd.DataFrame(data=)        

# Ak nenastavíme row labels tak sa vytvoria 0,1,2,3,4,5,.....

#### LABELS:
data.columns["height","sex"]   # column label
data.index["A","B","C"...]     # row label

# Build a list of labels: list_labels
list_labels = ["year","artist","song","chart weeks"]

# Assign the list of labels to the columns attribute: df.columns
df.columns = list_labels



#####################
# Zip the 2 lists together into one list of (key,value) tuples: zipped
zipped = list(zip(list_keys,list_values))

# Inspect the list using print()
print(zipped)

# Build a dictionary with the zipped list: data
data = dict(zipped)

# Build and inspect a DataFrame from the dictionary: df
df = pd.DataFrame(data)
print(df)
#####################


##############################
##### IMPORT & EXPORT ########
##############################

sunspots = pd.read.csv("filepath", header = None)     # ak nie je header tak to treba explicitne zadefinovat
sunspot.iloc[10:20, :]                                # rýchly slice a kuk na dáta

sunspots = pd.read.csv("filepath", header = None, names = ["col_lab1", "col_lab2",....], na_values = " -1")  # pozor, v danych dátach bolo treba pridať medzeru pred -1, nie je to však všeobecne
sunspots = pd.read.csv("filepath", header = None, names = ["col_lab1", "col_lab2",....], 
                       na_values = {"col_name":[" -1"]})  # definovanie NaN-ov sa dá aj cez dictionary zadefinovaním stĺpca a konkrétnej hodnoty
                       parse_dates = [[0,1,2]]            # zlepí prvý až tretí stĺpce 

sunspots.index = sunspots["year_month_day"]               # stĺpec year_month_day" dá ako row index 
     
     

###########################
###### PLOTING  ###########
###########################

data["column_name"] -> panda_series
data["column_name"].values -> numpy_array

#Plotting arrays (matplotlib)
plt.plot(data["column_name"].values)         # x-axis má indexovanie  1,2,3, .... 
plt.show()


#Plotting (pandas) series (matplotlib)
plt.plot(data["column_name"].values)         # x-axis už má row_labels značenie 
plt.show()

#Plotting series (pandas) , t.j. vbudovaná metóda pre pandas
data["column_name"].plot()                   # ešte viac vypimpovaný graf
plt.show()

# Cely dataframe plot:
data.plot()             # vsetky stlpce vykresli ako zvlášť čiaru (line_plot) aj s legendou
plt.show()

#Ak nevidíme všetky čiary kvoli rôznym škálam, tak je fajn použiť log-scale:
plt.yscale("log")
plt.show()

#Customize plot:
data.["column_name"].plot(color="b", style = ".-", legend = True)
plt.axis(("2001","2002",0,100))


# Priklad:

# Inspect dataframe df
type(df)
df.head()

df.plot(color = "red")
plt.title("Temperature in Austin")
plt.xlabel("Hours since midnight August 1, 2010")
plt.ylabel("Temperature (degrees F)")
plt.show()

# Priklad 2:

# Plot all columns (default)
df.plot()
plt.show()

# Plot all columns as subplots
df.plot(subplots = True)
plt.show()

# Plot just the Dew Point data
column_list1 = ['Dew Point (deg F)']
df[column_list1].plot()
plt.show()

# Plot the Dew Point and Temperature data, but not the Pressure data
column_list2 = ['Temperature (deg F)','Dew Point (deg F)']
df[column_list2].plot()
plt.show()


# Priklad 3

# Make a list of the column names to be plotted: cols
cols = ["weight","mpg"]

# Generate the box plots
df[cols].plot(kind="box", subplots = True)

# Display the plot
plt.show()


#Priklad 4

# This formats the plots such that they appear on separate rows
fig, axes = plt.subplots(nrows=2, ncols=1)

# Plot the PDF
df.fraction.plot(ax=axes[0], kind='hist', normed=True, bins= 30, range=(0,.3))
plt.show()

# Plot the CDF
df.fraction.plot(ax=axes[1],kind='hist', normed=True, cumulative = True, bins= 30, range=(0,.3))
plt.show()


#####################
##### SUMMARY #######
#####################

df.describe() # vráti summary tabulku dataframe-u df, kde pocita  count,avg,std, min,Q1,med,Q3,max pre vsetky stlpce pricom ignoruje NULL hodnoty
df.mean()     # colmeans
df.std()      # colstd
df.count()    # počty v stplpcoch.    Vo všetkých ignoruje NULL

df["column1"].median() 
df.mean(axis = "columns")   # ak dáme axis = "columns" tak vráti rowmeans (ako keby robi priemer zo stlpcov)



#### Príklad 1

# Print summary statistics of the fare column with .describe()
print(df.fare.describe())

# Generate a box plot of the fare column
df.fare.plot(kind = "box")   # ekvivalent s df["fare"]

# Show the plot
plt.show()

###################

### KVANTILY !! 

# Print the 5th and 95th percentiles
print(df.quantile([0.05, 0.95]))

####################

## Príklad na subploty

# Display the box plots on 3 separate rows and 1 column
fig, axes = plt.subplots(nrows = 3, ncols = 1)

# Generate a box plot of the fare prices for the First passenger class
titanic.loc[titanic['pclass'] == 1].plot(ax=axes[0], y ='fare', kind='box')

# Generate a box plot of the fare prices for the Second passenger class
titanic.loc[titanic['pclass'] == 2].plot(ax=axes[1], y ='fare', kind='box')

# Generate a box plot of the fare prices for the Third passenger class
titanic.loc[titanic['pclass'] == 3].plot(ax=axes[2], y ='fare', kind='box')

# Display the plot
plt.show()

#################################################################

####################
### TIME SERIES  ###
####################

# Prepare a format string: time_format
time_format = "%Y-%m-%d %H:%M"

# Convert date_list into a datetime object: my_datetimes
my_datetimes = pd.to_datetime(date_list, format= time_format)  

# Construct a pandas Series using temperature_list and my_datetimes: time_series
time_series = pd.Series(temperature_list, index = my_datetimes)

####
## Subsetovanie hodin/dní 
####

# Extract the hour from 9pm to 10pm on '2010-10-11': ts1
ts1 = ts0.loc['2010-10-11 21:00:00':'2010-10-11 22:00:00']

# Extract '2010-07-04' from ts0: ts2
ts2 = ts0.loc["2010-07-04"]

# Extract data from '2010-12-15' to '2010-12-31': ts3
ts3 = ts0.loc["2010-12-15" : "2010-12-31"]

####
## REINDEX
####

# ts2 obsahuje data BEZ vikendov, ts1 má aj víkendy
# Na ts2 datach urob indexing ts1-tky... v principe v ts2-ke sa doplnia aj tie vikendove dni s tým že sa 
# použijú hodnoty z piatka alebo pondelka pre dni sobota/nedela. Ak dáme method="ffill" čo je forward fill tak
# použije posledné dostupné (piatok) a ak dáme bfill čo je backward tak sa pozerá na pondelok, t.j. dopredu 

ts4 = ts2.reindex(ts1.index, method = "ffill")  # forward fill


###########
## RESAMPLING 
###########

#Legenda:

"min", "T"  = minute
"H"         = hour
"D"         = day
"B"         = business day
"W"         = week
"M"         = month
"Q"         = quarter
"A"         = year

sales.loc[:,"Units"].resample("2W").sum()       # agreguje na 2 tyzdnovej báze nas df=sales pre stlpec Units a zosumuje to

# Downsample to 6 hour data and aggregate by mean: df1
df1 = df.Temperature.resample("6H").mean()

# Downsample to daily data and count the number of data points: df2
df2 = df.Temperature.resample("1D").count()

# Extract temperature data for August: august
august = df.Temperature.loc["2010-08"]

# Downsample to obtain only the daily highest temperatures in August: august_highs
august_highs = august.resample("D").max()

# Extract temperature data for February: february
february = df.Temperature.loc["2010-02"]

# Downsample to obtain the daily lowest temperatures in February: february_lows
february_lows = february.resample("D").min()

#############
## ROLLING date functions
#############

#Príklad1:

# Extract data from 2010-Aug-01 to 2010-Aug-15: unsmoothed
unsmoothed = df['Temperature']["2010-08-01":"2010-08-15"]

# Apply a rolling mean with a 24 hour window: smoothed
smoothed = unsmoothed.rolling(window = 24).mean()

# Create a new DataFrame with columns smoothed and unsmoothed: august
august = pd.DataFrame({'smoothed':smoothed, 'unsmoothed':unsmoothed})

# Plot both smoothed and unsmoothed data using august.plot().
august.plot()
plt.show()


#Príklad2:

# Extract the August 2010 data: august
august = df['Temperature']["2010-August"]

# Resample to daily data, aggregating by max: daily_highs
daily_highs = august.resample("D").max()

# Use a rolling 7-day window with method chaining to smooth the daily high temperatures in August
daily_highs_smoothed = daily_highs.rolling(window = 7).mean()
print(daily_highs_smoothed)


############################
### STRING METHODS
###########################

# Ak máme vv nazvoch stlpcov prilis vela WHITE SPACE-ov, tak sa daju odstrániť:

df.columns.str.strip()    # df.columns vráti názvy stlpcov a .str.strip() robí ako keby TRIM


##### Priklad 1 

# Strip extra whitespace from the column names: df.columns
df.columns = df.columns.str.strip()

# Extract data for which the destination airport is Dallas: dallas
dallas = df['Destination Airport'].str.contains("DAL")

# Compute the total number of Dallas departures each day: daily_departures
daily_departures = dallas.resample("D").sum()

# Generate the summary statistics for daily Dallas departures: stats
stats = daily_departures.describe()


##### Priklad 2

# Reset the index of ts2 to ts1, and then use linear interpolation to fill in the NaNs: ts2_interp
ts2_interp = ts2.reindex(ts1.index).interpolate(how = "linear")

# Compute the absolute difference of ts1 and ts2_interp: differences 
differences = np.abs(ts1 - ts2_interp)

# Generate and print summary statistics of the differences
print(differences.describe())



############################
### DATE TIME METHODS
###########################

# Build a Boolean mask to filter out all the 'LAX' departure flights: mask
mask = df['Destination Airport'] == "LAX"
# Use the mask to subset the data: la
la = df[mask]

# Combine two columns of data to create a datetime series: times_tz_none 
times_tz_none = pd.to_datetime( la['Date (MM/DD/YYYY)'] + ' ' + la['Wheels-off Time'] )   #nazov stlpcov

# Localize the time to US/Central: times_tz_central
times_tz_central = times_tz_none.dt.tz_localize("US/Central")     # nastavenie časového pásma

# Convert the datetimes from US/Central to US/Pacific
times_tz_pacific = times_tz_central.dt.tz_convert("US/Pacific")   # zmena časového pásma


##################
## PLOT TIME SERIES
##################

b blue     o circle     : dotted
g green    * star       - dashed
r red      s square 
c cyan     + plus 
k black

# Plot the raw data before setting the datetime index
df.plot()
plt.show()

# Convert the 'Date' column into a collection of datetime objects: df.Date
df.Date = pd.to_datetime(df.Date)    

# Set the index to be the converted 'Date' column
df.set_index("Date",inplace = True)     #optional keyword argument inplace=True, so that you don't have to assign the result back to df

# Re-plot the DataFrame to see that the axis is now datetime aware!
df.plot()
plt.show()

#####################

# Plot the summer data
df.Temperature["2010-Jun":"2010-Aug"].plot()
plt.show()
plt.clf()

# Plot the one week data
df.Temperature["2010-06-10":"2010-06-17"].plot()
plt.show()
plt.clf()

###############################
### CASE STUDY 
#############################

# Import pandas
import pandas as pd

# Read in the data file: df
df = pd.read_csv("data.csv")

# Print the output of df.head()
print(df.head())

# Read in the data file with header=None: df_headers
df_headers = pd.read_csv("data.csv", header=None)

# Print the output of df_headers.head()
print(df_headers.head())

##### DOLEŽITÉ : 

# Split on the comma to create a list: column_labels_list
column_labels_list = column_labels.split(",")

# Assign the new column labels to the DataFrame: df.columns
df.columns = column_labels_list

# Remove the appropriate columns: df_dropped
df_dropped = df.drop(list_to_drop, axis = "columns")    ### METODA .DROP()

# Print the output of df_dropped.head()
print(df_dropped.head())

####################
# Convert the date column to string: df_dropped['date']
df_dropped['date'] = df_dropped.date.astype(str)

# Pad leading zeros to the Time column: df_dropped['Time']
df_dropped['Time'] = df_dropped['Time'].apply(lambda x:'{:0>4}'.format(x))    # Prida nuly na zaciatok cisla ak nie su štvorciferné

# Concatenate the new date and Time columns: date_string
date_string = df_dropped["date"] + df_dropped["Time"]

# Convert the date_string Series to datetime: date_times
date_times = pd.to_datetime(date_string, format='%Y%m%d%H%M')

# Set the index to be the new date_times container: df_clean
df_clean = df_dropped.set_index(date_times)

# Print the output of df_clean.head()
print(df_clean.head())


####################

# Print the dry_bulb_faren temperature between 8 AM and 9 AM on June 20, 2011
print(df_clean.loc["2011-06-20 08:00:00":"2011-06-20 09:00:00", "dry_bulb_faren"])

# Convert the dry_bulb_faren column to numeric values: df_clean['dry_bulb_faren']
df_clean['dry_bulb_faren'] = pd.to_numeric(df_clean['dry_bulb_faren'], errors="coerce")    # pismena "M" teraz premení ako keby na NaN (chábajúce čísla)

# Print the transformed dry_bulb_faren temperature between 8 AM and 9 AM on June 20, 2011
print(df_clean.loc["2011-06-20 08:00:00":"2011-06-20 09:00:00", "dry_bulb_faren"])


# Convert the wind_speed and dew_point_faren columns to numeric values
df_clean['wind_speed'] = pd.to_numeric(df_clean.wind_speed, errors = "coerce")
df_clean['dew_point_faren'] = pd.to_numeric(df_clean.dew_point_faren, errors="coerce")

######################

# POZNAMKY :   .values          nám ako keby z jednostlpcoveho dataframeu zoberie iba numpy array
#              .reset_index     nám resetne aktuálny index (napr. dátumový) a dá to na integerovú sekvenciu

# Downsample df_clean by day and aggregate by mean: daily_mean_2011
daily_mean_2011 = df_clean.resample("D").mean()

# Extract the dry_bulb_faren column from daily_mean_2011 using .values: daily_temp_2011
daily_temp_2011 = daily_mean_2011.dry_bulb_faren.values

# Downsample df_climate by day and aggregate by mean: daily_climate
daily_climate = df_climate.resample("D").mean()

# Extract the Temperature column from daily_climate using .reset_index(): daily_temp_climate
daily_temp_climate = daily_climate.reset_index()["Temperature"]

# Compute the difference between the two arrays and print the mean difference
difference = daily_temp_2011 - daily_temp_climate
print(difference.mean())

############################################xx

# Select days that are sunny: sunny
sunny = df_clean.loc[df_clean.sky_condition == "CLR"]

# Select days that are overcast: overcast
overcast = df_clean.loc[df_clean.sky_condition.str.contains("OVC")]

# Resample sunny and overcast, aggregating by maximum daily temperature
sunny_daily_max = sunny.resample("D").max()
overcast_daily_max = overcast.resample("D").max()

# Print the difference between the mean of sunny_daily_max and overcast_daily_max
print(sunny_daily_max.mean() - overcast_daily_max.mean())

##############################################

# Vizualizovaná IDE 

# Import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

# Select the visibility and dry_bulb_faren columns and resample them: weekly_mean
weekly_mean = df_clean.loc[:,["visibility","dry_bulb_faren"]].resample("W").mean()

# Print the output of weekly_mean.corr()
print(weekly_mean.corr())

# Plot weekly_mean with subplots=True
weekly_mean.plot(subplots = True)
plt.show()

######

# Create a Boolean Series for sunny days: sunny
sunny = df_clean.sky_condition == "CLR"

# Resample the Boolean Series by day and compute the sum: sunny_hours
sunny_hours = sunny.resample("D").sum()

# Resample the Boolean Series by day and compute the count: total_hours
total_hours = sunny.resample("D").count()

# Divide sunny_hours by total_hours: sunny_fraction
sunny_fraction = sunny_hours / total_hours

# Make a box plot of sunny_fraction
sunny_fraction.plot(kind = "box")
plt.show()

######

# Resample dew_point_faren and dry_bulb_faren by Month, aggregating the maximum values: monthly_max
monthly_max = df_clean.loc[:, ["dew_point_faren", "dry_bulb_faren"]].resample("M").max()

# Generate a histogram with bins=8, alpha=0.5, subplots=True
monthly_max.plot(kind = "hist", bins = 8, alpha = 0.5, subplots = True)

# Show the plot
plt.show()



def pace(h,m,s = 0):
    x = (60*h+m+s/60)/345
    print(str(math.floor(x)) + ":" + str(round((x % 1)*60)))
    
pace(23,9,0)








###########################################################################x
###########################################################################x
###########################################################################x
###########################################################################x

# ROZDIEL MEDZI df[] a df.loc[]
https://stackoverflow.com/questions/48409128/what-is-the-difference-between-using-loc-and-using-just-square-brackets-to-filte

#Praca s dataframe, ktoré obsahujú rôzny typ premenných.
# Numpy 2D array mohla mať len jeden typ (niečo ako matrix v Rku)

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