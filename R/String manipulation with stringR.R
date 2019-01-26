################################

#   STRING MANIPULATION IN R   #

################################


# Poznamka: ak je v texte aj ' aj ", tak je idealne dat string do "" a v nom kazde pouzitie " dat ako \" (kde \ je escape character)
#           3 casei: 

# Define line1
line1 <- "The table was a large one, but the three were all crowded together at one corner of it:"

# Define line2
line2 <- '"No room! No room!" they cried out when they saw Alice coming.'

# Define line3
line3 <- "\"There's plenty of room!\" said Alice indignantly, and she sat down in a large arm-chair at one end of the table."

# Problem: Ak dame print line2, tak nam printne:
print(line2)  #  "\"No room! No room!\" they cried out when they saw Alice coming."  .... t.j. aj escape character

###############################

### When you ask R for line2 it is actually calling print(line2) and the print() method for strings displays strings 
### as you might enter them. If you want to see the string it represents you'll need to use a different function: writeLines().

### You can pass writeLines() a vector of strings and it will print them to the screen, each on a new line. 
### This is a great way to check the string you entered really does represent the string you wanted.


# Putting lines in a vector
lines <- c(line1, line2, line3)

# Print lines
print(lines)

# Use writeLines() on lines   .... defaulte je separator "\n", t.j. novy riadok
writeLines(lines)                        

# Write lines with a space separator
writeLines(lines,sep = " ")

# Use writeLines() on the string "hello\n\U1F30D"
writeLines("hello\n\U1F30D",sep = " ")

# In "hello\n\U1F30D" there are two escape sequences: \n gives a newline, 
# and \U followed by up to 8 hex digits sequence denotes a particular Unicode character. 

###############################

# Should display: To have a \ you need \\
writeLines("To have a \\ you need \\\\")

# Should display: 
# This is a really 
# really really 
# long string
writeLines("This is a really \nreally really \nlong string")

# Use writeLines() with 
# "\u0928\u092e\u0938\u094d\u0924\u0947 \u0926\u0941\u0928\u093f\u092f\u093e"
writeLines("\u0928\u092e\u0938\u094d\u0924\u0947 \u0926\u0941\u0928\u093f\u092f\u093e")

# You just said "Hello World" in Hindi!

###############################

# Some vectors of numbers
percent_change  <- c(4, -1.91, 3.00, -5.002)
income <-  c(72.19, 1030.18, 10291.93, 1189192.18)
p_values <- c(0.12, 0.98, 0.0000191, 0.00000000002)

# Format c(0.0011, 0.011, 1) with digits = 1
format(c(0.0011, 0.011, 1), digits = 1)      # 1 signifikant digit,  s tym ze vsetky sa upravia podla toho.. v tomto pripade na 3 desatinne kvoli 0.0011

# Format c(1.0011, 2.011, 1) with digits = 1
format(c(1.0011, 2.011, 1), digits = 1)      # vsetko bude tym padom na cele cisla bez desatinnych

# Format percent_change to one place after the decimal point
format(percent_change,digits = 2)   

# Format income to whole numbers
format(income,digits = 2)   # vsimnime si medzeri, ktore vzniknu navyse pred "kratsimi" cislami. Je to v principe kvoli tomu, bay verikalne 
                            # boli zarovnane tie cisla podla decimal pointu. Ovplyvnit to mozeme pomocou trim = TRUE

# Format p_values in fixed format
format(p_values,scientific = FALSE)

###############################

formatted_income <- format(income, digits = 2)

# Print formatted_income
print(formatted_income)

# Call writeLines() on the formatted income
writeLines(formatted_income)  #pekne pod seba to vypise + zarovnanie kvoli trim = FALSE (defaultne vo format)

# Define trimmed_income
trimmed_income <- format(income, digits = 2, trim = TRUE)

# Call writeLines() on the trimmed_income
writeLines(trimmed_income)  # uz je to zarovnane dolava

# Define pretty_income
pretty_income <- format(income, digits = 2, big.mark = ",")  # po kazdych 3 cislach da ciarku: 1000000 -> 1,000,000

# Call writeLines() on the pretty_income
writeLines(pretty_income)

#################################

### The function formatC() provides an alternative way to format numbers based on C style syntax. 
#•"f" for fixed, 
#•"e" for scientific, and 
#•"g" for fixed unless scientific saves space

# digits is the number of digits after the decimal point. This is more predictable than format(), because the number of places 
# after the decimal is fixed regardless of the values being formatted. !!!! Toto sa mi paci, kedze pri format() to nebolo o decimal places, ale o significant places

# From the format() exercise
x <- c(0.0011, 0.011, 1)
y <- c(1.0011, 2.011, 1)

# formatC() on x with format = "f", digits = 1
formatC(x,format = "f", digits = 1)

# formatC() on y with format = "f", digits = 1
formatC(y,format = "f", digits = 1)

# Format percent_change to one place after the decimal point
formatC(percent_change, format = "f", digits = 1)

# percent_change with flag = "+"
formatC(percent_change, format = "f", digits = 1, flag = "+") # Kladne cisla vypise aj so znamienkom +

# Format p_values using format = "g" and digits = 2
formatC(p_values, format = "g", digits = 2)

#####################################

# Add $ to pretty_income
paste("$",pretty_income,  sep = "")

# Add % to pretty_percent
paste(pretty_percent, "%", sep = "")

# Create vector with elements like 2010: +4.0%`
year_percent <- paste(years,": ",paste(pretty_percent, "%", sep = ""), sep = "")

# Collapse all years into single string
paste(year_percent,collapse = ", ")

## Poznamka:  paste0() je to iste ako paste(..., sep = "")

#####################################

# Define the names vector
income_names <- c("Year 0", "Year 1", "Year 2", "Project Lifetime")

# Create pretty_income
pretty_income <- format(income,digits = 2, big.mark = ",")

# Create dollar_income
dollar_income <- paste0("$", pretty_income)   # ak dame trim = TRUE, tak dollar sign bude hned pred cislom

# Create formatted_names
formatted_names <- format(income_names, justify = "right")  # zarovnanie doprava

# Create rows
rows <- paste(formatted_names, dollar_income, sep = "   ")

# Write rows
writeLines(rows)

#####################################


# Randomly sample 3 toppings
my_toppings <- sample(toppings, size = 3)

# Print my_toppings
print(my_toppings)

# Paste "and " to last element: my_toppings_and
my_toppings_and <- paste0(c("","","and "),my_toppings)

# Collapse with comma space: these_toppings
these_toppings <- paste(my_toppings_and, collapse = ", ")

# Add rest of sentence: my_order
my_order <- paste0("I want to order a pizza with ",these_toppings,".")

# Order pizza with writeLines()
writeLines(my_order)

################################
### INTRODUCTION TO STRINGR  ###
################################

# all functions start whith str_
# first argument is vector of strings

library(stringr)


my_toppings <- c("cheese", NA, NA)
my_toppings_and <- paste(c("", "", "and "), my_toppings, sep = "")

# Print my_toppings_and
print(my_toppings_and)

# Use str_c() instead of paste(): my_toppings_str
my_toppings_str <- str_c(c("", "", "and "),my_toppings)

# Print my_toppings_str
my_toppings_str

# paste() my_toppings_and with collapse = ", "
paste(my_toppings_and, collapse = ", ")

# str_c() my_toppings_str with collapse = ", "
str_c(my_toppings_str, collapse = ", ")

# Poznamka: Paste premeni NA na "NA", kdezto str_c to povazuje za NA a pri collapse nam vrati NA,
# t.j. dokazeme hned odhalit, ze v texte mame missing value. Nahradit mozeme pomocou str_replace_na()


#######
### Babynames dataset
#######

#install.packages("babynames")
library(babynames)
library(dplyr)

# Extracting vectors for boys' and girls' names
babynames_2014 <- filter(babynames, year == 2014)
boy_names <- filter(babynames_2014, sex == "M")$name
girl_names <- filter(babynames_2014, sex == "F")$name

# Funkcia str_length() zobere vektor charakterov a vrati dlzku kazdeho slova
str_length(c("Bruce", "Wayne"))
nchar(c("Bruce", "Wayne"))        # ma svoje nevyhody...

# Take a look at a few boy_names
head(boy_names)

# Find the length of all boy_names
boy_length <- str_length(boy_names)

# Take a look at a few lengths
head(boy_length)

# Find the length of all girl_names
girl_length <- str_length(girl_names)

# Find the difference in mean length
mean(girl_length) - mean(boy_length) 

# Confirm str_length() works with factors
head(str_length(factor(boy_names)))


###

# Extract first letter from boy_names
boy_first_letter <- str_sub(boy_names, 1,1)

# Tabulate occurrences of boy_first_letter
table(boy_first_letter)

# Extract the last letter in boy_names, then tabulate
boy_last_letter <- str_sub(boy_names,-1,-1)
table(boy_last_letter)

# Extract the first letter in girl_names, then tabulate
girl_first_letter <-  str_sub(girl_names, 1,1)
table(girl_first_letter)

# Extract the last letter in girl_names, then tabulate
girl_last_letter <- str_sub(girl_names,-1,-1)   # ak by sme dali -4 a -1 tak by sme dostali posledne 4 pismena
table(girl_last_letter)

###

## FUNKCIE str_detect(string, pattern) -> , str_subset()

# Look for pattern "zz" in boy_names
contains_zz <- str_detect(string = boy_names, pattern = "zz")

# Examine str() of contains_zz
str(contains_zz)

# How many names contain "zz"?
sum(contains_zz)

# Which names contain "zz"?
boy_names[contains_zz]
str_subset(boy_names, pattern =  "zz") # toto je dva kroky v jednom..

# Which rows in boy_df have names that contain "zz"?
boy_df[contains_zz, ]

####

# Find boy_names that contain "zz"
str_subset(boy_names, pattern = "zz")

# Find girl_names that contain "zz"
str_subset(girl_names, pattern = "zz")

# Find girl_names that contain "U"
starts_U <- str_subset(girl_names, pattern  = "U")
starts_U

# Find girl_names that contain "U" and "z"
str_subset(starts_U, pattern = "z")

#####

# Count occurrences of "a" in girl_names
number_as <- str_count(girl_names, "a")

# Count occurrences of "A" in girl_names
number_As <- str_count(girl_names, "A")

# Histograms of number_as and number_As
hist(number_as)
hist(number_As)

# Find total "a" + "A"
total_as <- number_As + number_as

# girl_names with more than 4 a's
girl_names[total_as > 4]

#####

# str_split:    "Tom & Jerry"  -->  str_split(string = "Tom & Jerry", patter = " & ") --> vrati list s dvomi char premennymi
# str_split:       -->  str_split(string = "Tom & Jerry & Nikolas", patter = " & ", n = 2) --> n=2 znamena ze len raz nam to rozdeli, t.j. chceme 2 char texty

# Preco vrati list? Lebo ak mu dame vektor textov, tak nevieme zarucit ze po splite budu mat rovnaky pocet rozsplitovanych textov
# Vieme to vsak dat do matice pomocou simplify = TRUE,


# Some date data
date_ranges <- c("23.01.2017 - 29.01.2017", "30.01.2017 - 06.02.2017")

# Split dates using " - "
split_dates <- str_split(date_ranges, pattern  = " - ")
split_dates

split_dates_n <- str_split(date_ranges, pattern  = " - ", simplify = TRUE)
split_dates_n


# Subset split_dates_n into start_dates and end_dates
start_dates <- split_dates_n[,1]

# Split start_dates into day, month and year pieces
str_split(start_dates, pattern = fixed("."), simplify = TRUE)

####

# Split lines into words
words <- str_split(lines, pattern = " ")

# Number of words per line
lapply(words,length)

# Number of characters in each word
word_lengths <- lapply(words,str_length)

# Average word length per line   -- toto nie je uplne spravne, nakolko pouzivame tzv. punctuation symbols: \n, ktore tiez zaratame
lapply(word_lengths, mean)


#####
# Replacing matches in strings
#####

str_replace("Tom & Jerry & Nikolas", pattern = "&", replacement = "and")  # Nahradi len prve &
str_replace_all("Tom & Jerry & Nikolas", pattern = "&", replacement = "and")  # Nahradi lvsetky

###

# Some IDs
ids <- c("ID#: 192", "ID#: 118", "ID#: 001")

# Replace "ID#: " with ""
id_nums <- str_replace_all(ids,pattern = "ID#: ", replacement = "")

# Turn id_nums into numbers
id_ints <- as.numeric(id_nums)

###

# Some (fake) phone numbers
phone_numbers <- c("510-555-0123", "541-555-0167")

# Use str_replace() to replace "-" with " "
str_replace(phone_numbers, "-", " ")

# Use str_replace_all() to replace "-" with " "
str_replace_all(phone_numbers, "-", " ")

# Turn phone numbers into the format xxx.xxx.xxxx
str_replace_all(phone_numbers, fixed("-"), ".")   ### fixed() zabranuje, aby pri pattern nebolo vyhladavane REGEX

###

# Find the number of nucleotides in each sequence
str_length(genes)

# Find the number of A's occur in each sequence
str_count(genes,pattern = fixed("A"))

# Return the sequences that contain "TTTTTT"
str_subset(genes,pattern = fixed("TTTTTT"))

# Replace all the "A"s in the sequences with a "_"
str_replace_all(genes, fixed("A"), "_")

####

# Define some full names
names <- c("Diana Prince", "Clark Kent")

# Split into first and last names
names_split <- str_split(names, pattern = fixed(" "), simplify = TRUE)

# Extract the first letter in the first name
abb_first <- str_sub(names_split[,1],start = 1,end = 1)

# Combine the first letter ". " and last name
str_c(abb_first,".",names_split[,2])

# Use all names in babynames_2014
all_names <- babynames_2014$name

# Get the last two letters of all_names
last_two_letters <- str_sub(all_names, start = -2, end = -1)

# Does the name end in "ee"?
ends_in_ee <- str_detect(last_two_letters, pattern = fixed("ee"))

# Extract rows and "sex" column
sex <- babynames_2014$sex[ends_in_ee]

# Display result as a table
table(sex)


########################
## REGULAR EXPRESSION ##
########################

install.packages("rebus",dependencies = TRUE)
install.packages("htmlwidgets")
library(rebus)


# ^.[\d] +   --> the start of the string, followed by any single character, followed by one or more digits
START %R% ANY_CHAR %R% one_or_more(DGT)   

# Start string with c
START %R% "c"

####

# Some strings to practice with
x <- c("cat", "coat", "scotland", "tic toc")

# Print END
END  # <regex> $

# Run me
str_view(x, pattern = START %R% "c")  # tu je potrebne mat nainstalovane htmlwidgets 

# Match the strings that start with "co" 
str_view(x, pattern = START %R% "co")

# Match the strings that end with "at"
str_view(x, pattern = "at" %R% END)

# Match the string that is exactly "cat"
str_view(x, pattern = START %R% "cat" %R% END)

# ANY CHAR (any single char) --> Regex: . 

# Match two characters, where the second is a "t"  --> !! NEMUSI ZACINAT, HOCIAKE DVA
str_view(x, pattern = ANY_CHAR %R% "t")

# Match a "t" followed by any character
str_view(x, pattern = "t" %R% ANY_CHAR)

# Match two characters
str_view(x, pattern = ANY_CHAR %R% ANY_CHAR)  # Pozor... REGEX je lenivy, match-ne len prve co najde .. v tomto pripade vzdy prve dva znaky 

# Match a string with exactly three characters
str_view(x, pattern = START %R% ANY_CHAR %R% ANY_CHAR %R% ANY_CHAR %R% END)  # <regex> ^...$

#####


# Find names that have the pattern
names_with_q <- str_subset(boy_names, pattern = "q" %R% ANY_CHAR) 

# How many names were there?
length(names_with_q)

# Find part of name that matches pattern
part_with_q <- str_extract(boy_names, pattern = "q" %R% ANY_CHAR)

# Get a table of counts
table(part_with_q)



# Did any names have the pattern more than once?
count_of_q <- str_count(boy_names, pattern = pattern)

# Get a table of counts
table(count_of_q)



# Which babies got these names?
with_q <- str_detect(boy_names, pattern = "q.")

# What fraction of babies got these names?
mean(with_q)

###########
### REGULAR EXPRESSION
###########

#  Regular expression: nie ANY_CHARACTER, ale presne hladame .^$ -->  \. , \^, \$
#  Rebus: DOT, COLAR, DOLLAR

## ALTERNATION:
rebus: or("dog", "cat") .... regex : (?:dog|cat) 

rebus: char_class("Aa") .... regex: [Aa]   --> bud A alebo a  (resp. nieco z toho)
rebus: negated_char_class("Aa") ... regex: [^Aa] --> matchne hocico ine ako "A" alebo "a"... z "Aarona" matche "r"

# Repetition

rebus: optional() ....   regex: ?
  rebus: zero_or_more() .. regex: *
  rebus: one_or_more().... regex: +        ---> pattern = one_or_more("Aa") matchne aj "apple" na "a" a aj "Aaron" na "Aa"
rebus: repeated()......  regex:{n}{m}


# Match Jeffrey or Geoffrey
whole_names <- or("Jeffrey", "Geoffrey")
str_view(boy_names, pattern = whole_names, match = TRUE)ò

# Match Jeffrey or Geoffrey, another way
common_ending <- START %R% or("Je","Geo") %R% "ffrey"
str_view(boy_names, pattern = common_ending, match = TRUE)  # match =  TRUE  --> v HTML sa zoberazia len matchnute stringy  

# Match with alternate endings
by_parts <- START %R% or("Je", "Geo") %R% "ff" %R% or("ry","ery","rey", "erey")
str_view(boy_names, pattern = by_parts, match = TRUE) #Jeffrey, Jeffery, Geoffrey, Jeffry, Jefferey

# Match names that start with Cath or Kath
ckath <- START %R% or("C","K") %R% "ath"
str_view(girl_names, pattern = ckath, match = TRUE)

######

# Create character class containing vowels
vowels <- char_class("aeiouAEIOU")     ## Char_class vytvori regex [aeiouAEIOU] s pri napr, str_detect() matchne lubovolny charakter

# Print vowels
vowels

# Create character class containing vowels
vowels <- char_class("aeiouAEIOU")

# Print vowels
vowels

x <- c("grey sky", "gray elephant")

# See vowels in x with str_view()
str_view(x, pattern = vowels)

# See vowels in x with str_view()
str_view_all(x, pattern = vowels)    # matchne vsetky zhody v jednom stringu a nie len prvy match


#### boy_names

# Number of vowels in boy_names
num_vowels <- str_count(boy_names, pattern = vowels)

# Proportion of vowels in boy_names ... no proportion, but number of characters 
name_length <- str_length(boy_names)

# Calc mean number of vowels
mean(num_vowels)

# Calc mean fraction of vowels per name
mean(num_vowels / name_length)

#### 

# Vowels from last exercise
vowels <- char_class("aeiouAEIOU")

# See names with only vowels
str_view(boy_names, 
         pattern = START %R% one_or_more(vowels) %R% END, 
         match = TRUE)


str_view(x, pattern = zero_or_more(vowels))   ## .. toto matchne vsetko, u niektorych len START (zaciatok stringu = este pred 1. charakterom)



# Use `negated_char_class()` for everything but vowels
not_vowels <- negated_char_class("aeiouAEIOU")

# See names with no vowels
str_view(boy_names, 
         pattern = exactly(one_or_more(not_vowels)), 
         match = TRUE)


#######
## SHORTCUTS
#######


### Zacina dolarovym znakom a nasledne cislom :   regex = [\$["0123456789"]], alebo skratene [\$[0-9]
char_class("0-9")

## Match any lower case char:
char_class("a-z") .... regex.[a-z]

## A digit:
rebus: DGT alebo char_class("0-9") ......... regex:[0-9] alebo \d


## A word
rebus: WRD alebo char_class("a-za-z0-9_-")

## SPACE
rebus: SPC ... regex: \s

#####
contact <- c("Call me at 555-555-0191", "123 Main St","(555) 555 0191", "Phone: 555.555.0191 Mobile: 555.555.0192")

contact

# Create a three digit pattern
three_digits <- "[0-9]{3}"
three_digits <-  DGT %R% DGT %R% DGT
three_digits <- "[0-9][0-9][0-9]"

#Test it
str_view_all(contact, pattern = three_digits)

# Create a separator pattern
separator <- char_class("-.() ")

# Test it
str_view_all(contact, pattern = separator)

####

# Use these components
three_digits <- DGT %R% DGT %R% DGT
four_digits <- three_digits %R% DGT
separator <- char_class("-.() ")

# Create phone pattern
phone_pattern <- zero_or_more(OPEN_PAREN) %R%
  three_digits %R%
  zero_or_more(separator) %R%
  three_digits %R% 
  zero_or_more(separator) %R%
  four_digits

# Test it           
str_view_all(contact, pattern = phone_pattern)


# Extract phone numbers
str_extract(contact, pattern = phone_pattern)  # ak zo stringu nenajde pattern tak vrati NA (pre 123 Main St)

# Extract ALL phone numbers
str_extract_all(contact, pattern = phone_pattern)  # vrati list, kedze moze sa vyskytnut viac matchov pre jeden string


######
######

# Pattern to match one or two digits
age <- DGT %R% optional(DGT)

# Test it
str_view(narratives, pattern = age)

# Pattern to match one off units 
unit <- optional(SPC) %R% or("YO","YR","MO")

# Test pattern with age then units
str_view(narratives, pattern = age %R% unit)

# Pattern to match gender
gender <- or("M","F")

# Test pattern with age then units then gender
sr_view(narratives, pattern = age %R% unit %R% gender)

# Extract age, unit, gender
str_extract(narratives, pattern = age %R% unit %R% gender)

### FINTA:
'Miesto  DGT %R% optional(DGT) mozeme napisat dgt(1,2)'

######
# age_gender, age, gender, unit are pre-defined
ls.str()

# Extract age and make numeric
as.numeric(str_extract(age_gender, pattern = age))


# Replace age and units with ""
genders <- str_remove(age_gender, pattern = age %R% unit)

# Replace extra spaces
str_remove(genders, pattern = SPC)  # alebo str_trim()

'str_remove() je to iste ako str_replace(string, pattern = " ")'

# Numeric ages, from previous step
ages_numeric <- as.numeric(str_extract(age_gender, age))

# Extract units 
time_units <- str_extract(age_gender, pattern = "[\\s]?(?:YO|YR|MO)")  #or("YO","YR","MO")

# Extract first word character
time_units_clean <- str_extract(time_units, pattern = WRD)

# Turn ages in months to years
ifelse(time_units_clean == "Y", ages_numeric, ages_numeric/12)


#####
## CAPTURING
#####

# Capture ako keby oddeli celkovy regex na viacej regexov. 
ANY_CHAR %R% "a"                 #.a
capture(ANY_CHAR) %R% "a"        #(.) 


#Funguje to rovanko:
str_extract(c("Faat","cat"), pattern = "(.)a")
str_extract(c("Faat","cat"), pattern = ".a")

#Avsak, str_match() je prisposobeny na "capture" regexy:
str_match(c("Faat","cat"), pattern = "(.)a") # vrati najskor "celkový" regex match a nasledne ciastocny (v "capture") regex match
str_match(c("FRa1t","ca2t"), pattern = "(.)a(\\d)") # najprv vrati aa1 (celkovy regex match) potom prvy capture match z celkoveho matchu =  R je znak (.) pred "a" a nasledne cislo 1 ktore predstavuje (\\d)
# Prvy match (celkovy) je ten co dostaneme zo str_extract()

## T.j. capture nemeni celkove chovanie regexu, avsak indikuje to, ze by sme sa chceli pozriet aj na ciastkovy vysledok capture regexu, ktory davame do (....)

#Regex $jednotiek.centy ,maximalne do $99.99:
str_view(c("$5.50","$32.00"), pattern = DOLLAR %R% DGT %R% optional(DGT) %R% DOT %R% dgt(2))
str_view(c("$5.50","$32.00"), pattern = "\\$\\d[\\d]?\\.[\\d]{2}")  # rovnake prepisane bez rebusu do regexu 

#Ak by sme vsak chceli z nejakeho dovodu regexnut len hodnoty do 100$ a dalej by sme chceli analyzovat zvlast centy a zvlast cele dolare:
str_match(c("$5.50","$32.00"), pattern = DOLLAR %R% capture(DGT %R% optional(DGT)) %R% DOT %R% capture(dgt(2)))

### 1) Vyuzitie v emailoch:
# Capture parts between @ and . and after .
email <- capture(one_or_more(WRD)) %R% 
  "@" %R% capture(one_or_more(WRD)) %R% 
  DOT %R% capture(one_or_more(WRD))

hero_contacts <- c("(wolverine@xmen.com)","wonderwoman@justiceleague.org","thor@avengers.com")

# Check match hasn't changed
str_view(hero_contacts, pattern = email)

# Pull out match and captures
email_parts <- str_match(hero_contacts, pattern = email)
email_parts

# Save host
host <- email_parts[,3]
host

# Validaovanie emailovej adresy vsak moze byt uplne v pici:
"https://stackoverflow.com/questions/201323/how-to-validate-an-email-address-using-a-regular-expression/201378#201378"


### 2) vo phone numbers:
contact <- c("Call me at 555-555-0191", "123 Main St","(555) 555 0191", "Phone: 555.555.0191 Mobile: 555.555.0192")
separator <- "(?:[-.()]|\\s)"
three_digits <- rebus::dgt(3)
four_digits <- rebus::dgt(4)

phone_pattern <- three_digits %R% zero_or_more(separator) %R% 
  three_digits %R% zero_or_more(separator) %R%
  four_digits


# Add capture() to get digit parts
phone_pattern <- capture(three_digits) %R% zero_or_more(separator) %R% 
  capture(three_digits) %R% zero_or_more(separator) %R%
  capture(four_digits)

# Pull out the parts with str_match()
phone_numbers <- str_match(contact, pattern = phone_pattern)

# Put them back together
str_c("(",phone_numbers[,2],") ",phone_numbers[,3], "-",phone_numbers[,4]) 

# V poslednom contacte sa nachadzaju dve telefonne cisla, ak chceme matchnut obe tak str_match_all 

" 
Great job! If you wanted to extract beyond the first phone number, e.g. 
The second phone number in the last string, you could use str_match_all(). 
But, like str_split() it will return a list with one component for each input 
string, and you'll need to use lapply() to handle the result.
" 
temp <- str_match_all(contact,pattern = phone_pattern)

####

# narratives has been pre-defined
narratives

# Add capture() to get age, unit and sex
pattern <- capture(optional(DGT) %R% DGT) %R%  
  optional(SPC) %R% capture(or("YO", "YR", "MO")) %R%
  optional(SPC) %R% capture(or("M", "F"))

# Pull out from narratives
str_match(narratives, pattern = pattern)

###########
# Edit to capture just Y and M in units
pattern2 <- capture(optional(DGT) %R% DGT) %R%  
  optional(SPC) %R% capture(or("Y", "M")) %R% optional(or("O","R")) %R%
  optional(SPC) %R% capture(or("M", "F"))

# Check pattern
str_view(narratives, pattern2)

# Pull out pieces
str_match(narratives, pattern2)

#######
## Backreferences
######

# Ak by sme hladali ten isty match v regexe viackrat, mozeme sa odvolavat na jednotlive captures cez REF1 az REF9 (\1 az \9) ) 
# Pouziva sa to teda najma na hladanie duplicitnych casti
rebus::REF1  # <regex> \1 


str_view("Paris in the the spring",SPC %R% capture(one_or_more(WRD)) %R% SPC %R% REF1)
str_view("Paris in the the the spring",SPC %R% capture(one_or_more(WRD)) %R% SPC %R% REF1) #len prve dve "the" sa matchnu
str_view("Paris in the erashe spring",SPC %R% capture(one_or_more(WRD)) %R% SPC %R% REF1)  #nenaslo to iste co sa uz raz matchlo


#### Data set babynames

# Names with three repeated letters
repeated_three_times <- capture(LOWER) %R% REF1 %R% REF1   #LOWER = any lower case character
# Test it
str_view(boy_names, pattern = repeated_three_times, match = TRUE)


# Names with a pair of repeated letters
pair_of_repeated <- capture(LOWER %R% LOWER) %R% REF1
# Test it
str_view(boy_names, pattern = pair_of_repeated, match = TRUE)

# Names with a pair that reverses
pair_that_reverses <- capture(LOWER) %R% capture(LOWER) %R% REF2 %R% REF1
# Test it
str_view(boy_names, pattern = pair_that_reverses, match = TRUE)


# Four letter palindrome names  (add exactly)
four_letter_palindrome <- exactly(capture(LOWER) %R% capture(LOWER) %R% REF2 %R% REF1)   #<regex> ^([:lower:])([:lower:])\2\1$
# Test it
str_view(boy_names, pattern = four_letter_palindrome, match = TRUE)

#####
### STR_REPLACE
####

# View text containing phone numbers
contact

# Replace digits with "X"
str_replace(contact, pattern = DGT, replacement = "X")

# Replace all digits with "X"
str_replace_all(contact, DGT, replacement = "X")

# Replace all digits with different symbol
str_replace_all(contact, DGT, c("X", ".", "*", "_"))  # pre vsetky 4 riadky sa pouziju ine replacement znaky


str_view_all("ahoING adad sa saf ING sfINGdsa", "ING" %R% zero_or_more(SPC))
str_view(narratives, pattern)

######
# Build pattern to match words ending in "ING"
pattern <- one_or_more(WRD) %R% "ING"   ## wrd(1, Inf) --- k tomuto mam vyhrady, nakolko to nemusia byt len slova konciace na ING ....
str_view(narratives, pattern)

# Test replacement
str_replace(narratives, capture(pattern), 
            str_c("CARELESSLY", REF1, sep = " "))

# One adverb per narrative
adverbs_10 <- sample(adverbs, 10)

# Replace "***ing" with "adverb ***ly"
str_replace(narratives, 
            capture(pattern),
            str_c(adverbs_10, REF1, sep = " "))  


# Protipriklad na ten konciaci ING:
str_view("faINGsd",one_or_more(WRD) %R% "ING")

###########################
### UNICODE
#########################

# a = 61
# mu = 3BC

# Unicode sa pise cez \u + up to 4 characters, alebo \U + up to 8 characters

# Rozne variacie pre mu (znak strednej hodnoty). 0 na zaciatku nezmenia unicod. Ak teda mame len 2 miesty, je fajn pouzit\u00XY alebo \U000000XY
"\u03BC"   
"\u3BC"
"\U000003BC"
"\U3BC"
writeLines("\u3BC")
writeLines("\U0001F44F")  # toto by malo dat nejakehy emotikon, avsak windows to nespracuje

# Ako zistit hexadecimalnu hodnotu pre nejaky znak
as.hexmode(utf8ToInt("a"))

## Vyraz pre normalizovane normalne rozdelenie 
x <- "Normal(\u03BC = 0, \u03C3 = 1)"  
x
stringr::str_view(x, pattern = "\u03BC")

## Matching unicode:
library(rebus)
library(stringr)
library(stringi)

str_view_all(x, pattern = greek_and_coptic())
#... regex:   \p{name}


x <- c("\u00e8", "\u0065\u0300")
writeLines(x)

as.hexmode(utf8ToInt(stri_trans_nfd("\u00e8")))
as.hexmode(utf8ToInt(stri_trans_nfc("\u0065\u0300")))

### Cviko:

# Names with builtin accents
(tay_son_builtin <- c(
  "Nguy\u1ec5n Nh\u1ea1c", 
  "Nguy\u1ec5n Hu\u1ec7",
  "Nguy\u1ec5n Quang To\u1ea3n"
))

# Convert to separate accents
tay_son_separate <- stri_trans_nfd(tay_son_builtin)

# Verify that the string prints the same
tay_son_separate

# Match all accents
str_view_all(tay_son_separate, pattern = UP_DIACRITIC)   # <regex> \p{DIACRITIC}


#####
x <- c("Adele", "Ad\u00e8le", "Ad\u0065\u0300le")
writeLines(x) # vyzeraju podobne
str_view(x, "Ad" %R% ANY_CHAR %R% "le")   # iba dve matchne, kedze v druhom je é reprezentovane 2 charactermi
str_view(x, "Ad" %R% GRAPHEME %R% "le")   # riesenie  ... <regex> \X

# tay_son_separate has been pre-defined
tay_son_separate

# View all the characters in tay_son_separate
str_view_all(tay_son_separate, ANY_CHAR)

# View all the graphemes in tay_son_separate
str_view(tay_son_separate, pattern = GRAPHEME)

####
# Combine the diacritics with their letters
tay_son_builtin <- stri_trans_nfc(tay_son_separate)
tay_son_builtin

# View all the graphemes in tay_son_builtin
str_view_all(tay_son_builtin, pattern = GRAPHEME)

########
## CASE STUDY 
########


#earnest_file <- stri_read_lines(earnest_file)  # earnest_file je "path". Tato funkcia je rychlejsia ako klasicka readLines
earnest <- readLines("http://s3.amazonaws.com/assets.datacamp.com/production/course_2922/datasets/importance-of-being-earnest.txt")

# Detect start and end lines
start <- str_which(earnest, "START OF THE PROJECT")
end <- str_which(earnest, "END OF THE PROJECT")

# Get rid of gutenberg intro text
earnest_sub  <- earnest[(start+1):(end-1)]

# Detect first act
lines_start <- str_which(earnest_sub,"FIRST ACT")

# Set up index
intro_line_index <- 1:(lines_start - 1)

# Split play into intro and play
intro_text <- earnest_sub[intro_line_index]
play_text <- earnest_sub[-intro_line_index]

# Take a look at the first 20 lines
writeLines(play_text)

#######
# Odstranenie prazdnych riadkov:

is_empty <- stri_isempty(play_text)
play_lines <- play_text[!is_empty]
#writeLines(play_lines)
play_lines[1:20]

# Pattern for start, word then .
#is_empty <- stri_isempty(play_text) # existuje aj takato funckia ktora vrati logical 
pattern_1 <- START %R% one_or_more(WRD) %R% DOT

# Test pattern_1
str_view(play_lines, pattern_1, match = TRUE)   # nie je to spravny pattern..vyskytuju sa tam aj ine slova
str_view(play_lines, pattern_1, match = FALSE)

play_lines[10:15]  # vidno ze prve 4 riadky patria Algernon-ovi

####
## Chceme identifikovat pocet "riadkov" pre jednotlive postavy 

# Pattern for start, capital, word then .
pattern_2 <- START %R% ascii_upper() %R% one_or_more(WRD) %R% DOT  # <regex> ^[A-Z][\w]+\.


# Test pattern_2
str_view(play_lines, pattern_2, match = TRUE)
str_view(play_lines, pattern_2, match = FALSE)

# Get subset of lines that match
lines <- str_subset(play_lines,pattern = pattern_2)

# Extract match from lines
who <- str_extract(lines, pattern = pattern_2)

# Let's see what we have
unique(who)   # nie je to stale 100% ... zaradili sme sem MR., July. University a nezahruli Lady Bracknell a podobne..

"
So, try specifically looking for lines that start with their names. 
You'll find the or1() function from the rebus package helpful.
It specifies alternatives but rather than each alternative being 
an argument like in or(), you can pass in a vector of alternatives.
"

or("Lily","Harry")       # neberie vektor
or1(c("Lily","Harry"))   # berie vektor

# Create vector of characters
characters <- c("Algernon", "Jack", "Lane", "Cecily", "Gwendolen", "Chasuble", 
                "Merriman", "Lady Bracknell", "Miss Prism")

# Match start, then character name, then .
pattern_3 <- START %R% or1(characters) %R% DOT

# View matches of pattern_3
str_view(play_lines, pattern = pattern_3, match = TRUE)

# View non-matches of pattern_3
str_view(play_lines, pattern = pattern_3, match = FALSE)

# Pull out matches
lines <- str_subset(play_lines, pattern = pattern_3)

# Extract match from lines
who <- str_extract(lines, pattern = pattern_3)

# Let's see what we have
unique(who)

# Count lines per character
table(who)

## jedine co sme vynechali boli riadky, ktore hovorili spolu: Jack and Algernon [Speaking together.]
"One solution might be to look for these 'Speaking together' lines, parse out the characters, and add to your counts."

#####
## Ignorovanie case sensitive
#####

x <- c("Cat", "CAT", "cAt") 
str_view(x, "cat")
str_view(x, "Cat")
str_view(str_to_lower(x), "cat")   #pri html zobrazeni zmneni povodny vektor x (vsetko da na male)
str_view(x, pattern = regex("cat", ignore_case = TRUE))


catcidents <- 
  c("79yOf Fractured fingeR tRiPPED ovER cAT ANd fell to FlOOr lAst nIGHT AT HOME*", 
    "21 YOF REPORTS SUS LACERATION OF HER LEFT HAND WHEN SHE WAS OPENING A CAN OF CAT FOOD JUST PTA. DX HAND LACERATION%", 
    "87YOF TRIPPED OVER CAT, HIT LEG ON STEP. DX LOWER LEG CONTUSION ", 
    "bLUNT CHest trAUma, R/o RIb fX, R/O CartiLAgE InJ To RIB cAge; 32YOM walKiNG DOG, dog took OfF aFtER cAt,FelL,stRucK CHest oN STepS,hiT rIbS", 
    "42YOF TO ER FOR BACK PAIN AFTER PUTTING DOWN SOME CAT LITTER DX: BACK PAIN, SCIATICA", 
    "4YOf DOg jUst hAd PUpPieS, Cat TRIED 2 get PuPpIes, pT THru CaT dwn stA Irs, LoST foOTING & FELl down ~12 stePS; MInor hEaD iNJuRY", 
    "unhelmeted 14yof riding her bike with her dog when she saw a cat and sw erved c/o head/shoulder/elbow pain.dx: minor head injury,left shoulder", 
    "24Yof lifting a 40 pound bag of cat litter injured lt wrist; wrist sprain", 
    "3Yof-foot lac-cut on cat food can-@ home ", "Rt Shoulder Strain.26Yof Was Walking Dog On Leash And Dot Saw A Cat And Pulled Leash.", 
    "15 mO m cut FinGer ON cAT FoOd CAn LID. Dx:  r INDeX laC 1 cm.", 
    "31 YOM SUSTAINED A CONTUSION OF A HAND BY TRIPPING ON CAT & FALLING ON STAIRS.", 
    "ACCIDENTALLY CUT FINGER WHILE OPENING A CAT FOOD CAN, +BLEEDING >>LAC", 
    "4 Yom was cut on cat food can. Dx:  r index lac 1 cm.", "4 YO F, C/O FOREIGN BODY IN NOSE 1/2 HOUR, PT NOT REPORTING NATURE OF F B, PIECE OF CAT LITTER REMOVED FROM RT NOSTRIL, DX FB NOSE", 
    "21Yowf  pT STAteS 4-5 DaYs Ago LifTEd 2 - 50 lB BagS OF CAT lItter.  al So sORTIng ClOThES & W/ seVERe paIn.  DX .  sTrain  LUMbOSaCRal.", 
    "67 YO F WENT TO WALK DOG, IT STARTED TO CHASE CAT JERKED LEASH PULLED H ER OFF PATIO, FELL HURT ANKLES. DX BILATERAL ANKLE FRACTURES", 
    "17Yof Cut Right Hand On A Cat Food Can - Laceration ", "46yof taking dog outside, dog bent her fingers back on a door. dog jerk ed when saw cat. hand holding leash caught on door jamb/ct hand", 
    "19 YOF-FelL whIle WALKINg DOWn THE sTAIrS & TRiPpEd over a caT-fell oNT o \"TaIlBoNe\"         dx   coNtusIon LUMBaR, uti      *", 
    "50YOF CUT FINGER ON CAT FOOD CAN LID.  DX: LT RING FINGER LAC ", 
    "lEFT KNEE cOntusioN.78YOf triPPEd OVEr CaT aND fell and hIt knEE ON the fLoOr.", 
    "LaC FInGer oN a meTAL Cat fOOd CaN ", "PUSHING HER UTD WITH SHOTS DOG AWAY FROM THE CAT'S BOWL&BITTEN TO FINGE R>>PW/DOG BITE", 
    "DX CALF STRAIN R CALF: 15YOF R CALF PN AFTER FALL ON CARPETED STEPS, TR YING TO STEP OVER CAT, TRIPPED ON STAIRS, HIT LEG", 
    "DISLOCATION TOE - 80 YO FEMALE REPORTS SHE FELL AT HOME - TRIPPED OVER THE CAT LITTER BOX & FELL STRIKING TOE ON DOOR JAMB - ALSO SHOULDER INJ", 
    "73YOF-RADIUS FX-TRIPPED OVER CAT LITTER BOX-FELL-@ HOME ", "57Yom-Back Pain-Tripped Over A Cat-Fell Down 4 Steps-@ Home ", 
    "76YOF SUSTAINED A HAND ABRASION CLEANING OUT CAT LITTER BOX THREE DAYS AGO AND NOW THE ABRASION IS INFECTED CELLULITIS HAND", 
    "DX R SH PN: 27YOF W/ R SH PN X 5D. STATES WAS YANK' BY HER DOG ON LEASH W DOG RAN AFTER CAT; WORSE' PN SINCE. FULL ROM BUT VERY PAINFUL TO MOVE", 
    "35Yof FeLt POp iN aBdoMeN whIlE piCKInG UP 40Lb BaG OF CAt litTeR aBdomINAL sTrain", 
    "77 Y/o f tripped over cat-c/o shoulder and upper arm pain. Fell to floo r at home. Dx proximal humerus fx", 
    "FOREHEAD LAC.46YOM TRIPPED OVER CAT AND FELL INTO A DOOR FRAME. ", 
    "39Yof dog pulled her down the stairs while chasing a cat dx: rt ankle inj", 
    "10 YO FEMALE OPENING A CAN OF CAT FOOD.  DX HAND LACERATION ", 
    "44Yof Walking Dog And The Dof Took Off After A Cat And Pulled Pt Down B Y The Leash Strained Neck", 
    "46Yof has low back pain after lifting heavy bag of cat litter lumbar spine sprain", 
    "62 yOf FELL PUShIng carT W/CAT liTtER 3 DAYs Ago. Dx:  l FIfTH rib conT.", 
    "PT OPENING HER REFRIGERATOR AND TRIPPED OVER A CAT AND FELL ONTO SHOULD ER FRACTURED HUMERUS", 
    "Pt Lifted Bag Of Cat Food. Dx:  Low Back Px, Hx Arthritic Spine."
  )


# Construct pattern of DOG in boundaries
whole_dog_pattern <- whole_word("DOG")

# Rozdiel medzi whole_word("DOG") a exactly("DOG") je taka, ze ak mame "I am a DOG" tak exactly to nematchne, lebo by DOG musel byt samostatny string

# See matches to word DOG
str_view(catcidents, pattern = whole_dog_pattern, match = TRUE)  # iba match vidime
#str_view(catcidents, pattern = whole_dog_pattern, match = FALSE)
#str_view(catcidents, pattern = whole_dog_pattern, match = NA)

# Transform catcidents to upper case
catcidents_upper <- str_to_upper(catcidents)

# View matches to word "DOG" again
str_view(catcidents_upper, pattern = whole_dog_pattern, match = TRUE)

# Ak by sme si chceli ponechat povodne zmiesane upper/lower case znaky, da sa to vyriesit napr. takto:

# Which strings match?
has_dog <- str_detect(catcidents_upper, pattern = whole_dog_pattern)

# Pull out matching strings in original 
catcidents[has_dog]

#####
# cez regex ignor_case

# View matches to "TRIP"
str_view(catcidents, pattern = "TRIP", match = TRUE)

# Construct case insensitive pattern
trip_pattern <- regex("TRIP", ignor_case = TRUE)

# View case insensitive matches to "TRIP"
str_view(catcidents, pattern = trip_pattern, match = TRUE)

####
# Get subset of matches
trip <- str_subset(catcidents, pattern = trip_pattern)

# Extract matches
str_extract(trip, pattern = trip_pattern)


####
# String to Title
####

# Stringr ponuka funkciu str_to_title() --> kazde slovo zacina na velke pismeno
# Stringi ponuka funkciu stri_trans_totitle() --> je to rovnaka funkcia, avsak flexibilnejsia, 
#                               nakolko mozeme pridat argument type = "sentence" a velke pismeno bude v ramci vety/celeho stringu 


library(stringi)

# Get first five catcidents
cat5 <- catcidents[1:5]

# Take a look at original
writeLines(cat5)

# Transform to title case
writeLines(str_to_title(cat5))

# Transform to title case with stringi
stri_trans_totitle(cat5)

# Transform to sentence case with stringi
writeLines(stri_trans_totitle(cat5, type = "sentence"))


stri_trans_totitle("asdj.saddsa . dasjoi", type = "sentence")


#######
## ZHRNUTIE
#######

# Stringr:  https://www.rdocumentation.org/packages/stringr/versions/1.3.1
# Stringr je vybudovana na stringi package, takze ak nieco nenajdeme v stringr, moze sa nachadzat v stringi 
# Regular expressions: https://www.regular-expressions.info/ 
#: mastering regular expression by Jeffrey Friedl
#  R4DS : https://r4ds.had.co.nz/strings.html#matching-patterns-with-regular-expressions


######################################################
######################################################

'https://users.cs.cf.ac.uk/Dave.Marshall/PERL/node79.html'
'https://www.regular-expressions.info/charclass.html'

[^0-9] # any single non-digit
[^aeiouAEIOU] # any single non-vowel

*?     Match 0 or more times
+?     Match 1 or more times
??     Match 0 or 1 time
{n}?   Match exactly n times
{n,}?  Match at least n times
{n,m}? Match at least n but not more than m times

fa*t matches to ft, fat, faat, faaat etc

.*) can be used a wild card match for any number (zero or more) of any characters.
Thus f.*k matches to fk, fak, fork, flunk, etc.


## Skusanie..
x <- c("aaa","bbbb"," ","")
str_detect(x, char_class("ab"))
str_detect(x, pattern = "[ab]")

str_detect(x, pattern = "ab")
str_detect(x, pattern = "[.b]")
str_detect(x, pattern = "[.aaa]")   # bodka moze predstavovat aj nula vyskytov
str_detect(x, pattern = ".")  







na_omit_nove <- function(data)  data[rowSums(is.na(data)) == 0, ]






