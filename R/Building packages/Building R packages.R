# You can use devtools to create the basic structure of an R package by using the create() function.

# There are some optional arguments to this function but the main one that you will use is the path argument.
# You use this to specify where your package will be created and the name that your package will take.

# If you want to create the package in your current working directory, as you often will, 
# you just need to supply the name for the package. When naming your package remember to think about:
  
      # If the name is already taken by another package.
      # Whether the name makes it clear what the package does.

#Devtools is loaded in your workspace.
library(devtools)

# Use the create function to set up your first package
create("datasummary")   # nazov balika

# Take a look at the files and folders in your package
dir("datasummary")   

##################

# As a minimum there are only two directories that must be included 
# along with two additional files --- he DESCRIPTION and the NAMESPACE.

#All the code for your package goes in the R directory!

##################

# Tvorba funkcie:

# Create numeric_summary() function
numeric_summary <- function(x, na.rm) {
  
  # Include an error if x is not numeric
  if(!is.numeric(x)){
    stop("Data must be numeric")
  }
  
  # Create data frame
  data.frame( min = min(x, na.rm = na.rm),
              median = median(x, na.rm = na.rm),
              sd = sd(x, na.rm = na.rm),
              max = max(x, na.rm = na.rm))
}

# Test numeric_summary() function
numeric_summary(airquality$Ozone, na.rm = TRUE)

###################

# What is in the R directory before adding a function?
dir("datasummary/R")  # Zatial nie je .R (project) naplneny ziadnym skriptom

# Use the dump() function to write the numeric_summary function
dump("numeric_summary", file = "datasummary/R/numeric_summary.R")  # tu by mohlo byt ak dobre rozumiem: dump(list s nazvami funkcii a na konci funkcie.R)

# naplnili sme object numeric_summary ako samostatnu funkciu do path:"datasummary/R/numeric_summary.R"

# Verify that the file is in the correct directory
dir("datasummary/R")

###################

#Role : 
    # cre is used to mark the maintainer of the package (who should get an email if something goes wrong).
    # aut je autor
    # cph je copyright holder
    # ctb je contributor

# Okrem .Rproj, DESCRIPTION, NAMESPACE, R, mozeme pridat aj ine priecinky: Data, Vignettes, Tests, Compiled Code (C++), Translations, Demos


###################
# Pridanie dat a vignetov

# What is in the package at the moment?
dir("datasummary")

weather <- tibble::tibble(Day = 1:7, Temp = sample(14:20,size = 7))

# Add the weather data
use_data(weather, pkg = "datasummary", overwrite = TRUE)



# Add a vignette called "Generating Summaries with Data Summary"
use_vignette("Generating_Summaries_with_Data_Summary", pkg = "datasummary") # Note that when adding vignettes, it's best not to include any spaces in the vignette name.
  

# What directories do you now have in your package now?
dir("datasummary")

###################
# Pridanie dalsej funkcie

data_summary <- function(x, na.rm = TRUE){
  
  num_data <- select_if(x, .predicate = is.numeric) 
  
  map_df(num_data, .f = numeric_summary, na.rm = TRUE, .id = "ID")
  
}

# Write the function to the R directory
dump("data_summary", file = "datasummary/R/data_summary.R")

# Function dir
dir("datasummary/R")


####################################
### ROXYGEN - dokumentacia k funkcii ide tam kde je Rskript vo funkcii
##############

# Template vyzera takto: Prve 3 riadky su specialne medzi ktorymi treba "prazdne" riadky zacinajuce #'
# Komentovanie je povoene len jednoduchym #. 
# Dokumentacia argumentov funkcie je cez @param variable "variable description"
# Ak chceme pouzit funkciu z ineho balika, tak treba @import dplyt, alebo ak len jednu funkciu tak @importFrom tidyr gather


#' Title goes here
#'
#' Description goes here
#'
#' Details go here
#' @param var "var description"

#### Napríklad:

#'  Numeric Summaries
#'
#' Summarises numeric data and returns a data frame containing the minimum value, median, standard deviation, and maximum value.
#' 
#' Add appropriate tag and details to document the first argument
#' @param x "a numeric vector containing the values to summarize."
#' @param na.rm A logical indicating whether missing values should be removed
#'
#' @author Nikolas Markus <markusfmfi@gmail.com>
#'
#' @import dplyr
#' @import purrr
#' @importFrom tidyr gather
#' 
#' @export
#'
#' @examples
#' data_summary(iris)
#' data_summary(airquality, na.rm = FALSE)
#' 
## Update the details for the return value
#' @return 
#' \itemize{
#'  \item This function returns a \code{data.frame} including columns:".
#'  \item ID
#'  \item min
#'  \item median
#'  \item sd
#'  \item max
#' }
#' @seealso \link[base]{summary}
data_summary <- function(x, na.rm = TRUE){
  
  num_data <- select_if(x, .predicate = is.numeric) 
  
  map_df(num_data, .f = numeric_summary, na.rm = na.rm, .id = "ID")
  
}
# 
# numeric_summary <- function(x, na.rm){
#   
#   if(!is.numeric(x)){
#     stop("Data must be numeric")
#   }
#   
#   data.frame( min = min(x, na.rm = na.rm),
#               median = median(x, na.rm = na.rm),
#               sd = sd(x, na.rm = na.rm),
#               max = max(x, na.rm = na.rm))
# }


#####################################
## EXPORTED vs NON-EXPORTED functions
#####

# Exportovana funkcia je taka, ktoru moze pouzivat end-user. Neexportovana je len kvazi pomocna funkcia
# Ak chceme aby bola funkcia exportovana, pridame do roygen header @export
# K neexportovanej funkcii pristupujeme pomocou ::: (3 krat)... package:::function

####################################
##  Examples by mali byt jednoduche a spustitelne bez erroru
##  Ak nechceme aby sa spustilo, pouzijeme @examples \n #' \dontrun{ function(...) }

####################################
## You document the return value of a function using the tag @return. This is where you can tell 
## users what they can expect to get from the function, be that data, a graphic or any other output.
## For code format (font) you use \code{text to format}
## To link to other functions you use \link[packageName]{functionName}, although note the package name 
##            is only required if the function is not in your package
## To include an unordered list you use \itemize{}. Inside the brackets you mark new items 
##            with \item followed by the item text


##################################
###### DOKUMENTACIA BALIKA #######
##################################

#### Ked to dokumentujeme v takejto podobe, tak to ukladame do /R file s rovnakym nazvom ako je package..tj: package.R

#' datasummary: Custom Data Summaries
#'
#' Easily generate custom data frame summaries
#'
#' @author Nikolas Markus \email{markusfmfi@gmail.com}
#' @docType package
#' @name datasummary
"_PACKAGE"


##################################
##### DOKUMENTACIA DATASETU ######
##################################

#' Random Weather Data
#'
#' A dataset containing randomly generated weather data.
#'
#' @format A data frame of 7 rows and 3 columns
#' \describe{
#'  \item{Day}{Numeric values giving day of the week, 1 = Monday, 7 = Sunday}
#'  \item{Temp}{Numeric values giving temperature in Celsius}
#'  \item{Weather}{Categorical values giving the type of Weather}
#' }
#' @source Randomly generated data
"weather"

#################################
## GENEROVANIE DOKUMENTACII
##########

# Generate package documentation
devtools::document("datasummary")   # ulozi do man priecinka

# Examine the contents of the man directory
dir("datasummary/man")

# View the documentation for the data_summary function
help("data_summary")

# View the documentation for the weather dataset
help("weather")

###############################
## KONTROLA BALIKA
###########

# Check your package
check("datasummary")   # kontroluje cca 50 veci, ci je vsetko spravnenastavene, zdokumentovane atd.. nekontroluje spravnost pouzitelnosti kodu


#### Errors, warnings, notes

# Ak zabudneme zdokumentovat nejake vstupne premenne funkcie, dostaneme WARNING (zbehne to.., lebo nie ERROR):
#     Undocumented arguments in documentation object 'numeric_summary','na.rm'
#     Note: you wouldn't normally get this error for non-exported functions

# To remove this warning, you'll need to update the documentation for the parameter in the function's .R file,
# and then run check() again. You might think you need to run the document() function again. 
# However, there's no need to do this, as check() automatically runs document() for you before completing its checks.



#### Problem s globalnymi premennymi - nerozumiem uplne, ale niekedy sa nazov tabulky a stlpcov neulozi ako globalna premenna a vyhadzuje error


## The way in which you define variables in tidyverse package functions can cause confusion for the R CMD check, 
## which sees column names and the name of your dataset, and flags them as "undefined global variables".
## To get around this, you can manually specify the data and its columns as a vector to utils::globalVariables(),
## by including a line of code  similar to the following in your package-level documentation:
##   utils::globalVariables(c("dataset_name", "col_name_1", "col_name_2"))

## Roxygen:

#' datasummary: Custom Data Summaries
#'
#' Easily generate custom data frame summaries
#'
#' @docType package
#' @name datasummary
"_PACKAGE"

#get_mean_temp <- function() summarize(weather, meanTemp = mean(Temp)  ## Toto by bola funkcia napriklad 

# Update this function call
utils::globalVariables(c("weather","Temp"))


#########
## DEPENDENCIES
####

search() 

# Adding a dependency:
use_package("dplyr") ##adds to imports
use_package("ggplot2", "suggests") ## adds to suggests (to je jeden segment v dokumentáci)

  # 
  # The Depends and Imports fields in the DESCRIPTION file can cause a lot of confusion to those new to package 
  # building! Both of these fields contain package dependencies which are installed when you install the package. 
  # However, the difference between them is that the packages listed in depends are attached when the package 
  # is loaded, whereas the packages listed in imports are not.
  # 
  # This distinction is important because if two different packages have an identically named function, the version
  # of the function from the package which has been loaded most recently will be used. Therefore, to make sure you 
  # are using the correct function, it is best practice to use imports to list your dependencies and then in your 
  # code explicitly name the package and function its from in the form package::function(), e.g. dplyr::select().

# Add dplyr as an imported dependency to the DESCRIPTION file
use_package("dplyr", pkg = "datasummary")

# Add purrr as an imported dependency to the DESCRIPTION file
use_package("purrr", pkg = "datasummary")

# Add tidyr as an imported dependency to the DESCRIPTION file
use_package("tidyr", pkg = "datasummary")


### Building packages with continuous integration

devtools::build("datasummary")  # build the source -- .tar.gz
devtools::build("datasummary", binary = TRUE)  # build the binary .. spominali vyuzitie pri pre-compile kodoch s inymyi jazykmi

# COntinuous integration: automaticky spusti chceck ked sa kod zmeni, verziovanie, 
use_travis("datasummary")   # TRAVIS CI je na to, aby sa automaticky pri zmene kodu robil check
use_github()   # dobre na verziovanie

### Cviko:

# Build the package
build("datasummary")
# Examine the contents of the current directory
dir("datasummary")


######################
#### UNIT TESTS 
######################

## pisu sa testy do zvlast .R skriptu 
expect_identical  # checkuje exact equality
expect_equal      # check equality with numerical tolerance
expect_equivalent # more relaxed version of equals
expect_error      # chcecks that an expression throws an error
expect_warning    # checks that an expression gives a warning
expect_output     # checks that output matches a specified value


library(testthat)
expect_identical()  # najstriktnejsie porovnanie. Porovnava values, attributes a aj type

myvector <- c("First" = 1, "Second" = 2)

expect_identical(myvector, c("First" = 1, "Second" = 2))  # nevrati nic
expect_identical(myvector, c(1,2))                        # vrati chybu
expect_identical(myvector,c("First" = 1L, "Second" = 2L)) # vrati chybu

##
expect_equal()     # porovnava iba value a attribute ale NIE type
expect_equal(myvector, c(1,2))                            # vrati chybu, lebo attribute nie je rovnaky 
expect_equal(myvector,c("First" = 1L, "Second" = 2L))     # prejde testom, aj ked nemaju rovnaky type

##
expect_equal(myvector,c("First" = 1.2, "Second" = 2.1))   # vypise priemernu odchylku numericku
expect_equal(myvector,c("First" = 1.2, "Second" = 2.1), tolerance = 0.15)   # priemerna tolerancia ja mensia rovna, cize presiel testom

##
expect_equivalent  # je najmenej striktna, pozera sa iba na value 
expect_equivalent(myvector, c(1,2))                       # vrati chybu, lebo attribute nie je rovnaky 

### Priprava frameworku:

# Set up the test framework
use_testthat("datasummary")  # vytvori script testthat.R a priecinok tests/testthat

# Look at the contents of the package root directory
dir("datasummary")

# Look at the contents of the new folder which has been created 
dir("datasummary/tests")


# You save your tests in the tests/testthat/ directory in files with filenames 
# beginning with test-. So, for example, the simutils package has tests named:
# 
#   test-na_counter.R
#   test-sample_from_data.R

### Mini test num. 1

# Create a summary of the iris dataset using your data_summary() function
iris_summary <- data_summary(iris)

# Count how many rows are returned
summary_rows <- nrow(iris_summary)

# Use expect_equal to test that calling data_summary() on iris returns 4 rows
expect_equal(summary_rows, nrow(data_summary(iris)))

result <- data_summary(weather)

# Update this test so it passes
expect_equal(result$sd, c(2.1, 3.6), tolerance = 0.1)

### Mini test num. 2

result <- data_summary(weather)

expected_result <- list(
  ID = c("Day", "Temp"),
  min = c(1L, 14L),
  median = c(4L, 19L),
  sd = c(2.16024689946929, 3.65148371670111),
  max = c(7L, 24L)
)

# Write a passing test that compares expected_result to result
expect_equivalent(result, expected_result)  # preslo, aj ked jedno je list a druhe df


### Testing error / warning

sqrt(-1)      # warning
sqrt("foo")   # error

expect_warning(sqrt(-1))
expect_error(sqrt("foo"))


### Testing non-exported functions

expected <- data.frame(min = 14L, median = 19L, sd = 3.65148371670111, max = 24L)

# Create variable result by calling numeric summary on the temp column of the weather dataset
result <- datasummary:::numeric_summary(weather$Temp, na.rm = TRUE)

# Test that the value returned matches the expected value
expect_equal(result, expected)

### Grouping Test and Execution Output

# Use context() and test_that() to group the tests below together
context("Test data_summary()")

test_that("data_summary() handles errors correctly", {
  
  # Create a vector
  my_vector <- 1:10
  
  # Use expect_error()
  expect_error(data_summary(my_vector))
  
  # Use expect_warning()
  expect_warning(data_summary(airquality, na.rm = FALSE))
  
})

#### Testovanie: 

# Run the tests on the datasummary package
devtools::test("datasummary") 

# This function looks for all tests located in the tests/testhat or inst/tests directory 
# with filenames beginning with test- and ending in .R, and executes each of them. As with 
# the other devtools functions, you supply the path to the package as the first argument to the test() function.


