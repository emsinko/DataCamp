#################################
### INTRODUCTION TO NLP IN R  ###
#################################

# https://www.tidytextmining.com/dtm.html
# https://www.kaggle.com/kernels/scriptcontent/25728798/download

tidytext::get_stopwords(language = "svk", source = "snowball")
stopwords::stopwords
####################################
## CHAPTER 1 - TRUE FUNDAMENTALS  ##
####################################

text <- c("John's favorite color two colors are blue and red.", 
          "John's favorite number is 1111.", 
          "John lives at P Sherman, 42 Wallaby Way, Sydney", 
          "He is 7 feet tall", 
          "John has visited 30 countries", 
          "John only has nine fingers.", 
          "John has worked at eleven different jobs", 
          "He can speak 3 languages", 
          "john's favorite food is pizza", 
          "John can name 10 facts about himself."
)
## Regular expression basics

# Print off each item that contained a numeric number
grep(pattern = "\\d", x = text, value = TRUE) # value = FALSE --> indexes

# Find all items with a number followed by a space
grep(pattern = "\\d\\s", x = text)

# How many times did you write down 'favorite'?
length(grep(pattern = "favorite", x = text))

# Print off the text for every time you used your boss's name, John
grep('John', x = text, value = TRUE)

# Try replacing all occurences of "John" with "He"
gsub(pattern = 'John', replacement = 'He', x = text)

# Replace all occurences of "John " with 'He '.
clean_text <- gsub(pattern = 'John\\s', replacement = 'He ', x = text)
clean_text

# Replace all occurences of "John's" with 'His'
gsub(pattern = "John\\'s", replacement = 'His', x = clean_text)

########
## Tokenization
########

install.packages("tidytext", dependencies = TRUE)
library(tidytext)
library(tidyverse)

# Types of tokenizations: charakters, words, sentences, documents,  regular expression separations
# Tidytext package --> "Text minig package using dplyr, ggplot2 and other tidy tools 

animal_farm <- read.csv("https://assets.datacamp.com/production/repositories/4966/datasets/a3968804c57e7178b6000e0499ac241728b3c73a/animal_farm.csv", stringsAsFactors = FALSE)
animal_farm %>% glimpse()

# Function for tokenization:  
animal_farm %>% 
  unnest_tokens(output = "word", input = text_column, token = "words") %>%  # token = c("sentences", "lines", "regex", "words")
  count(word, sort = TRUE)
 
## Sentence count 

# Split the text_column into sentences
animal_farm %>%
  unnest_tokens(output = "sentences", input = text_column, token = "sentences") %>%
  # Count sentences, per chapter
  count(chapter)

# Split the text_column using regular expressions
# Using a regular expression, split the text of Animal Farm into sentences whenever a period (\\.) is found.
animal_farm %>%
  unnest_tokens(output = "sentences", input = text_column,
                token = "regex", pattern = "\\.") %>%
  count(chapter)

# Great job. Notice how the two methods produce slightly different results. 
# You'll notice that a lot when processing text. 
# It's all about the technique used to do the analysis.

animal_farm %>% filter(chapter == "Chapter 1")

####
## Text preprocessing: remove stop words
####

# Stop words are unavoidable in writing. However, to determine how similar two pieces of text are 
# to each other are or when trying to find themes within text, stop words can make things difficult. 

# In the book Animal Farm, the first chapter contains only 2,636 words, while almost 200 of them are the word "the".

# Usually, "the" will not help us in text analysis projects. 
# In this exercise you will remove the stop words from the first chapter of Animal Farm.

# Tokenize animal farm's text_column column
tidy_animal_farm <- animal_farm %>%
  unnest_tokens(output = word, input = text_column, token = "words")  # defaultne je tokenizovane podla slova

# Print the word frequencies
tidy_animal_farm %>%
  count(word, sort = TRUE)

# Vidime, ze "the" = 2187, "and" = 966, "of" = 899 .... treba sa ich zbavit
# Tidytext ma slovnik stop wordov

tidytext::stop_words
stop_words %>% group_by(lexicon) %>% summarise(cnt_of_words = n())
stop_words %>% group_by(word) %>% summarise(in_how_many_lexicons = n())

# Pridanie vlastneho slova:
add_row(stop_words, word = "nove_slovo", lexicon = "custom") %>% tail()

# Remove stop words, using stop_words from tidytext
tidy_animal_farm <- 
  tidy_animal_farm %>%
    anti_join(stop_words)

# Excellent. You should always consider removing stop words before performing text analysis. 
# They muddy your results and can increase computation time for large analysis tasks.

####
## Text preprocessing: Stemming - koreò slova 
####

# The root of words are often more important than their endings, especially when it comes to text analysis. 
# The book Animal Farm is obviously about animals. 
# However, knowing that the book mentions animal's 248 times, and animal 107 times might not be helpful for your analysis.

# tidy_animal_farm contains a tibble of the words from Animal Farm, tokenized and without stop words. 
# The next step is to stem the words and explore the results

library(SnowballC)

SnowballC::wordStem()
wordStem(c("win", "winning", "winner", "win", "wins", "winners", "random"))

# Perform stemming on tidy_animal_farm
stemmed_animal_farm <- tidy_animal_farm %>%
  mutate(word = wordStem(word))

# Print the old word frequencies 
tidy_animal_farm%>%
  count(word, sort = TRUE)

# Print the new word frequencies
stemmed_animal_farm %>%
  count(word, sort = TRUE)


########################################
## CHAPTER 2 - REPRESENTATION OF TEXT ##
########################################

######
## Understanding an R corpus
#####

# Corpora = plural od slova Corpus
# Corpora = collection of documents containing natural language text
# From "tm" package as "corpus"
# VCorpus = most common representation

library(tm)
data("acq")

acq # struktura -> VCORPUS
str(acq)


acq[[1]]$meta         # metadata prveho dokumentu
acq[[1]]$meta$places  
acq[[1]]$content      # obsah

# Tidying a corpus

tidy_data <- tidy(acq)
tidy_data  # --> transformovalo $meta data do tibble. 

# Corpusing a tidy data

corpus <- VCorpus(VectorSource(tidy_data$text))
meta(corpus, "Author") <- tidy_data$author
meta(corpus, "oldid") <- tidy_data$oldid

head(meta(corpus))


####
## Explore an R corpus

# One of your coworkers has prepared a corpus of 20 documents discussing crude oil, named crude. 
# This is only a sample of several thousand articles you will receive next week. 
# In order to get ready for running text analysis on these documents, you have decided to explore their content and metadata. 
# Remember that in R, a VCorpus contains both meta and content regarding each text. 
# In this lesson, you will explore these two objects.

# Print out the corpus
print(crude)

# Print the content of the 10th article
crude[[10]]$content

# Find the first ID
crude[[1]]$meta$id

# Make a vector of IDs
ids <- c()
for(i in c(1:20)){
  ids <- append(ids, crude[[i]]$meta$id)
}

# Well done. You now understand the basics of an R corpus.
# However, creating the ID vector was a bit of work. 
# Let's use the tidy() function to help make this process easier.

###
## Creating a tibble from a corpus
###

#To further explore the corpus on crude oil data that you received from a coworker,
# you have decided to create a pipeline to clean the text contained in the documents. 
# Instead of exploring how to do this with the tm package, you have decided to transform the corpus 
# into a tibble so you can use the functions unnest_tokens(), count(), and anti_join() that you are already familiar with. 
# The corpus crude contains both the metadata and the text of each document.

# Create a tibble & Review
crude_tibble <- tidy(crude)
names(crude_tibble)

crude_counts <- crude_tibble %>%
  # Tokenize by word 
  unnest_tokens(output = word, input = text) %>%
  # Count by word
  count(word, sort = TRUE) %>%
  # Remove stop words
  anti_join(stop_words)


###
## Creating a corpus
###

russian_tweets <- read.csv("https://assets.datacamp.com/production/repositories/4966/datasets/125d845e6fe39bb0eb7799e0b91725f424510f16/russian_1.csv", stringsAsFactors = FALSE)
glimpse(russian_tweets)

# You have created a tibble called russian_tweets that contains around 20,000 tweets auto generated 
# by bots during the 2016 U.S. election cycle so that you can preform text analysis. 

# However, when searching through the available options for performing the analysis you have chosen to do, you 
#  believe that the tm package offers the easiest path forward. 

# In order to conduct the analysis, you first must create a corpus and attach potentially useful metadata.

# Be aware that this is real data from Twitter and as such there is always a risk that 
# it may contain profanity or other offensive content (in this exercise, and any following exercises that also use real Twitter data).

# Create a corpus
tweet_corpus <- VCorpus(VectorSource(russian_tweets$content))

# Attach following and followers
meta(tweet_corpus, 'following') <- russian_tweets$following
meta(tweet_corpus, 'followers') <- russian_tweets$followers

# Review the meta data
head(meta(tweet_corpus))

#### 
## The bag-of-words representation (BoW)

# Russian tweets: 20 000 tweets, 43 000 unique non stop words --> 20k * 43k = 860 000 000 value sparse matrix -> only 177000 non 0 entries

## Practice BoW
# Given the following texts
t1 <- c("Today will be an awesome day. The best day of the week")
t2 <- c("Yesterday was an awesome day. Better than today.")
t3 <- c("Tomorrow will be the best day. Better than yesterday and today.")

# You have created a word vector
word_vector <- c("today", "awesome", "day", "best", "week", "yesterday", "better", "tomorrow")

# Possible BoW vector representation of text t3: 1, 0, 1, 1, 0, 1, 1, 1

## BoW Example
# In literature reviews, researchers read and summarize as many available texts about a subject as possible. 
# Sometimes they end up reading duplicate articles, or summaries of articles they have already read. 
# You have been given 20 articles about crude oil as an R object named crude_tibble. 
# Instead of jumping straight to reading each article, you have decided to see what words are shared across these articles. 
# To do so, you will start by building a bag-of-words representation of the text.

# Vytvrorenie crude_tibble, aby sme vedeli pracovat s touto premennou. 
crude_tibble <- 
  structure(list(author = c(NA, "BY TED D'AFFLISIO, Reuters", NA, 
                            NA, NA, NA, "By Jeremy Clift, Reuters", NA, NA, NA, NA, NA, NA, 
                            NA, NA, NA, NA, NA, "By BERNICE NAPACH, Reuters", NA), datetimestamp = structure(c(541357256, 
                                                                                                               541359251, 541361880, 541362061, 541364457, 541567546, 541568354, 
                                                                                                               541574847, 541585350, 541621904, 541645549, 541669163, 541669402, 
                                                                                                               541669421, 541671942, 541682405, 541682906, 541685626, 541694314, 
                                                                                                               541694946), class = c("POSIXct", "POSIXt"), tzone = ""), description = c("", 
                                                                                                                                                                                        "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", 
                                                                                                                                                                                        "", "", ""), heading = c("DIAMOND SHAMROCK (DIA) CUTS CRUDE PRICES", 
                                                                                                                                                                                                                 "OPEC MAY HAVE TO MEET TO FIRM PRICES - ANALYSTS", "TEXACO CANADA <TXC> LOWERS CRUDE POSTINGS", 
                                                                                                                                                                                                                 "MARATHON PETROLEUM REDUCES CRUDE POSTINGS", "HOUSTON OIL <HO> RESERVES STUDY COMPLETED", 
                                                                                                                                                                                                                 "KUWAIT SAYS NO PLANS FOR EMERGENCY OPEC TALKS", "INDONESIA SEEN AT CROSSROADS OVER ECONOMIC CHANGE", 
                                                                                                                                                                                                                 "SAUDI RIYAL DEPOSIT RATES REMAIN FIRM", "QATAR UNVEILS BUDGET FOR FISCAL 1987/88", 
                                                                                                                                                                                     "SAUDI ARABIA REITERATES COMMITMENT TO OPEC PACT", "SAUDI FEBRUARY CRUDE OUTPUT PUT AT 3.5 MLN BPD", 
                                                                                                                                                                                     "GULF ARAB DEPUTY OIL MINISTERS TO MEET IN BAHRAIN", "SAUDI ARABIA REITERATES COMMITMENT TO OPEC ACCORD", 
                                                                                                                                                                                     "KUWAIT MINISTER SAYS NO EMERGENCY OPEC TALKS SET", "PHILADELPHIA PORT CLOSED BY TANKER CRASH", 
                                                                                                                                                                                     "STUDY GROUP URGES INCREASED U.S. OIL RESERVES", "STUDY GROUP URGES INCREASED U.S. OIL RESERVES", 
                                                                                                                                                                                     "UNOCAL <UCL> UNIT CUTS CRUDE OIL POSTED PRICES", "NYMEX WILL EXPAND OFF-HOUR TRADING APRIL ONE", 
                                                                                                                                                                                     "ARGENTINE OIL PRODUCTION DOWN IN JANUARY 1987"), id = c("127", 
                                                                                                                                                                                                                                              "144", "191", "194", "211", "236", "237", "242", "246", "248", 
                                                                                                                                                                                                                                              "273", "349", "352", "353", "368", "489", "502", "543", "704", 
                                                                                                                                                                                                                                              "708"), language = c("en", "en", "en", "en", "en", "en", "en", 
                                                                                                                                                                                                                                                                   "en", "en", "en", "en", "en", "en", "en", "en", "en", "en", "en", 
                                                                                                                                                                                                                                                                   "en", "en"), origin = c("Reuters-21578 XML", "Reuters-21578 XML", 
                                                                                                                                                                                                                                                                                           "Reuters-21578 XML", "Reuters-21578 XML", "Reuters-21578 XML", 
                                                                                                                                                                                                                                                                                           "Reuters-21578 XML", "Reuters-21578 XML", "Reuters-21578 XML", 
                                                                                                                                                                                                                                                                                           "Reuters-21578 XML", "Reuters-21578 XML", "Reuters-21578 XML", 
                                                                                                                                                                                                                                                                                           "Reuters-21578 XML", "Reuters-21578 XML", "Reuters-21578 XML", 
                                                                                                                                                                                                                                                                                           "Reuters-21578 XML", "Reuters-21578 XML", "Reuters-21578 XML", 
                                                                                                                                                                                                                                                                                           "Reuters-21578 XML", "Reuters-21578 XML", "Reuters-21578 XML"
                                                                                                                                                                                                                                                                   ), topics = c("YES", "YES", "YES", "YES", "YES", "YES", "YES", 
                                                                                                                                                                                                                                                                                 "YES", "YES", "YES", "YES", "YES", "YES", "YES", "YES", "YES", 
                                                                                                                                                                                                                                                                                 "YES", "YES", "YES", "YES"), lewissplit = c("TRAIN", "TRAIN", 
                                                                                                                                                                                                                                                                                                                             "TRAIN", "TRAIN", "TRAIN", "TRAIN", "TRAIN", "TRAIN", "TRAIN", 
                                                                                                                                                                                                                                                                                                                             "TRAIN", "TRAIN", "TRAIN", "TRAIN", "TRAIN", "TRAIN", "TRAIN", 
                                                                                                                                                                                                                                                                                                                             "TRAIN", "TRAIN", "TRAIN", "TRAIN"), cgisplit = c("TRAINING-SET", 
                                                                                                                                                                                                                                                                                                                                                                               "TRAINING-SET", "TRAINING-SET", "TRAINING-SET", "TRAINING-SET", 
                                                                                                                                                                                                                                                                                                                                                                               "TRAINING-SET", "TRAINING-SET", "TRAINING-SET", "TRAINING-SET", 
                                                                                                                                                                                                                                                                                                                                                                               "TRAINING-SET", "TRAINING-SET", "TRAINING-SET", "TRAINING-SET", 
                                                                                                                                                                                                                                                                                                                                                                               "TRAINING-SET", "TRAINING-SET", "TRAINING-SET", "TRAINING-SET", 
                                                                                                                                                                                                                                                                                                                                                                               "TRAINING-SET", "TRAINING-SET", "TRAINING-SET"), oldid = c("5670", 
                                                                                                                                                                                                                                                                                                                                                                                                                                          "5687", "5734", "5737", "5754", "8321", "8322", "8327", "8331", 
                                                                                                                                                                                                                                                                                                                                                                                                                                          "8333", "12456", "12532", "12535", "12536", "12550", "12672", 
                                                                                                                                                                                                                                                                                                                                                                                                                                          "12685", "12726", "12887", "12891"), places = structure(list(
                                                                                                                                                                                                                                                                                                                                                                                                                                            `127` = "usa", `144` = "usa", `191` = "canada", `194` = "usa", 
                                                                                                                                                                                                                                                                                                                                                                                                                                            `211` = "usa", `236` = c("kuwait", "ecuador"), `237` = c("indonesia", 
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     "usa"), `242` = c("bahrain", "saudi-arabia"), `246` = "qatar", 
                                                                                                                                                                                                                                                                                                                                                                                                                                            `248` = c("bahrain", "saudi-arabia"), `273` = c("saudi-arabia", 
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            "uae"), `349` = c("uae", "bahrain", "saudi-arabia", "kuwait", 
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              "qatar"), `352` = c("saudi-arabia", "bahrain"), `353` = "kuwait", 
                                                                                                                                                                                                                                                                                                                                                                                                                                            `368` = "usa", `489` = "usa", `502` = "usa", `543` = "usa", 
                                                                                                                                                                                                                                                                                                                                                                                                                                            `704` = "usa", `708` = "argentina"), .Names = c("127", "144", 
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            "191", "194", "211", "236", "237", "242", "246", "248", "273", 
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            "349", "352", "353", "368", "489", "502", "543", "704", "708"
                                                                                                                                                                                                                                                                                                                                                                                                                                            )), people = c(NA, NA, NA, NA, NA, NA, NA, NA, NA, "hisham-nazer", 
                                                                                                                                                                                                                                                                                                                                                                                                                                                           NA, NA, "hisham-nazer", NA, NA, NA, NA, NA, NA, NA), orgs = c(NA, 
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         "opec", NA, NA, NA, "opec", "worldbank", "opec", NA, "opec", 
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         "opec", "opec", "opec", "opec", NA, NA, NA, NA, NA, NA), exchanges = c(NA, 
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, 
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                NA, "nymex", NA), text = c("Diamond Shamrock Corp said that\neffective today it had cut its contract prices for crude oil by\n1.50 dlrs a barrel.\n    The reduction brings its posted price for West Texas\nIntermediate to 16.00 dlrs a barrel, the copany said.\n    \"The price reduction today was made in the light of falling\noil product prices and a weak crude oil market,\" a company\nspokeswoman said.\n    Diamond is the latest in a line of U.S. oil companies that\nhave cut its contract, or posted, prices over the last two days\nciting weak oil markets.\n Reuter", 
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           "OPEC may be forced to meet before a\nscheduled June session to readdress its production cutting\nagreement if the organization wants to halt the current slide\nin oil prices, oil industry analysts said.\n    \"The movement to higher oil prices was never to be as easy\nas OPEC thought. They may need an emergency meeting to sort out\nthe problems,\" said Daniel Yergin, director of Cambridge Energy\nResearch Associates, CERA.\n    Analysts and oil industry sources said the problem OPEC\nfaces is excess oil supply in world oil markets.\n    \"OPEC's problem is not a price problem but a production\nissue and must be addressed in that way,\" said Paul Mlotok, oil\nanalyst with Salomon Brothers Inc.\n    He said the market's earlier optimism about OPEC and its\nability to keep production under control have given way to a\npessimistic outlook that the organization must address soon if\nit wishes to regain the initiative in oil prices.\n    But some other analysts were uncertain that even an\nemergency meeting would address the problem of OPEC production\nabove the 15.8 mln bpd quota set last December.\n    \"OPEC has to learn that in a buyers market you cannot have\ndeemed quotas, fixed prices and set differentials,\" said the\nregional manager for one of the major oil companies who spoke\non condition that he not be named. \"The market is now trying to\nteach them that lesson again,\" he added.\n    David T. Mizrahi, editor of Mideast reports, expects OPEC\nto meet before June, although not immediately. However, he is\nnot optimistic that OPEC can address its principal problems.\n    \"They will not meet now as they try to take advantage of the\nwinter demand to sell their oil, but in late March and April\nwhen demand slackens,\" Mizrahi said.\n    But Mizrahi said that OPEC is unlikely to do anything more\nthan reiterate its agreement to keep output at 15.8 mln bpd.\"\n    Analysts said that the next two months will be critical for\nOPEC's ability to hold together prices and output.\n    \"OPEC must hold to its pact for the next six to eight weeks\nsince buyers will come back into the market then,\" said Dillard\nSpriggs of Petroleum Analysis Ltd in New York.\n    But Bijan Moussavar-Rahmani of Harvard University's Energy\nand Environment Policy Center said that the demand for OPEC oil\nhas been rising through the first quarter and this may have\nprompted excesses in its production.\n    \"Demand for their (OPEC) oil is clearly above 15.8 mln bpd\nand is probably closer to 17 mln bpd or higher now so what we\nare seeing characterized as cheating is OPEC meeting this\ndemand through current production,\" he told Reuters in a\ntelephone interview.\n Reuter", 
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           "Texaco Canada said it lowered the\ncontract price it will pay for crude oil 64 Canadian cts a\nbarrel, effective today.\n    The decrease brings the company's posted price for the\nbenchmark grade, Edmonton/Swann Hills Light Sweet, to 22.26\nCanadian dlrs a bbl.\n    Texaco Canada last changed its crude oil postings on Feb\n19.\n Reuter", 
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           "Marathon Petroleum Co said it reduced\nthe contract price it will pay for all grades of crude oil one\ndlr a barrel, effective today.\n    The decrease brings Marathon's posted price for both West\nTexas Intermediate and West Texas Sour to 16.50 dlrs a bbl. The\nSouth Louisiana Sweet grade of crude was reduced to 16.85 dlrs\na bbl.\n    The company last changed its crude postings on Jan 12.\n Reuter", 
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           "Houston Oil Trust said that independent\npetroleum engineers completed an annual study that estimates\nthe trust's future net revenues from total proved reserves at\n88 mln dlrs and its discounted present value of the reserves at\n64 mln dlrs.\n    Based on the estimate, the trust said there may be no money\navailable for cash distributions to unitholders for the\nremainder of the year.\n    It said the estimates reflect a decrease of about 44 pct in\nnet reserve revenues and 39 pct in discounted present value\ncompared with the study made in 1985.\n Reuter", 
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           "Kuwait\"s Oil Minister, in remarks\npublished today, said there were no plans for an emergency OPEC\nmeeting to review oil policies after recent weakness in world\noil prices.\n    Sheikh Ali al-Khalifa al-Sabah was quoted by the local\ndaily al-Qabas as saying: \"None of the OPEC members has asked\nfor such a meeting.\"\n    He denied Kuwait was pumping above its quota of 948,000\nbarrels of crude daily (bpd) set under self-imposed production\nlimits of the 13-nation organisation.\n    Traders and analysts in international oil markets estimate\nOPEC is producing up to one mln bpd above a ceiling of 15.8 mln\nbpd agreed in Geneva last December.\n    They named Kuwait and the United Arab Emirates, along with\nthe much smaller producer Ecuador, among those producing above\nquota. Kuwait, they said, was pumping 1.2 mln bpd.\n    \"This rumour is baseless. It is based on reports which said\nKuwait has the ability to exceed its share. They suppose that\nbecause Kuwait has the ability, it will do so,\" the minister\nsaid.\n    Sheikh Ali has said before that Kuwait had the ability to\nproduce up to 4.0 mln bpd.\n    \"If we can sell more than our quota at official prices,\nwhile some countries are suffering difficulties marketing their\nshare, it means we in Kuwait are unusually clever,\" he said.\n    He was referring apparently to the Gulf state of qatar,\nwhich industry sources said was selling less than 180,000 bpd\nof its 285,000 bpd quota, because buyers were resisting\nofficial prices restored by OPEC last month pegged to a marker\nof 18 dlrs per barrel.\n    Prices in New York last week dropped to their lowest levels\nthis year and almost three dollars below a three-month high of\n19 dollars a barrel.\n    Sheikh Ali also delivered \"a challenge to any international\noil company that declared Kuwait sold below official prices.\"\n    Because it was charging its official price, of 16.67 dlrs a\nbarrel, it had lost custom, he said but did not elaborate.\n    However, Kuwait had guaranteed markets for its oil because\nof its local and international refining facilities and its own\ndistribution network abroad, he added.\n    He reaffirmed that the planned meeting March 7 of OPEC\"s\ndifferentials committee has been postponed until the start of\nApril at the request of certain of the body\"s members.\n    Ecuador\"s deputy energy minister Fernando Santos Alvite said\nlast Wednesday his debt-burdened country wanted OPEC to assign\na lower official price for its crude, and was to seek this at\ntalks this month of opec\"s pricing committee.\n    Referring to pressure by oil companies on OPEC members, in\napparent reference to difficulties faced by Qatar, he said: \"We\nexpected such pressure. It will continue through March and\nApril.\" But he expected the situation would later improve.\n REUTER", 
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           "Indonesia appears to be nearing a\npolitical crossroads over measures to deregulate its protected\neconomy, the U.S. Embassy says in a new report.\n    To counter falling oil revenues, the government has\nlaunched a series of measures over the past nine months to\nboost exports outside the oil sector and attract new\ninvestment.\n    Indonesia, the only Asian member of OPEC and a leading\nprimary commodity producer, has been severely hit by last year\"s\nfall in world oil prices, which forced it to devalue its\ncurrency by 31 pct in September.\n    But the U.S. Embassy report says President Suharto\"s\ngovernment appears to be divided over what direction to lead\nthe economy.\n    \"(It) appears to be nearing a crossroads with regard to\nderegulation, both as it pertains to investments and imports,\"\nthe report says. It primarily assesses Indonesia\"s agricultural\nsector, but also reviews the country\"s general economic\nperformance.\n    It says that while many government officials and advisers\nare recommending further relaxation, \"there are equally strong\npressures being exerted to halt all such moves.\"\n    \"This group strongly favours an import substitution economy,\"\nthe report says.\n    Indonesia\"s economic changes have been welcomed by the World\nBank and international bankers as steps in the right direction,\nthough they say crucial areas of the economy like plastics and\nsteel remain highly protected, and virtual monopolies.\n    Three sets of measures have been announced since last May,\nwhich broadened areas for foreign investment, reduced trade\nrestrictions and liberalised imports.\n    The report says Indonesia\"s economic growth in calendar 1986\nwas probably about zero, and the economy may even have\ncontracted a bit. \"This is the lowest rate of growth since the\nmid-1960s,\" the report notes.\n    Indonesia, the largest country in South-East Asia with a\npopulation of 168 million, is facing general elections in\nApril.\n    But the report hold out little hope for swift improvement\nin the economic outlook. \"For 1987 early indications point to a\nslightly positive growth rate not exceeding one pct. Economic\nactivity continues to suffer due to the sharp fall in export\nearnings from the petroleum industry.\"\n    \"Growth in the non-oil sector is low because of weak\ndomestic demand coupled with excessive plant capacity, real\ndeclines in construction and trade, and a reduced level of\ngrowth in agriculture,\" the report states.\n    Bankers say continuation of present economic reforms is\ncrucial for the government to get the international lending its\nneeds.\n    A new World Bank loan of 300 mln dlrs last month in balance\nof payments support was given partly to help the government\nmaintain the momentum of reform, the Bank said.\n REUTER", 
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           "Saudi riyal interbank deposits were\nsteady at yesterday's higher levels in a quiet market.\n    Traders said they were reluctant to take out new positions\namidst uncertainty over whether OPEC will succeed in halting\nthe current decline in oil prices.\n    Oil industry sources said yesterday several Gulf Arab\nproducers had had difficulty selling oil at official OPEC\nprices but Kuwait has said there are no plans for an emergency\nmeeting of the 13-member organisation.\n    A traditional Sunday lull in trading due to the European\nweekend also contributed to the lack of market activity.\n    Spot-next and one-week rates were put at 6-1/4, 5-3/4 pct\nafter quotes ranging between seven, six yesterday.\n    One, three, and six-month deposits were quoted unchanged at\n6-5/8, 3/8, 7-1/8, 6-7/8 and 7-3/8, 1/8 pct respectively.\n    The spot riyal was quietly firmer at 3.7495/98 to the\ndollar after quotes of 3.7500/03 yesterday.\n REUTER", 
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           "The Gulf oil state of Qatar, recovering\nslightly from last year's decline in world oil prices,\nannounced its first budget since early 1985 and projected a\ndeficit of 5.472 billion riyals.\n    The deficit compared with a shortfall of 7.3 billion riyals\nin the last published budget for 1985/86.\n    In a statement outlining the budget for the fiscal year\n1987/88 beginning today, Finance and Petroleum Minister Sheikh\nAbdul-Aziz bin Khalifa al-Thani said the government expected to\nspend 12.217 billion riyals in the period.\n    Projected expenditure in the 1985/86 budget had been 15.6\nbillion riyals.\n    Sheikh Abdul-Aziz said government revenue would be about\n6.745 billion riyals, down by about 30 pct on the 1985/86\nprojected revenue of 9.7 billion.\n    The government failed to publish a 1986/87 budget due to\nuncertainty surrounding oil revenues.\n    Sheikh Abdul-Aziz said that during that year the government\ndecided to limit recurrent expenditure each month to\none-twelfth of the previous fiscal year's allocations minus 15\npct.\n    He urged heads of government departments and public\ninstitutions to help the government rationalise expenditure. He\ndid not say how the 1987/88 budget shortfall would be covered.\n    Sheikh Abdul-Aziz said plans to limit expenditure in\n1986/87 had been taken in order to relieve the burden placed on\nthe country's foreign reserves.\n    He added in 1987/88 some 2.766 billion riyals had been\nallocated for major projects including housing and public\nbuildings, social services, health, education, transport and\ncommunications, electricity and water, industry and\nagriculture.\n    No figure was revealed for expenditure on defence and\nsecurity. There was also no projection for oil revenue.\n    Qatar, an OPEC member, has an output ceiling of 285,000\nbarrels per day.\n    Sheikh Abdul-Aziz said: \"Our expectations of positive signs\nregarding (oil) price trends, foremost among them OPEC's\ndetermination to shoulder its responsibilites and protect its\nwealth, have helped us make reasonable estimates for the coming\nyear's revenue on the basis of our assigned quota.\"\n REUTER", 
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           "Saudi Arabian Oil Minister Hisham Nazer\nreiterated the kingdom's commitment to last December's OPEC\naccord to boost world oil prices and stabilise the market, the\nofficial Saudi Press Agency SPA said.\n    Asked by the agency about the recent fall in free market\noil prices, Nazer said Saudi Arabia \"is fully adhering by the\n... Accord and it will never sell its oil at prices below the\npronounced prices under any circumstance.\"\n    Nazer, quoted by SPA, said recent pressure on free market\nprices \"may be because of the end of the (northern hemisphere)\nwinter season and the glut in the market.\"\n    Saudi Arabia was a main architect of the December accord,\nunder which OPEC agreed to lower its total output ceiling by\n7.25 pct to 15.8 mln barrels per day (bpd) and return to fixed\nprices of around 18 dlrs a barrel.\n    The agreement followed a year of turmoil on oil markets,\nwhich saw prices slump briefly to under 10 dlrs a barrel in\nmid-1986 from about 30 dlrs in late 1985. Free market prices\nare currently just over 16 dlrs.\n    Nazer was quoted by the SPA as saying Saudi Arabia's\nadherence to the accord was shown clearly in the oil market.\n    He said contacts among members of OPEC showed they all\nwanted to stick to the accord.\n    In Jamaica, OPEC President Rilwanu Lukman, who is also\nNigerian Oil Minister, said the group planned to stick with the\npricing agreement.\n    \"We are aware of the negative forces trying to manipulate\nthe operations of the market, but we are satisfied that the\nfundamentals exist for stable market conditions,\" he said.\n    Kuwait's Oil Minister, Sheikh Ali al-Khalifa al-Sabah, said\nin remarks published in the emirate's daily Al-Qabas there were\nno plans for an emergency OPEC meeting to review prices.\n    Traders and analysts in international oil markets estimate\nOPEC is producing up to one mln bpd above the 15.8 mln ceiling.\n    They named Kuwait and the United Arab Emirates, along with\nthe much smaller producer Ecuador, among those producing above\nquota. Sheikh Ali denied that Kuwait was over-producing.\n REUTER", 
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           "Saudi crude oil output last month fell\nto an average of 3.5 mln barrels per day (bpd) from 3.8 mln bpd\nin January, Gulf oil sources said.\n    They said exports from the Ras Tanurah and Ju'aymah\nterminals in the Gulf fell to an average 1.9 mln bpd last month\nfrom 2.2 mln in January because of lower liftings by some\ncustomers.\n    But the drop was much smaller than expected after Gulf\nexports rallied in the fourth week of February to 2.5 mln bpd\nfrom 1.2 mln in the third week, the sources said.\n    The production figures include neutral zone output but not\nsales from floating storage, which are generally considered\npart of a country's output for Opec purposes.\n    Saudi Arabia has an Opec quota of 4.133 mln bpd under a\nproduction restraint scheme approved by the 13-nation group\nlast December to back new official oil prices averaging 18 dlrs\na barrel.\n    The sources said the two-fold jump in exports last week\nappeared to be the result of buyers rushing to lift February\nentitlements before the month-end.\n    Last week's high export levels appeared to show continued\nsupport for official Opec prices from Saudi Arabia's main crude\ncustomers, the four ex-partners of Aramco, the sources said.\n    The four -- Exxon Corp <XON>, Mobil Corp <MOB>, Texaco Inc\n<TX> and Chevron Corp <CHV> -- signed a long-term agreement\nlast month to buy Saudi crude for 17.52 dlrs a barrel.\n    However the sources said the real test of Saudi Arabia's\nability to sell crude at official prices in a weak market will\ncome this month, when demand for petroleum products\ntraditionally tapers off. Spot prices have fallen in recent\nweeks to more than one dlr below Opec levels.\n    Saudi Arabian oil minister Hisham Nazer yesterday\nreiterated the kingdom's commitment to the December OPEC accord\nand said it would never sell below official prices.\n    The sources said total Saudi refinery throughput fell\nslightly in February to an average 1.1 mln bpd from 1.2 mln in\nJanuary because of cuts at the Yanbu and Jubail export\nrefineries.\n    They put crude oil exports through Yanbu at 100,000 bpd\nlast month, compared to zero in January, while throughput at\nBahrain's refinery and neutral zone production remained steady\nat around 200,000 bpd each.\n REUTER", 
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           "Deputy oil ministers from six Gulf\nArab states will meet in Bahrain today to discuss coordination\nof crude oil marketing, the official Emirates news agency WAM\nreported.\n    WAM said the officials would be discussing implementation\nof last Sunday's agreement in Doha by Gulf Cooperation Council\n(GCC) oil ministers to help each other market their crude oil.\n    Four of the GCC states - Saudi Arabia, the United Arab\nEmirates (UAE), Kuwait and Qatar - are members of the\nOrganiaation of Petroleum Exporting Countries (OPEC) and some\nface stiff buyer resistance to official OPEC prices.\n Reuter", 
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           "Saudi Arabian Oil Minister Hisham Nazer\nreiterated the kingdom's commitment to last December's OPEC\naccord to boost world oil prices and stabilize the market, the\nofficial Saudi Press Agency SPA said.\n    Asked by the agency about the recent fall in free market\noil prices, Nazer said Saudi Arabia \"is fully adhering by the\n... accord and it will never sell its oil at prices below the\npronounced prices under any circumstance.\"\n    Saudi Arabia was a main architect of December pact under\nwhich OPEC agreed to cut its total oil output ceiling by 7.25\npct and return to fixed prices of around 18 dollars a barrel.\n Reuter", 
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           "Kuwait's oil minister said in a newspaper\ninterview that there were no plans for an emergency OPEC\nmeeting after the recent weakness in world oil prices.\n    Sheikh Ali al-Khalifa al-Sabah was quoted by the local\ndaily al-Qabas as saying that \"none of the OPEC members has\nasked for such a meeting.\"\n    He also denied that Kuwait was pumping above its OPEC quota\nof 948,000 barrels of crude daily (bpd).\n    Crude oil prices fell sharply last week as international\noil traders and analysts estimated the 13-nation OPEC was\npumping up to one million bpd over its self-imposed limits.\n Reuter", 
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           "The port of Philadelphia was closed\nwhen a Cypriot oil tanker, Seapride II, ran aground after\nhitting a 200-foot tower supporting power lines across the\nriver, a Coast Guard spokesman said.\n    He said there was no oil spill but the ship is lodged on\nrocks opposite the Hope Creek nuclear power plant in New\nJersey.\n    He said the port would be closed until today when they\nhoped to refloat the ship on the high tide.\n    After delivering oil to a refinery in Paulsboro, New\nJersey, the ship apparently lost its steering and hit the power\ntransmission line carrying power from the nuclear plant to the\nstate of Delaware.\n Reuter", 
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           "A study group said the United States\nshould increase its strategic petroleum reserve to one mln\nbarrels as one way to deal with the present and future impact\nof low oil prices on the domestic oil industry.\n    U.S. policy now is to raise the strategic reserve to 750\nmln barrels, from its present 500 mln, to help protect the\neconomy from an overseas embargo or a sharp price rise.\n    The Aspen Institute for Humanistic Studies, a private\ngroup, also called for new research for oil exploration and\ndevelopment techniques.\n    It predicted prices would remain at about 15-18 dlrs a\nbarrel for several years and then rise to the mid 20s, with\nimports at about 30 pct of U.S. consumption.\n    It said instead that such moves as increasing oil reserves\nand more exploration and development research would help to\nguard against or mitigate the risks of increased imports.\n Reuter", 
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           "A study group said the United States\nshould increase its strategic petroleum reserve to one mln\nbarrels as one way to deal with the present and future impact\nof low oil prices on the domestic oil industry.\n    U.S. policy now is to raise the strategic reserve to 750\nmln barrels, from its present 500 mln, to help protect the\neconomy from an overseas embargo or a sharp price rise.\n    The Aspen Institute for Humanistic Studies, a private\ngroup, also called for new research for oil exploration and\ndevelopment techniques.\n    It predicted prices would remain at about 15-18 dlrs a\nbarrel for several years and then rise to the mid 20s, with\nimports at about 30 pct of U.S. consumption.\n    The study cited two basic policy paths for the nation: to\nprotect the U.S. industry through an import fee or other such\ndevice or to accept the full economic benefits of cheap oil.\n    But the group did not strongly back either option, saying\nthere were benefits and drawbacks to both.\n    It said instead that such moves as increasing oil reserves\nand more exploration and development research would help to\nguard against or mitigate the risks of increased imports.\n Reuter", 
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           "Unocal Corp's Union Oil Co said it\nlowered its posted prices for crude oil one to 1.50 dlrs a\nbarrel in the eastern region of the U.S., effective Feb 26.\n    Union said a 1.50 dlrs cut brings its posted price for the\nU.S. benchmark grade, West Texas Intermediate, to 16 dlrs.\nLouisiana Sweet also was lowered 1.50 dlrs to 16.35 dlrs, the\ncompany said.\n    No changes were made in Union's posted prices for West\nCoast grades of crude oil, the company said.\n Reuter", 
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           "The New York Mercantile Exchange set\nApril one for the debut of a new procedure in the energy\ncomplex that will increase the use of energy futures worldwide.\n     On April one, NYMEX will allow oil traders that do not\nhold a futures position to initiate, after the exchange closes,\na transaction that can subsequently be hedged in the futures\nmarket, according to an exchange spokeswoman.\n    \"This will change the way oil is transacted in the real\nworld,\" said said Thomas McKiernan, McKiernan and Co chairman.\n    Foreign traders will be able to hedge trades against NYMEX\nprices before the exchange opens and negotiate prices at a\ndifferential to NYMEX prices, McKiernan explained.\n     The expanded program \"will serve the industry because the\noil market does not close when NYMEX does,\" said Frank Capozza,\nsecretary of Century Resources Inc.\n     The rule change, which has already taken effect for\nplatinum futures on NYMEX, is expected to increase the open\ninterest and liquidity in U.S. energy futures, according to\ntraders and analysts.\n    Currently, at least one trader in this transaction, called\nan exchange for physical or EFP, must hold a futures position\nbefore entering into the transaction.\n    Under the new arrangement, neither party has to hold a\nfutures position before entering into an EFP and one or both\nparties can offset their cash transaction with a futures\ncontract the next day, according to exchange officials.\n    When NYMEX announced its proposed rule change in December,\nNYMEX President Rosemary McFadden, said, \"Expansion of the EFP\nprovision will add to globalization of the energy markets by\nproviding for, in effect, 24-hour trading.\"\n    The Commodity Futures Trading Commission approved the rule\nchange in February, according to a CFTC spokeswoman.\n Reuter", 
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           "Argentine crude oil production was\ndown 10.8 pct in January 1987 to 12.32 mln barrels, from 13.81\nmln barrels in January 1986, Yacimientos Petroliferos Fiscales\nsaid.\n    January 1987 natural gas output totalled 1.15 billion cubic\nmetrers, 3.6 pct higher than 1.11 billion cubic metres produced\nin January 1986, Yacimientos Petroliferos Fiscales added.\n Reuter"
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                ), article_id = 1:20), .Names = c("author", "datetimestamp", 
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  "description", "heading", "id", "language", "origin", "topics", 
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  "lewissplit", "cgisplit", "oldid", "places", "people", "orgs", 
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          "exchanges", "text", "article_id"), row.names = c(NA, -20L), class = c("tbl_df", 
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     "tbl", "data.frame"))
glimpse(crude_tibble) 

# Count occurrence by article_id and word
words <- crude_tibble %>%
  unnest_tokens(output = "word", token = "words", input = text) %>%
  anti_join(stop_words) %>%
  count(article_id,word, sort=TRUE)

# How many different word/article combinations are there?
unique_combinations <- nrow(words)

# Filter to responses with the word "prices"
words_with_prices <- words %>%
  filter(word == "prices")

# How many articles had the word "prices"?
number_of_price_articles <- nrow(words_with_prices)

###
##Sparse matrices
###

# During the video lesson you learned about sparse matrices. 
# Sparse matrices can become computational nightmares as the number of text documents and the number of unique words grow. 
# Creating word representations with tweets can easily create sparse matrices because emojis, slang, acronyms, and other forms of language are used.

# In this exercise you will walk through the steps to calculate how sparse the Russian tweet dataset is. 
# Note that this is a small example of how quickly text analysis can become a major computational problem.
# Tokenize and remove stop words

tidy_tweets <- russian_tweets %>%
  unnest_tokens(word, content) %>%
  anti_join(stop_words)

# Count by word
unique_words <- tidy_tweets %>%
  count(word)

# Count by tweet (tweet_id) and word
unique_words_by_tweet <- tidy_tweets %>%
  count(tweet_id, word)

# Find the size of matrix
size <- nrow(russian_tweets) * nrow(unique_words)

# Find percent of entries that would have a value
percent <- nrow(unique_words_by_tweet) / size
percent

#### 
## The TFIDF

clean_t1 <- "john friend joe tacos"
clean_t1 <- "common friend john joe names"
clean_t1 <- "tacos favorite food eat buddy joe"

# TF: term frequency - the proportion of words in a text that are that term 
                    # (john is 1/4 words in clean_t1, tf = 0.25)

# IDS: inverse document frequency - the weight of how common a term is across all documents.
                    # john is 3/3 documents, IDF = 0

# IDF equation - has many forms, we will use most common:
# IDF = log(N/nt), N = total number of documents in the corpus,  nt = number of documents, where the term appears


# TFIDF for "tacos": 
  # clean_t1 = 1/4 * log(3/2) = 0.101
  # clean_t2 = 0/5 * log(3/2) = 0
  # clean_t3 = 1/6 * log(3/2) = 0.0675


####
## Manual calculation exercise:

t1 <- "government turtle blue ocean"
t2 <- "crazy turtle ocean waves"
t3 <- "massive turtle washington lion"
t4 <- "lion pride massive ocean dinner"

#Calculate the TF and IDF weights for 'turtle' in t1

# TF: word 'turtle' is once in t1. There are 4 words in t1, so: TF(t1, turtle) = 1/4 = 0.25
# IDF: word 'turtle' is in t1, t2, t3 but not in t4, som IDF = log(4/3)

#TFIDF : 1/4 * log(4/3) = 0.07192

####
## TFIDF Practice

# Earlier you looked at a bag-of-words representation of articles on crude oil. 
# Calculating TFIDF values relies on this bag-of-words representation, but takes into account how often a word appears in an article, and how often that word appears in the collection of articles.

# To determine how meaningful words would be when comparing different 
# articles, calculate the TFIDF weights for the words in crude, a collection of 20 articles about crude oil.

# Create a tibble with TFIDF values
crude_weights <- crude_tibble %>%
  unnest_tokens(output = "word", token = "words", input = text) %>%
  anti_join(stop_words) %>%
  count(article_id, word) %>%
  bind_tf_idf(term = word, document = article_id, n) # pozor na poradie ! 

# Find the highest TFIDF values
crude_weights %>%
  arrange(desc(tf_idf))

# Find the lowest non-zero TFIDF values
crude_weights %>%
  filter(tf_idf != 0) %>%
  arrange(tf_idf)

# Excellent. We see that 'prices' and 'petroleum' have very low values for some articles. 
# This could be because they were mentioned just a few times in that article, or because they 
#     were used in too many articles.

## ZÁVER: TF-IDF(slovo, document_id, SUBODR-DOKUMENTOV).
#  Ak je TF-IDF nízke, znamena to, ze je slovo sa vykytuje malo krat v danom dokumente, ALEBO! velmi casto v subore dokumentov
#  Ak je TF-IDF vysoke, znamena to, ze dane slovo je pre clanok "vynimocne", nakolko sa musi vyskytovat vela krat v clanku a zaroven sa nevyskytuje casto vo viacerych odlisnych clankoch CORPUSU


########
## COSINE SIMILARITY

# measure of similarity between two vvecotrs 
# measured the angle formed by thee two vectors

# najdenie uhla medzi dvomi vektormi v multidimenzionalnej sustave
# similarity = cos(alfa) = A * B / (|A|*|B|)  --> skalarny sucin predeleny normami

# We will use PAIRWISE SIMILARITY:
# pairwise_similarity(tbl, item, feature, value, ...) 
#     -- item to compere (tweets, articles),
#     -- feature: column describing the link between the items (i.e. words)
#     -- value: the column of values (i.e. n or tf_idf)

####
## An example of failing at text analysis - similarity without removing stop-words

# Early on, you discussed the power of removing stop words before conducting text analysis. 
# In this most recent chapter, you reviewed using cosine similarity to identify texts that are similar to each other.

# In this exercise, you will explore the very real possibility of failing to use text analysis properly. 
# You will compute cosine similarities for the chapters in the book Animal Farm, without removing stop-words.

# Create word counts
animal_farm_counts <- animal_farm %>%
  unnest_tokens(word, text_column) %>%
  count(chapter, word)

# Calculate the cosine similarity by chapter, using words
comparisons <- animal_farm_counts %>%
  pairwise_similarity(item = chapter, feature = word, value = n) %>%
  arrange(desc(similarity))

# Print the mean of the similarity values
comparisons %>%
  summarize(mean = mean(similarity))

# Well done. Unfortunately, these results are useless. 
# As every single chapter is highly similar to every other chaper. 
# We need to remove stop words to see which chapters are more similar to each other.
# As well, we should include only one pair A-B, not both A-B and B-A

# Create word counts 
animal_farm_counts <- animal_farm %>%
  unnest_tokens(word, text_column) %>%
  anti_join(stop_words) %>%
  count(chapter, word) %>%
  bind_tf_idf(word, chapter, n)  ## toto mali podla mna zle v DataCampe, malo by to byt word, chapter miesto chapter,word

# Calculate cosine similarity on word counts
animal_farm_counts %>%
  pairwise_similarity(chapter, word, n) %>%
  arrange(desc(similarity))
# Calculate cosine similarity using tf_idf values
animal_farm_counts %>%
  pairwise_similarity(chapter, word, tf_idf) %>%
  arrange(desc(similarity))


#####
## CHAPTER 3: Classification and Topic modeling 
#####

animal_matrix <- animal_tokens %>%
  count(sentence_id, word) %>% 
  tidytext::cast_dtm(document = sentence_id, term = word, value = n, weighting = tm::weightTfIdf)

# cast_dtm(document = sentence_id, term = word, value = n, weighting = tm::weightTfIdf)
#  vytvori pre kazdy document (mozu to byt aj vety) jeden riadok tolko stlpcov, kolko je unikatnych slov 
#  vytvoria sa tym padom sparse matice (0 = nenachadza sa v dokumente / vete dane slovo. POZOR!! Ak pouzijeme TF-IDF, sparse hladi prave na 0-ove hodnoty TF-IDF a nie len term frequency)
#  sparsity = podiel 0 hodnot v danej matici 
#  POZNAMKA: tym ze sme pouzili weighting: TF-IDF, v matici budu vyplnene TF-IDF koeficienty. Defaultne je TF.. 

# OPAKOM cast_dtm je tidy(), ktora z dtm objektu vytvori sumarizovanu tabulku s document_id, word, count stlpcami

# Obrovske sparse matice mozu sposobovat vypoctove problemy a preto mozeme moc spars "terms = word"  vymazat a zmensit dimenzie matice
# funkcia: removeSparseTerms(animal_matrix, sparse = 0.90) - zredukuje slova, ktore chybaju vo viac ako 90% dokumentoch (vetách)
#                                                          - ak podiel 0-vych hodnot pre stlpcec (term) > 0.9, tak ho vyhodi 

### Exercise
# Data preparation

# During the 2016 US election, Russian tweet bots were used to constantly distribute political 
# rhetoric to both democrats and republicans. You have been given a dataset of such tweets called russian_tweets. 
# You have decided to classify these tweets as either left- (democrat) or right-leaning(republican). 
# Before you can build a classification model, you need to clean and prepare the text for modeling.

# Stem the tokens
russian_tokens <- russian_tweets %>%
  unnest_tokens(output = "word", token = "words", input = content) %>%
  anti_join(stop_words) %>%
  mutate(word = wordStem(word))

# Create a document term matrix using TFIDF weighting
tweet_matrix <- russian_tokens %>%
  count(tweet_id, word) %>%
  cast_dtm(document = tweet_id, term = word,
           value = n, weighting = tm::weightTfIdf)

# Print the matrix details 
tweet_matrix

#### Removing sparse terms

# Running classification models on sparse matrices can be a computational nightmare. 
# Without access to GPUs or cloud compute resources, you might run into time and memory issues on your local computer. 
# You have been given a document-term matrix and plan on running several different algorithms to find the best classification model. 
# In this exercise, you will remove some of the sparse terms from the provided matrix, matrix, at different sparsity levels.
# For each level of sparsity, note the number of remaining terms in the matrix.

less_sparse_matrix <-
  removeSparseTerms(matrix, sparse = 0.9999) # ostalo 9444 slov z povodnych 38391

# Print results
matrix
less_sparse_matrix



###
#  Classification modeling example

# You have previously prepared a set of Russian tweets for classification.
# Of the 20,000 tweets, you have filtered to tweets with an account_type of Left or Right, and selected the first 2000 tweets of each. 
# You have already tokenized the tweets into words, removed stop words, and performed stemming.
# Furthermore, you converted word counts into a document-term matrix with TFIDF values for weights and saved this matrix as: left_right_matrix_small.

# You will use this matrix to predict whether a tweet was generated from a left-leaning tweet bot, or a right-leaning tweet bot. 
# The labels can be found in the vector, left_right_labels.

library(randomForest)

# Create train/test split
set.seed(1111)
sample_size <- floor(0.75 * nrow(left_right_matrix_small))
train_ind <- sample(nrow(left_right_matrix_small), size = sample_size)
train <- left_right_matrix_small[train_ind, ]
test <- left_right_matrix_small[-train_ind, ]

# Create a random forest classifier
rfc <- randomForest(x = as.data.frame(as.matrix(train)), 
                    y = left_right_labels[train_ind],
                    nTree = 50)
# Print the results
print(rfc)

####
## TFIDF tibble vs dtm

# TFIDF can be used for document similarity, text classification, and tasks. 
# Consider the tibble, left_right_tfidf, and the document-term matrix, left_right_matrix. 
# Both have been loaded into the console.

# Which of the following statements is true?
  
# A: The tibble contains one row per document and a column for each word used in all of the text.
# B: The tibble contains the word counts, tf, idf, and tfidf weights for each word in each document document.
# C: The tibble and the matrix have the same number of rows.
# D: The columns of the document-term matrix can be used in classification models.

# B & D right answers

#####
## Topic modelling:  https://www.tidytextmining.com/topicmodeling.html
#####

# Collection of texts is likely to be made up an collection of topics
# Sports stories: scores, player gossip, team news, 
# Weather in Zambia: ??, ??, ... netusime o tejto temu 

# LDA = Latent Dirichlet allocation
#     - 2 basic principles: 
  
# 1) documents are mixtures of topic. E.g. Star pitcher traded - popisane gamma koeficientom
      # Team news = 70 %
      # Player Gossip = 30 % 

# 2) topics are  mixtures of words  - popisane beta koeficientom
      # Team News: trade, pitcher, move, new
      # Player Gossip: angry, change, money

# In order to perfrom LDA() we need a document-term matrix with term frequency weights  (tm:weightTf) - not TfIdf

# LDA is a mathematical method for estimating both of these at the same time: 
#   finding the mixture of words that is associated with each topic, 
#   while also determining the mixture of topics that describes each document. 
# There are a number of existing implementations of this algorithm, and we’ll explore one of them in depth.

#### 
## Word-topic probabilities - beta parameter
####

# We introduced the tidy() method, originally from the broom package (Robinson 2017), for tidying model objects. 
# The tidytext package provides this method for extracting the per-topic-per-word probabilities, called “beta”,from the model.

library(tidytext)
library(topicmodels)
data("AssociatedPress")

# set a seed so that the output of the model is predictable
ap_lda <- LDA(AssociatedPress, k = 2, control = list(seed = 1234))
ap_lda
ap_topics <- tidy(ap_lda, matrix = "beta")
ap_topics

# Notice that this has turned the model into a one-topic-per-term-per-row format. 
# For each combination, the model computes the probability of that term being generated from that topic. 
# For example, the term “aaron” has a 1.686917x 10-12 probability of being generated from topic 1, but a 3.89×10-5 probability of being generated from topic 2.

library(ggplot2)
library(dplyr)

ap_top_terms <- ap_topics %>%
  group_by(topic) %>%
  top_n(10, beta) %>%
  ungroup() %>%
  arrange(topic, -beta)


ap_top_terms %>%
  mutate(term = reorder_within(term, beta, topic)) %>%
  ggplot(aes(term, beta, fill = factor(topic))) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ topic, scales = "free") +
  coord_flip() +
  scale_x_reordered()

#### 
## Document-topic probabilities - gamm parameter
####

# Popisuje, ake su distribucie úre jednotlive dokumenty / vety vzhladom na topic. 

# Besides estimating each topic as a mixture of words, LDA also models each document as a mixture of topics. 
# We can examine the per-document-per-topic probabilities, called “gamma”, with the matrix = "gamma" argument to tidy().
# Each of these values is an estimated proportion of words from that document that are generated from that topic

ap_documents <- tidy(ap_lda, matrix = "gamma")
ap_documents

ap_documents %>% filter(document == 23)

###
# LDA practice
##

# You are interested in the common themes surrounding the character Napoleon in your favorite new book, Animal Farm.
# Napoleon is a Pig who convinces his fellow comrades to overthrow their human leaders. 
# He also eventually becomes the new leader of Animal Farm.

# You have extracted all of the sentences that mention Napoleon's name, pig_sentences, 
#   and created tokenized version of these sentences with stop words removed and stemming completed, pig_tokens. 
# Complete LDA on these sentences and review the top words associated with some of the topics.

library(topicmodels)
# Perform Topic Modeling
sentence_lda <-
  LDA(pig_matrix, k = 10, method = 'Gibbs', control = list(seed = 1111))
# Extract the beta matrix 
sentence_betas <- tidy(sentence_lda, matrix = "beta")

# Topic #2
sentence_betas %>%
  filter(topic == 2) %>%
  arrange(-beta)

# Topic #3
sentence_betas %>%
  filter(topic == 3) %>%
  arrange(-beta)

# Poznamka: pre kazdy topic mame bety v sucte = 1

# Well done. Notice the differences in words for topic 2 and topic 3. 
# Each topic should be made up of mostly different words, otherwise all topics would end up being the same. 
# We will give meaning to these differences in the next lesson


####
## Assigning topics to documents
###

# Creating LDA models are useless unless you can interpret and use the results. 
# You have been given the results of running an LDA model, sentence_lda on a set of sentences, pig_sentences. 
# You need to explore both the beta, top words by topic, and the gamma, top topics per document, matrices to fully understand the results of any LDA analysis.

# Given what you know about these two matrices, extract the results for a specific topic and see if the output matches expectations.

# Extract the beta and gamma matrices
sentence_betas <- tidy(sentence_lda, matrix = "beta")
sentence_gammas <- tidy(sentence_lda, matrix = "gamma")

# Explore Topic 5 Betas
sentence_betas %>%
  filter(topic == 5) %>%
  arrange(-beta)

# Explore Topic 5 Gammas
sentence_gammas %>%
  filter(topic == 5) %>%
  arrange(-gamma)

## How to use results?
## How to choose number opf topics? # perplexity / other metrics ... solution that works for your situatuon

### Perplexity:  https://en.wikipedia.org/wiki/Perplexity
      # measure how well a probability model fits new data
      # the lower the better
      # used to compare models:  * in LDA parameter tuning....  * selecting number of topics

####
# Testing perplexity
####

# You have been given a dataset full of tweets that were sent by tweet bots during the 2016 US election. 
# Your boss has identified two different account types of interest, Left and Right. 
# Your boss has asked you to perform topic modeling on the tweets from Right tweet bots. 
# Furthermore, your boss is hoping to summarize the content of these tweets with topic modeling. 
# Perform topic modeling on 5, 15, and 50 topics to determine a general idea of how many topics are contained in the data.

library(topicmodels)
# Setup train and test data
sample_size <- floor(0.90 * nrow(right_matrix))
set.seed(1111)
train_ind <- sample(nrow(right_matrix), size = sample_size)
train <- right_matrix[train_ind, ]
test <- right_matrix[-train_ind, ]

# Peform topic modeling 
lda_model <- LDA(train, k = 5, method = "Gibbs",   # k = 5, 15, 50
                 control = list(seed = 1111))
# Train
perplexity(lda_model, newdata = train) 

# Test
perplexity(lda_model, newdata = test) 

####
## Reviewing LDA results
####

# You have developed a topic model, napoleon_model, with 5 topics for the sentences from the book Animal Farm that reference the main character Napoleon. 
# You have had 5 local authors review the top words and top sentences for each topic and they have provided you with themes for each topic.

# To finalize your results, prepare some summary statistics about the topics. 
# You will present these summary values along with the themes to your boss for review.

# Extract the gamma matrix 
gamma_values <- tidy(napoleon_model, matrix = "gamma")

# Create grouped gamma tibble
grouped_gammas <- gamma_values %>%
  group_by(document) %>%
  arrange(desc(gamma)) %>%
  slice(1) %>%
  group_by(topic)

# Count (tally) by topic
grouped_gammas %>% 
  tally(topic, sort=TRUE)

# Average topic weight for top topic for each sentence
grouped_gammas %>% 
  summarise(avg=mean(gamma)) %>%
  arrange(desc(avg))

########################
## SENTIMENT ANALYSIS ##
########################

# https://www.tidytextmining.com/sentiment.html

library(tidytext)
tidytext::sentiments ## rozdielne ako ukazuju na datacampe

# The tidytext package contains several sentiment lexicons. Three general-purpose lexicons are:
# 1) AFINN from Finn Arup Nielsen,
# 2) bing from Bing Liu and collaborators, and
# 3) nrc from Saif Mohammad and Peter Turney.

# All three of these lexicons are based on unigrams, i.e., single words. 
# These lexicons contain many English words and the words are assigned scores 
# for positive/negative sentiment, and also possibly emotions like joy, anger, sadness, and so forth. 

# The nrc lexicon categorizes words in a binary fashion (“yes”/“no”) into categories of positive, negative, anger, 
    # anticipation, disgust, fear, joy, sadness, surprise, and trust. 
# The bing lexicon categorizes words in a binary fashion into positive and negative categories. 
# The AFINN lexicon assigns words with a score that runs between -5 and 5, with negative scores indicating negative sentiment and positive scores indicating positive sentiment. 

# All of this information is tabulated in the sentiments dataset, and tidytext provides a function get_sentiments() to get specific sentiment lexicons without the columns that are not used in that lexicon.

get_sentiments("afinn")
get_sentiments("bing")
get_sentiments("nrc")

# Not every English word is in the lexicons because many English words are pretty neutral. 
# It is important to keep in mind that these methods do not take into account qualifiers before a word, 
#     such as in “no good” or “not true”; a lexicon-based method like this is based on unigrams only. 
# For many kinds of text (like the narrative examples below), there are not sustained sections of sarcasm or negated text, so this is not an important effect

### BING lexikon: 
# Print the lexicon
get_sentiments("bing")

# Count the different sentiment types
get_sentiments("bing") %>%
  count(sentiment) %>%
  arrange(desc(n))

### NRC lexikon: 
# Print the lexicon
get_sentiments("nrc")

# Count the different sentiment types
get_sentiments("nrc") %>%
  count(sentiment) %>%
  arrange(desc(n))

### AFINN lexikon: 
# Print the lexicon
get_sentiments("afinn")

# Count how many times each score was used
get_sentiments("afinn") %>%
  count(score) %>%
  arrange(desc(n))

####
# Sentiment scores

# In the book Animal Farm, three main pigs are responsible for the events of the book: Napoleon, Snowball, and Squealer. 
# Throughout the book they are spreading thoughts of rebellion and encouraging the other animals to take over the farm from Mr. Jones - the owner of the farm.

# Using the sentences that mention each pig, determine which character has the most negative sentiment
# associated with them. The sentences tibble contains a tibble of the sentences from the book Animal Farm.

# Print the overall sentiment associated with each pig's sentences
for(name in c("napoleon", "snowball", "squealer")) {
  # Filter to the sentences mentioning the pig
  pig_sentences <- sentences[grepl(pattern = name, sentences$sentence), ]
  # Tokenize the text
  napoleon_tokens <- pig_sentences %>%
    unnest_tokens(output = "word", token = "words", input = sentence) %>%
    anti_join(stop_words)
  # Use afinn to find the overall sentiment score
  result <- napoleon_tokens %>% 
    inner_join(get_sentiments("afinn")) %>%
    summarise(sentiment = sum(score))
  # Print the result
  print(paste0(name, ": ", result$sentiment))
}

####  
## Sentiment and emotion

# Within the sentiments dataset, the lexicon nrc contains a dictionary of words and an emotion associated with that word. 
# Emotions such as joy, trust, anticipation, and others are found within this dataset.

# In the Russian tweet bot dataset you have been exploring, you have looked at tweets sent out by both a left- and a right-leaning tweet bot. 
# Explore the contents of the tweets sent by the left-leaning (democratic) tweet bot by using the nrc lexicon. 
# The left tweets, left, have been tokenized into words, with stop-words removed.

left_tokens <- left %>%
  unnest_tokens(output = "word", token = "words", input = content) %>%
  anti_join(stop_words)

# Dictionaries 
anticipation <- get_sentiments("nrc") %>% 
  filter(sentiment == "anticipation")
joy <- get_sentiments("nrc") %>% 
  filter(sentiment == "joy")

# Print top words for Anticipation and Joy
left_tokens %>%
  inner_join(anticipation, by = "word") %>%
  count(word, sort = TRUE)
left_tokens %>%
  inner_join(joy, by = "word") %>%
  count(word, sort = TRUE)

#####
# WORD EMBEDDINGS
###

## word2vec - 
#  * 2013 developed by google
#  * represents words as a large vector space
#  * captures multiple similarities between words 
#  * words of similarly meanings are closer within the space 

# Do word2vec in R:
library(h2o) 
h2o.init()
h2o_object = as.h2O(animal_farm)

# Tokenize using h2o:
words <- h2o.tokenize(h2o_object$text_column, '\\\\W+"')
words <- h2o.tolower(words)
words <- word[is.na(words) || (!words %in% stop_words$word), ]

word2vec_model <- h2O.word2vec(words, min_word_freq = 5, epochs = 5) # epochs - number of training iterations to run (viac textu -> viac epochs)

# word synonyms:
h2o.findSynonyms(w2v.model, "animal")

####
## h2o practice

# There are several machine learning libraries available in R. 
# However, the h2o library is easy to use and offers a word2vec implementation. 
# h2o can also be used for several other machine learning tasks. 
# In order to use the h2o library however, you need to take additional pre-processing steps with your data. 
# You have a dataset called left_right which contains tweets that were auto-tweeted during the 2016 US election campaign.
# Instead of preparing your data for other text analysis techniques, prepare this dataset for use with the h2o library.


###########
## Moje skusanie - testovanie nevysvetlenych detailov-

# https://www.kaggle.com/kernels/scriptcontent/25728798/download

library(dplyr)
library(tidytext)

skuska <- tibble(lines = 1:2, text = c("Ja som som som  Nikolas", "Ja som Viki"))

stopwords::stopwords(language = "sk", source = "stopwords-iso")

#https://www.rdocumentation.org/packages/tm/versions/0.7-6/topics/weightTfIdf
#https://www.rdocumentation.org/packages/tm/versions/0.7-6/topics/weightTf

# Nezhody: bind_tf_idf pocita idf ako ln(N/#vyskytov), kdezto tm:weightTfIdf pocita idf cast cez log(N/#vyskytov, base = 2)
# bind_tf_idf berie tf = podiel slova v danom dokumente a nie POCET kolkokrat sa nachadza v dokumente- tm::weightTf
skuska %>% 
  unnest_tokens(word, text) %>%
  count(lines, word) %>%
  bind_tf_idf(word, lines, n = n)

skuska %>% 
  unnest_tokens(word, text) %>%
  count(lines, word) %>%
  ungroup() %>%
  cast_dtm(document = lines, term = word, value = n, weighting = tm::weightTfIdf) %>%
  as.matrix()
# Non sparse / sparse: 2/6 = 75 %  

# !! Sparse sa pocita na zaklade podielu 0 (avsak cisla sa beru prave vahy urcene!)
#    Ak pouzijeme tf-idf, tak slovo ktore sa nachadza v kazdom dokumente ma tiez vahu 0 a spada to potom do SPARSE hodnoty
# !! 

skuska %>% 
  unnest_tokens(word, text) %>%
  count(lines, word) %>%
  ungroup() %>%
  cast_dtm(document = lines, term = word, value = n, weighting = tm::weightTf) %>%
  as.matrix()
# Non sparse / sparse: 6/2 = 25 %  




