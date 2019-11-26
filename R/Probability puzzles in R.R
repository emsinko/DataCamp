## Chapter 1

# ================================

# Set seed to 1
set.seed(1)

# Write a function to roll k dice
roll_dice <- function(k){
  all_rolls <- sample(c(1,2,3,4,5,6), 
                      size = k, 
                      replace = TRUE)
  final_answer <- sum(all_rolls)
  return(final_answer)
}

# Run the function to roll five dice
roll_dice(5)

# =================================


# Initialize a vector to store the output
output <- rep(NA, 10000)

# Loop for 10000 iterations
for(i in 1:10000){
  # Fill in the output vector with the result from rolling two dice
  output[i] <- roll_dice(k = 2)
}

# =================================

# Birthday problem: n=50

set.seed(1)
n <- 50
match <- 0

# Simulate 10000 rooms and check for matches in each room
for(i in 1:10000){
  birthdays <- sample(1:365, n, replace = TRUE)
  if(length(unique(birthdays)) < n){
    match <- match + 1
  } 
}

# Calculate the estimated probability of a match and print it
p_match <- match / 10000
print(p_match)

# Implemented function:

# Calculate the probability of a match for a room size of 50
pbirthday(n = 50)

## Plot of probability depending on N:

# Define the vector of sample sizes
room_sizes <- 1:50

# Run the pbirthday function within sapply on the vector of sample sizes
match_probs <- sapply(room_sizes, pbirthday)

# Create the plot
plot(match_probs ~ room_sizes)

# =================================
# Monty Hall

set.seed(1)
doors <- c(1,2,3)

# Randomly select one of the doors to have the prize
prize <- sample(x = doors, size = 1)
initial_choice <- 1

# Check if the initial choice equals the prize
if(prize == initial_choice){
  print("The initial choice was correct!")
}

print(prize)

# Probability of win - if we dont change our first quess

set.seed(1)
doors <- c(1,2,3)

# Define counter
win_count <- 0

# Run 10000 iterations of the game
for(i in 1:10000){
  prize <- sample(x = doors, size = 1)
  initial_choice <- 1
  if(initial_choice == prize){
    win_count = win_count + 1
  }
}

# Print the answer
print(win_count / 10000)

## Probability with changing our fits choice

reveal_door <- function(doors, prize, initial_choice){
  if(prize == initial_choice){
    # Sample at random from the two remaining doors
    reveal <- sample(x = doors[-initial_choice], size = 1)
  } else {
    
    # When the prize and initial choice are different, reveal the only remaining door 
    reveal <- doors[-c(initial_choice, prize)]
  }  
}


set.seed(1)
prize <- sample(doors,1)
initial_choice <- 1

# Use the reveal_door function to do the reveal
reveal <- reveal_door(doors, prize, initial_choice)

# Switch to the remaining door
final_choice <- doors[-c(initial_choice, reveal)]
print(final_choice)

# Check whether the final choice equals the prize
if(final_choice == prize){
  print("The final choice is correct!")
}

# Probability simulation 
# Initialize the win counter
win_count <- 0

for(i in 1:10000){
  prize <- sample(doors,1)
  initial_choice <- 1
  reveal <- reveal_door(doors, prize, initial_choice)
  final_choice <- doors[-c(initial_choice, reveal)]
  if(final_choice == prize){
    
    # Increment the win counter
    win_count = win_count + 1
  }
}

# Print the estimated probability of winning
print(win_count / 10000)

# =================================
# Yahtzee - rolling 5 dices

# Calculate the size of the sample space
s_space <- 6^5

# Calculate the probability of a Yahtzee
p_yahtzee <- 6 / s_space

# Print the answer
print(p_yahtzee)

##
# Probability of a large straight
# A large straight occurs when the five dice land on consecutive denominations, specifically either {1,2,3,4,5} or {2,3,4,5,6}.
# Let's calculate the probability of a "large straight" in a single roll of the five dice.


s_space <- 6^5

# Calculate the probabilities
p_12345 <- factorial(5) * choose(5,5) * 1 / s_space
p_23456 <- factorial(5) * choose(5,5) * 1 / s_space
p_large_straight <- p_12345 + p_23456

# Print the large straight probability
print(p_large_straight)

##
# Probability of a full house
# A full house occurs when three of the dice are of one denomination, and the remaining two dice are of another denomination. 
# In other words, it consists of a "set of three" and "a pair." An example is {2,2,2,5,5}.
# Let's calculate the probability of a "full house" in a single roll of the five dice.

s_space <- 6^5

# Calculate the number of denominations possible
n_denom <- 6 * 5 

# Calculate the number of ways to form the groups
n_groupings <- factorial(5) / (factorial(2) * factorial(3))

# Calculate the total number of full houses
n_full_house <- n_denom * n_groupings

# Calculate and print the answer
p_full_house <- n_full_house / s_space
print(p_full_house)

###
# Settlers of Catan

set.seed(1)

# Simulate one game (60 rolls) and store the result
rolls <- replicate(60, roll_dice(2))

# Display the result
table(rolls)

### Pravdepodobnost ze viac ako 2 krat padne sucet dvoch kociek 2 alebo viac ako 2 krat padne sucet 12

set.seed(1)
counter <- 0

for(i in 1:10000){
  # Roll two dice 60 times
  rolls <- replicate(60, roll_dice(2))
  
  # Check whether 2 or 12 was rolled more than twice
  if(sum(rolls == 2) > 2 | sum(rolls == 12) > 2){
    counter <- counter + 1
  }  
}

# Print the answer
print(counter / 10000)

# =================================
# Craps - rolling 2 dices

roll_after_point <- function(point){
  new_roll <- 0
  # Roll until either a 7 or the point is rolled 
  while( (new_roll != 7) &  (new_roll != point)){
    new_roll <- roll_dice(2)
    if(new_roll == 7){
      won <- FALSE      
    }
    # Check whether the new roll gives a win
    if(new_roll == point){
      won <- TRUE
    }
  }
  return(won)
}


evaluate_first_roll <- function(roll){
  # Check whether the first roll gives an immediate win
  if(roll %in% c(7, 11)){
    won <- TRUE
  }
  # Check whether the first roll gives an immediate loss
  if(roll %in% c(2, 3, 12)){
    won <- FALSE
  }
  if(roll %in% c(4,5,6,8,9,10) ){
    # Roll until the point or a 7 is rolled and store the win/lose outcome
    won <- roll_after_point(roll)
  }  
  return(won)
}

## Probability of winning the pass line bet

# Now, we'll use the functions that we created previously to simulate 10,000 games of Craps to estimate the 
# probability of winning the pass line bet. 
# The roll_dice and evaluate_first_roll functions have been preloaded for you, to be used in this exercise.

set.seed(1)
won <- rep(NA, 10000)

for(i in 1:10000){
  # Shooter's first roll
  roll <- roll_dice(2)
  
  # Determine result and store it
  won[i] <- evaluate_first_roll(roll)
}

sum(won)/10000

# =================================
# Inspired from the WEB


### Algebra

is_factorable <- function(a,b,c){
  # Check whether solutions are imaginary
  if(b^2 - 4*a*c  < 0){
    return(FALSE)
    # Designate when the next section should run
  } else {
    sqrt_discriminant <- sqrt(b^2 - 4*a*c) 
    # return TRUE if quadratic is factorable
    return(sqrt_discriminant == round(sqrt_discriminant))    
  }
}

counter <- 0

# Nested for loop
for(a in 1:100){
  for(b in 1:100){
    for(c in 1:100){
      # Check whether factorable
      if(is_factorable(a,b,c)){
        counter <- counter + 1
      }
    }
  }
}

print(counter / 100^3)

# =================================
# Iphone Passcodes

# Suppose that you pick up an iPhone with four smudge marks on it, at the locations of 3, 4, 5 and 9.
# The actual value of the passcode has been preloaded as a variable called passcode.

counter <- 0

# Store known values 
values <- c(3,4,5,9)

for(i in 1:10000){
  # Create the guess
  guess <- sample(values, size = 4, replace = FALSE)
  # Check condition 
  if(identical(guess,  passcode)){
    counter <- counter + 1
  }
}

print(counter/10000)



## Three known values
# Now, we will simulate the probability of correctly guessing when the passcode consists of three distinct digits where one of the values is repeated.
# Here, suppose that the smudge marks are at the values of 2, 4, and 7. 
# One of these values will be repeated in the passcode, but we do not know which one, nor where the repeated value is within the passcode.

counter <- 0
# Store known values
unique_values <- c(2,4,7)

for(i in 1:10000){
  # Pick repeated value
  all_values <- c(unique_values, sample(unique_values,1))
  # Make guess
  guess <- sample(all_values, 4, replace = FALSE)
  if(identical(passcode, guess)){
    counter <- counter + 1
  }
}

print(counter / 10000)  # 0.0282

## ========================
# Simulate sign errors: constant probabilities

# Let us simulate our math problems and sign errors.
# Here, we will assume that the math problem has just 3 steps, and each step has the same probability of a sign switch. We will simulate the completion of the problem with:
# a 0.10 probability of making a sign switch on each step.
# a 0.45 probability of making a sign switch on each step.

set.seed(1)

# Run 10000 iterations, 0.1 sign switch probability
switch_a <- rbinom(10000, size = 3, prob = 0.1)

# Calculate probability of correct answer
mean(round(switch_a / 2) == switch_a / 2)

# Run 10000 iterations, 0.45 sign switch probability
switch_b <- rbinom(10000, size = 3, prob = 0.45)

# Calculate probability of correct answer
mean(round(switch_b / 2) == switch_b / 2)

## 
# Simulate sign errors: changing probabilities
# In the previous exercise, each simulation had a fixed probability of a sign error at each step.
# In the original question as posed, the probability of a sign error can be different at each step; the only requirement is that each probability is less than 0.50.
# Here, let us simulate a math problem with 2 steps in which the probability of a switch on each step is 0.49 and 0.01, meaning that there is a high probability of becoming incorrect on the first step and a low probability of switching back.

set.seed(1)
counter <- 0

for(i in 1:10000){
  # Simulate switches
  each_switch <- sapply(X = c(0.49, 0.01), FUN = rbinom, n = 1, size = 1)
  # Count switches
  num_switches <- sum(each_switch == 1)
  # Check solution
  if(num_switches / 2 == round(num_switches / 2)){
    counter <- counter + 1
  }
}

print(counter/10000)

## ========================
# Texas Hold'em

# Outs: Cards that improve hand from losing to winning !!!

# Calculate expected value with one card to come
# Suppose that there is just one card left to come, and you know that 8 of the 46 remaining cards in the deck will give you a win (recall that these 8 cards are known as outs). Otherwise, you will lose.
# There is currently $50 in the pot, and you are currently facing a $10 bet from your opponent. 
# Assuming no further betting, then, from this point, if you call the bet and win, your profit is $50. 
# If you call the bet and lose, your profit is negative $10.

p_win <- 8 / 46
curr_pot <- 50
bet <- 10

# Define vector of probabilities
probs <- c(p_win, 1 - p_win)

# Define vector of values
values <- c(curr_pot, - bet)

# Calculate expected value
sum(probs * values)



### Two cards to come

# Let's now consider the point at which two cards will still come. 
# Here, we will find the probability of winning for any number of outs.
# At this point, there are 3 cards face up, and 2 in your hand. 
# With 52 total cards in the deck, this leaves 47 unseen cards, so the denominator is choose(47,2) to represent the total number of combinations for the two cards to come.
# An often-used approximation among poker players is that the win probability is equal to 0,04 * outs. How good is this approximation?

# OUTS: pocet kariet, ktore vdaka ktorym vyhrame
outs <- c(0:25)

# Calculate probability of not winning
p_no_outs <- choose(47 - outs, 2) / choose(47, 2)

# Calculate probability of winning
p_win <- 1 - p_no_outs

print(p_win)

### Two consecutive years

#First, let us see how common it is for someone to cash two years in a row.
# Here, we will assume just 60 entrants, with cash awarded to the top 6. 
# Each player is represented by a number from 1 through 60.

players <- c(1:60)
count <- 0

for(i in 1:10000){
  cash_year1 <- sample(players, 6)
  cash_year2 <- sample(players, 6)
  # Find those who cashed both years
  cash_both <- intersect(cash_year1,cash_year2) 
  # Check whether anyone cashed both years
  if(length(cash_both) > 0){
    count <- count + 1
  }  
}

print(count/10000)

## Function to evaluate set of five years

check_for_five <- function(cashed){
  # Find intersection of five years
  all_five <- Reduce(intersect, list(cashed[,1],cashed[,2],cashed[,3],cashed[,4],cashed[,5]))
  # Check intersection
  if(length(all_five) > 0 ){
    return(TRUE)
    # Specify when to return FALSE
  } else {
    return(FALSE)
  }
}

### Simulate probability for a given set of five years

# Now, let's use the check_for_five function from the previous question 
# to simulate 10000 iterations of five years to get an idea of 
# how rare Ronnie Bardah's feat of five consecutive cashes was.


players <- c(1:6000)
count <- 0

for(i in 1:10000){
  # Create matrix of cashing players
  cashes <- replicate(5, sample(players, 600, replace = FALSE))
  # Check for five time winners
  if(check_for_five(cashes)){
    count <- count + 1
  }
}

print(count/10000)

###
# One round of von Neumann Poker
# Let us simulate one round of von Neumann poker, in which each player of two players, A and B, receives a value drawn at random from a uniform distribution.

# Here, we will ignore the betting aspect, and just check which player wins.

# Generate values for both players
A <- runif(1)
B <- runif(1)

# Check winner
if(A > B){
  print("Player A wins")
} else {
  print("Player B wins")
}

print(A)
print(B)


### Function to simulate one round with betting

#Now let us write a function that simulates one round under the von Neumann model, with betting incorporated.

# Player B will observe their value, and decide whether to bet $1 or not. 
# If the decision to bet is made, then the two players compare values and the higher value wins. 
# If Player B decides not to bet, then no money is won or lost by either player.

# Here, we will assume that Player B has a fixed strategy: if their value is above a certain cutoff, then they 
# will bet. This cutoff, bet_cutoff, will be the argument to the function.

one_round <- function(bet_cutoff){
  a <- runif(n = 1)
  b <- runif(n = 1)
  # Fill in betting condition
  if(b > bet_cutoff){
    # Return result of bet
    return(ifelse(b > a, 1, -1))
  } else {
    return(0)
  }  
}

## Simulate many iterations of von Neumann model
# Now, let us use the one_round function that we wrote in the previous exercise 
# to simulate 10000 iterations, when Player B's strategy is to bet if and only if their 
# value is greater than 0.5. Our goal will be to estimate Player B's expected value under this strategy.

b_win <- rep(NA, 10000)

for(i in 1:10000){
  # Run one and store result
  b_win[i] <- one_round(0.5)
}

# Print expected value
mean(b_win)

