####################################
#### INTRODUCTION TO TIDYVERSE  ####
####################################

# Load the gapminder adn dplyr package
library(gapminder)
library(dplyr)

# Look at the gapminder dataset
gapminder

# Filter the gapminder dataset for the year 1957
gapminder %>% filter(year == 1957)

# Filter for China in 2002
gapminder %>% filter(country == "China", year == 2002)

  # Sort in ascending order of lifeExp
gapminder %>% arrange(lifeExp)

# Sort in descending order of lifeExp
gapminder %>% arrange(desc(lifeExp))

# Filter for the year 1957, then arrange in descending order of population
gapminder %>% filter(year == 1957) %>% arrange(desc(pop))

# Use mutate to change lifeExp to be in months
gapminder %>% mutate(lifeExp = lifeExp * 12)

# Use mutate to create a new column called lifeExpMonths
gapminder %>% mutate(lifeExpMonths = lifeExp * 12)

# Filter, mutate, and arrange the gapminder dataset
gapminder %>% filter(year == 2007) %>% mutate(lifeExpMonths = 12*lifeExp) %>% arrange(desc(lifeExpMonths))

library(ggplot2)
# Create gapminder_1952
gapminder_1952 <- gapminder %>% filter(year == 1952)


# Change to put pop on the x-axis and gdpPercap on the y-axis
ggplot(gapminder_1952, aes(y = gdpPercap, x = pop)) +
  geom_point()


# Create a scatter plot with pop on the x-axis and lifeExp on the y-axis
gapminder_1952 %>% ggplot(aes(x=pop, y=lifeExp))+geom_point()

# Change this plot to put the x-axis on a log scale
ggplot(gapminder_1952, aes(x = pop, y = lifeExp)) +
  geom_point() + scale_x_log10()



# Scatter plot comparing pop and gdpPercap, with both axes on a log scale
gapminder_1952 %>% ggplot(aes(x = pop,y = gdpPercap)) + geom_point()
gapminder_1952 %>% ggplot(aes(x = pop,y = gdpPercap)) + geom_point() + scale_x_log10() + scale_y_log10()
?scale_x_log10

data.frame(x=1:10, y = 2^(1:10)) %>%  ggplot(aes(x = x,y = y)) + geom_point() 
data.frame(x=1:10, y = 2^(1:10)) %>%  ggplot(aes(x = x,y = y)) + geom_point() + scale_y_log10() 
data.frame(x=1:10, y = 2^(1:10)) %>%  ggplot(aes(x = x,y = y)) + geom_point() + scale_x_log10() + scale_y_log10()


# Scatter plot comparing pop and lifeExp, with color representing continent
gapminder_1952 %>% ggplot(aes(x=pop,y=lifeExp,col=continent)) + geom_point()+ scale_x_log10()

# Scatter plot comparing pop and lifeExp, faceted by continent
gapminder_1952 %>% ggplot(aes(x=pop,y=lifeExp)) + geom_point()+ scale_x_log10() + facet_wrap(~continent)

# Scatter plot comparing gdpPercap and lifeExp, with color representing continent
# and size representing population, faceted by year
gapminder  %>% ggplot(aes(x=gdpPercap,y=lifeExp, color = continent, size = pop)) + geom_point()+ scale_x_log10() + facet_wrap(~year)

# Summarize to find the median life expectancy
gapminder %>% summarize(medianLifeExp = median(lifeExp))
  
# Filter for 1957 then summarize the median life expectancy
gapminder %>% filter(year == 1957) %>% summarize(medianLifeExp = median(lifeExp))

# Filter for 1957 then summarize the median life expectancy and the maximum GDP per capita
gapminder %>% filter(year == 1957) %>% summarize(medianLifeExp = median(lifeExp), maxGdpPercap = max(gdpPercap))

# Find median life expectancy and maximum GDP per capita in each year
gapminder %>% group_by(year) %>% summarize(medianLifeExp = median(lifeExp), maxGdpPercap = max(gdpPercap))

# Find median life expectancy and maximum GDP per capita in each continent in 1957
gapminder %>% filter(year == 1957) %>% group_by(continent) %>% summarize(medianLifeExp = median(lifeExp), maxGdpPercap = max(gdpPercap))

# Find median life expectancy and maximum GDP per capita in each year/continent combination
gapminder %>% group_by(continent, year) %>% summarize(medianLifeExp = median(lifeExp), maxGdpPercap = max(gdpPercap))


by_year <- gapminder %>%
  group_by(year) %>%
  summarize(medianLifeExp = median(lifeExp),
            maxGdpPercap = max(gdpPercap))

# Create a scatter plot showing the change in medianLifeExp over time
by_year %>% ggplot(aes(x=year, y=medianLifeExp)) + geom_point() + expand_limits(y=0)

########

# Summarize the median GDP and median life expectancy per continent in 2007
by_continent_2007 <- gapminder %>% filter(year == 2007) %>% group_by(continent) %>% summarize(medianGdpPercap = median(gdpPercap), medianLifeExp = median(lifeExp))

# Use a scatter plot to compare the median GDP and median life expectancy
by_continent_2007 %>% ggplot(aes(x=medianGdpPercap, y=medianLifeExp, color = continent)) + geom_point() + expand_limits(y=0)

# Summarize medianGdpPercap within each continent within each year: by_year_continent
  by_year_continent <- gapminder %>% 
    group_by(year, continent) %>%
    summarize(medianGdpPercap = median(gdpPercap))

# Plot the change in medianGdpPercap in each continent over time
by_year_continent %>% ggplot(aes(color = continent, x=year, y=medianGdpPercap)) + geom_point() + expand_limits(y=0)




# Summarize the median gdpPercap by year, then save it as by_year
by_year <- gapminder %>% 
  group_by(year) %>%
  summarize(medianGdpPercap = median(gdpPercap))

# Create a line plot showing the change in medianGdpPercap over time
by_year %>% ggplot(aes(x=year,y=medianGdpPercap)) + geom_line() + expand_limits(y=0)

# Summarize the median gdpPercap by year & continent, save as by_year_continent
by_year_continent <- gapminder %>% 
  group_by(year, continent) %>%
  summarize(medianGdpPercap = median(gdpPercap))

# Create a line plot showing the change in medianGdpPercap by continent over time
by_year_continent %>% ggplot(aes(x = year, y = medianGdpPercap, color = continent)) + geom_line() + expand_limits(y=0)


# Summarize the median gdpPercap by year and continent in 1952
by_continent <- gapminder %>% 
  filter(year == 1952) %>%
  group_by(continent) %>%
  summarize(medianGdpPercap = median(gdpPercap))

# Create a bar plot showing medianGdp by continent
by_continent %>% ggplot(aes(x=continent,y=medianGdpPercap)) + geom_col()