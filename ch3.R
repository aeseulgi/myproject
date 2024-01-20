install.packages('nycflights13')
library(nycflights13)
library(tidyverse)

head(flights)
dim(flights)

#FILTER
filter(flights, month == 1, day == 1)
jan1 <- filter(flights, month == 1, day == 1)

#Comparison operators
filter(flights, month = 1) # the easiest mistake

sqrt(2) ^ 2 == 2 # floating point numbers
1 / 49 * 49 == 1

near(sqrt(2) ^ 2, 2)
near(1 / 49 * 49, 1)

flightsNovDec <- filter(flights, month == 11 | month == 12)
head(flightsNovDec)

filter(flights, !(arr_delay > 120 | dep_delay > 120))
filter(flights, arr_delay <= 120, dep_delay <= 120)

# Missing values

NA > 5
10 == NA
NA + 10
NA / 2
NA == NA

x <- NA
y <- NA
x == y

is.na(x)

df <- tibble(x = c(1, NA, 3))
filter(df, x > 1)
filter(df, is.na(x) | x > 1)

#ARRANGE ... changes the order of rows, sorting
tmp <- arrange(flights, year, month, day)
head(tmp)

tmp <- arrange(flights, desc(arr_delay)) #sort in descending order
head(tmp)

df <- tibble(x = c(5, 2, NA))
arrange(df, x)
arrange(df, desc(x))
# NA are always sorted at the end.

#SELECT ... select columns
tmp <- select(flights, year, month, day)
head(tmp)
tmp <- select(flights, year:day)
head(tmp)
tmp <- select(flights, -(year:day))
head(tmp)
tmp <- select(flights, seq(1, 10, by = 2))
head(tmp)
tmp <- select(flights, time_hour, air_time, everything())
head(tmp)
tmp <- rename(flights, tail_num = tailnum) #rename a variable
head(tmp)

'''
start_with("abc"): matches names that begin with "abc"
ends_with("xyz"): matches names that end with "xyz"
contains("ijk"): matches names that contain "ijk"
matches("(,)\\1"): selects variables that match a regular expression
num_range("x", 1:3): matches x1, x2 and x3
'''

#MUTATE ... add new variables
flights_sml <-
  select(flights, year:day, ends_with("delay"), distance, air_time)
head(flights_sml)

tmp <- mutate(flights_sml, gain = arr_delay - dep_delay,
              speed = distance / air_time * 60)
head(tmp)

tmp <- mutate(flights_sml, gain = arr_delay - dep_delay,
              hours = air_time / 60, gain_per_hour = gain / hours)
head(tmp)

#transmute: only keep the new variables
tmp <- transmute(flights, gain = arr_delay - dep_delay,
                 hours = air_time / 60, gain_per_hour = gain / hours)
head(tmp)

#USEFUL CREATION FUNCTION
tmp <- transmute(flights,
                 dep_time,
                 hour = dep_time %/% 100,
                 minute = dep_time %% 100
                 )
# %/%: integer division
# %% remainder
head(tmp)
#log(), log2(), log10()
#lead(), lag() ... leading or lagging values

x <- 1:10
lag(x)
lead(x)
x
cumsum(x) #cumulate
cummean(x)

y <- c(1, 2, 2, NA, 3, 4)
min_rank(y) # ranking function
min_rank(desc(y))

# SUMMARISE
summarise(flights, delay = mean(dep_delay, na.rm = TRUE))
by_day <- group_by(flights, year, month, day)
head(by_day) # grouping doesn't change how the data looks (different with select)

summarise(by_day, delay = mean(dep_delay, na,rm = TRUE)) 
#It changes hot it acts with the other dplyr verbs
#group_by + summarise ... summarize the data by the parameter of group_by

# PIPE
by_dest <- group_by(flights, dest) #grouping by the feature 'dest'
delay <- summarise(by_dest,
                   count = n(), dist = mean(distance, na.rm = TRUE), delay = mean(arr_delay, na.rm = TRUE))
head(delay)
(delay <- filter(delay, count > 20, dest != "HNL"))


(
  delays <- flights %>%
    group_by(dest) %>%
    summarise(
      count = n(), dist = mean(distance, na.rm = TRUE), 
      delay = mean(arr_delay, na.rm = TRUE)) %>%
  filter(count > 20, dest != "HNL")
    )
)

ggplot(data = delays, mapping = aes(x = dist, y = delay)) +
  geom_point(aes(size = count), alpha = 1/3) + geom_smooth(se = FALSE)

#OTHER SUMMARY FUNCTIONS
not_cancelled <- flights %>% filter(!is.na(dep_delay), !is.na(arr_delay))

not_cancelled %>%
  group_by(year, month, day) %>%
  summarise(
    avg_delay1 = mean(arr_delay),
    avg_delay2 = mean(arr_delay[arr_delay > 0])
  )

not_cancelled %>%
  group_by(dest) %>%
  summarise(distance_sd = sd(distance)) %>%
  arrange(desc(distance_sd))

not_cancelled %>%
  group_by(year, month, day) %>%
  summarise(
    first = min(dep_time),
    last = max(dep_time)
  )

not_cancelled %>%
  group_by(year, month, day) %>%
  summarise(
    first_dep = first(dep_time),
    last_dep = last(dep_time)
  )

not_cancelled %>%
  group_by(dest) %>%
  summarise(carriers = n_distinct((carrier)) %>%
              arrange(desc(carriers)))

not_cancelled %>%
  count(dest)

not_cancelled %>% 
  count(tailnum, wt = distance) # total number of miles a plane flew:

# How many flights left before 5am? (these usually indicate delayed
# flights from the previous day)
not_cancelled %>% 
  group_by(year, month, day) %>% 
  summarise(n_early = sum(dep_time < 500))

# What proportion of flights are delayed by more than an hour?
not_cancelled %>% 
  group_by(year, month, day) %>% 
  summarise(hour_perc = mean(arr_delay > 60))

#GROUPED FILTERS
flights_sml %>% 
  group_by(year, month, day) %>%
  filter(rank(desc(arr_delay)) < 10)

popular_dests <- flights %>% 
  group_by(dest) %>% 
  filter(n() > 10000 )
dim(popular_dests)