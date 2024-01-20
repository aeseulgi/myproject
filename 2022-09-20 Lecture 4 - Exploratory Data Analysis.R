#Goals of EDA : develop an understanding of your data.
#During the initial phases of EDA you should feel free to investigate every idea that occurs to you. Some of these ideas will pan out, and some will be dead ends.

#Data cleaning: an application of EDA
#Ask questions about whether your data meets your expectations or not.


#An observation (data point) is a set of measurements made under similar conditions 
#(you usually make all of the measurements in an observation at the same time and on the same object). An observation will contain several values, each associated with a different variable.
#Tabular data is a set of values, each associated with a variable and an observation.

#Variation : Tendency of the values of a variable to change from measurement to measurement.

library(tidyverse)

#Categorical variables
ggplot(data = diamonds) + 
  geom_bar(mapping = aes(x = cut))

diamonds %>% dplyr::count(cut)


#Continuous variables
ggplot(data = diamonds) + 
  geom_histogram(mapping = aes(x = carat), binwidth = 0.5)

diamonds %>% count(ggplot2::cut_width(carat, 0.5))
#0.5단위로 width 자르기


smaller <- diamonds %>% 
  filter(carat < 3)  # zoom into just the diamonds with a size of less than three carats

ggplot(data = smaller, mapping = aes(x = carat)) + 
  geom_histogram(binwidth = 0.1)
#binwidth 줄이기

ggplot(data = smaller, mapping = aes(x = carat, colour = cut)) +
  geom_freqpoly(binwidth = 0.1)     # use geom_freqploy() instead of geom_histogram()


#Subgroups
#Clusters of similar values suggest that subgroups exist in your data.

#To understand the subgroups, ask:
#How are the observations within each cluster similar to each other?
#How are the observations in separate clusters different from each other?
#How can you explain or describe the clusters?
#Why might the appearance of clusters be misleading?

ggplot(data = faithful, mapping = aes(x = eruptions)) + 
  geom_histogram(binwidth = 0.25)
#Eruption times appear to be clustered into two groups: there are short eruptions (of around 2 minutes) and long eruptions (4-5 minutes), but little in between.

ggplot(diamonds) + 
  geom_histogram(mapping = aes(x = y), binwidth = 0.5)
#Wide x-axis

ggplot(diamonds) + 
  geom_histogram(mapping = aes(x = y), binwidth = 0.5) +
  coord_cartesian(ylim = c(0, 50))
#zoom in(ylim 설정)

unusual <- diamonds %>% 
  filter(y < 3 | y > 20) %>% 
  select(price, x, y, z) %>%
  arrange(y)
unusual
#unusual 데이터만 모으기(데이터 입력 잘못함)

diamonds2 <- diamonds %>% 
  mutate(y = ifelse(y < 3 | y > 20, NA, y))
#ifelse
#Has three arguments. The first argument test should be a logical vector.
#The result will contain the value of the second argument, yes, when test is TRUE, and the value of the third argument, no, when it is false.

ggplot(data = diamonds2, mapping = aes(x = x, y = y)) + 
  geom_point()

ggplot(data = diamonds2, mapping = aes(x = x, y = y)) + 
  geom_point(na.rm = TRUE)

library(nycflights13)
nycflights13::flights %>% 
  mutate(
    cancelled = is.na(dep_time),
    sched_hour = sched_dep_time %/% 100,
    sched_min = sched_dep_time %% 100,
    sched_dep_time = sched_hour + sched_min / 60
  ) %>% 
  ggplot(mapping = aes(sched_dep_time)) + 
  geom_freqpoly(mapping = aes(colour = cancelled), binwidth = 1/4)
#cancelled flights 구분 가능

ggplot(data = diamonds, mapping = aes(x = price)) + 
  geom_freqpoly(mapping = aes(colour = cut), binwidth = 500)

ggplot(data = diamonds, mapping = aes(x = price, y = ..density..)) + 
  geom_freqpoly(mapping = aes(colour = cut), binwidth = 500)
#It appears that fair diamonds (the lowest quality) have the highest average price! We need a better visualization of the distribution to investigate this observation.

ggplot(data = diamonds, mapping = aes(x = cut, y = price)) +
  geom_boxplot()

ggplot(data = mpg) +
  geom_boxplot(mapping = aes(x = reorder(class, hwy, FUN = median), y = hwy))
#hwy의 median을 기준으로 FUN을 이용해 reorder 함(covariation 탐구 가능)

ggplot(data = mpg, mapping = aes(x = class, y = hwy)) +
  geom_boxplot()

#Two categorical variables
ggplot(data = diamonds) +
  geom_count(mapping = aes(x = cut, y = color))

diamonds %>% 
  count(color, cut)

diamonds %>% 
  count(color, cut) %>%  
  ggplot(mapping = aes(x = color, y = cut)) +
  geom_tile(mapping = aes(fill = n))
#geom_tile() : 타일 색깔이 n 나타냄

ggplot(data = diamonds) +
  geom_point(mapping = aes(x = carat, y = price))
#exponential/overlap

ggplot(data = diamonds) + 
  geom_point(mapping = aes(x = carat, y = price), alpha = 1 / 100)
#점 투명도 조절하여 조금 더 시각화 정도 ↑

ggplot(data = smaller) +
  geom_bin2d(mapping = aes(x = carat, y = price))
#geom_tile()과 비슷함 - large dataset에 유용한 geom_bin2d

# install.packages("hexbin")
# library(hexbin)
ggplot(data = smaller) +
  geom_hex(mapping = aes(x = carat, y = price))

ggplot(data = smaller, mapping = aes(x = carat, y = price)) + 
  geom_boxplot(mapping = aes(group = cut_width(carat, width=0.1)))

# approximately the same number of points in each bin
ggplot(data = smaller, mapping = aes(x = carat, y = price)) + 
  geom_boxplot(mapping = aes(group = cut_number(carat, n=20)))

ggplot(data = faithful) + 
  geom_point(mapping = aes(x = eruptions, y = waiting))

library(modelr)
# model: assume exponential relation between `price` and `carat`
mod <- lm(log(price) ~ log(carat), data = diamonds)

diamonds2 <- diamonds %>% 
  add_residuals(mod) %>% 
  mutate(resid = exp(resid))  
#residuals(회귀분석 모델에 의해서 구해진 값과, 원래 데이터로 피팅되어 있는 값 사이에 차이값) of the model
#? 여기서 어떻게 carat의 효과를 뺀 건지 궁금합니다. => add_residuals()가 해줬다(함수에 내재되어 있음)


ggplot(data = diamonds2) + 
  geom_point(mapping = aes(x = carat, y = resid))
#Once you’ve removed the strong relationship between carat and price, you can see what you expect in the relationship between cut and price: relative to their size, better quality diamonds are more expensive.

ggplot(data = diamonds2) + 
  geom_boxplot(mapping = aes(x = cut, y = resid))
#carat을 통제하니 cut에 따른 price의 증가가 보인다.

