# INTRODUCTION-------------

# Setting working directory
setwd("C:/Users/supra/OneDrive/Desktop/Leeds University/Leeds 2nd Semester/Machine Learning in Practice/ASSIGNMENT")
#setwd(choose.dir())

# Clear Environment
rm(list = ls())

# Install packages
install.packages("WDI")
install.packages("countrycode")
install.packages("naivebayes")
install.packages("adabag")
install.packages("htmltools", dependencies = TRUE)
install.packages("xgboost")
install.packages("gbm")
#devtools::install_github('topepo/caret/pkg/caret')

# load libraries
library(gbm)
library(xgboost)
library(adabag)
library(randomForest)
library(psych)
library(WDI)
library(VIM)
library(stats)
library(imputeTS)
library(mice)
library(tm)
library(SnowballC)
library(tidyr)
library(dplyr)
library(caret)
library(gmodels)
library(cluster)
library(pROC)
library(ROCR)
library(neuralnet)
library(C50)
library(class)
library(e1071)
library(kernlab)
library(tidyverse)
library(Metrics)
library(ggplot2)
library(ggcorrplot)
library(corrplot)
library(wordcloud)
library(RColorBrewer)
library(wesanderson)
library(countrycode)
library(corrgram)
library(lubridate)
library(naivebayes)
library(textstem)
library(glmnet)

# Display numbers in natural form
options(scipen=999)

# Set seed
set.seed(467)

# FUNCTIONS------
# Function to check for missing values, blank values and white spaces in each column
missing_values <- function(data) {
  
  # Convert date columns to character format
  date_columns <- sapply(data, function(x) inherits(x, "Date"))
  data[date_columns] <- lapply(data[date_columns], as.character)
  
  # Replace missing values with NA
  data[data == "" | data == " "] <- NA
  
  # Number of NA values in each column
  na_values <- colSums(is.na(data))
  
  # Number of blank values in each column
  blank_values <- sapply(data, function(x) sum(x == ""))
  
  # Number of white space values in each column
  whitespace_values <- sapply(data, function(x) sum(grepl("^\\s*$", x)))
  
  # Combine results into a data frame
  quality_summary <- data.frame(
    Missing = na_values,
    #Blank_Values = blank_values,
    Whitespaces = whitespace_values
  )
  
  # Add TOTAL column
  quality_summary$MissingValues <- rowSums(quality_summary)
  
  # Filter rows where TOTAL column value is not equal to 0
  quality_summary <- quality_summary[quality_summary$MissingValues != 0, ]
  
  if (nrow(quality_summary) == 0) {
    smiley <- "\U263A"  # Unicode character for smiley face
    return(paste0("There are no missing values", " ", smiley))
  }
  
  
  # Sort the table based on the TOTAL column in descending order
  quality_summary <- quality_summary[order(-quality_summary$MissingValues), ]
  output <- subset(quality_summary, select = -c(Missing, Whitespaces))
  return(output)
}

# Create a custom function to remove symbols and special characters
remove_special_chars <- function(text) {
  # Remove symbols and special characters
  cleaned_text <- gsub("[^[:alnum:] ]", "", text)
  return(cleaned_text)
}

# Define a custom function to replace accented characters
replace_accented_chars <- function(text) {
  # Use iconv to convert text to ASCII encoding, which will remove accents
  cleaned_text <- iconv(text, to = "ASCII//TRANSLIT")
  return(cleaned_text)
}


# # Define a custom function to normalize numerical data
# normalize <- function(x) {
#   return((x - min(x)) / (max(x) - min(x)))
# }


# DATA CLEANING-------------

# Read the data
df <- read.csv("Data.csv", header = TRUE)

# Check structure of the data
str(df)

# View summary of the data
summary(df)
sample_n(df, 5)
head(df)

# Check for missing values, blanks and white spaces
missing_values(df)

# Create a copy
data <- df

# Check for duplicates
duplicates <- data[duplicated(data),]
duplicates

summary(data)

# Impute missing values for priceUSD and teamSize with mean
data$priceUSD[is.na(data$priceUSD)] <- mean(data$priceUSD, na.rm = TRUE)
data$teamSize[is.na(data$teamSize)] <- mean(data$teamSize, na.rm = TRUE)
str(data)
missing_values(data)

# Dropping the ID column
data <- data %>% select(-ID)

# EXTRA CODE
# data$social <- ifelse(df$hasVideo == 0 & df$hasGithub == 0 & df$hasReddit == 0, 0, 1)
# data <- data %>% select(-hasVideo, -hasGithub, -hasReddit)
# data <- data %>% mutate("is_success" = ifelse(success == "Yes", 1, 0))
# data$retained_percentage <- 1 - data$distributedPercentage
# data$coins_for_sale <-  round((data$distributedPercentage * data$coinNum),0)
# data$max_cap <- data$distributedPercentage * data$coinNum * data$priceUSD
# data <- data %>% select(-distributedPercentage, -coinNum, -priceUSD)
# 
# # Calculate frequency of each categorical variable
# platform_freq <- table(data$platform) / nrow(data)
# country_freq <- table(data$countryRegion) / nrow(data)
# 
# # Replace the platform categories with their frequencies
# data$platform_freq <- platform_freq[data$platform]
# data$country_freq <- country_freq[data$countryRegion]

# Check for correlation of numeric variables
numeric_data <- data[, sapply(data, is.numeric)]
ggcorrplot(cor(numeric_data)) + 
  labs(title = "Correlation Plot of Numerical Variables") + 
  theme(plot.title = element_text(hjust = 0, size = 10, face = "bold")) + 
  scale_fill_gradient2(low = "red", mid = "white", high = "blue", 
                       midpoint = 0, limits = c(-1, 1), name = "Correlation")


# Dropping the ID column
df <- df %>% select(-ID)

# Understanding the success categorical variable
unique(df$success)

# Factoring the success column
df$success <- factor(df$success, levels = c("Y", "N"), labels = c("Yes", "No"))
table(df$success)
round(prop.table(table(df$success)) * 100, digits = 1)

# Understanding hasVideo variable
unique(df$hasVideo)
summary(df$hasVideo)
class(df$hasVideo)

# Understanding rating variable
summary(df$rating)

# Understanding priceUSD variable
summary(df$priceUSD)

# Filter out prices that are equal to 0
df <- df %>% filter(priceUSD > 0 | is.na(priceUSD))

# Understanding countryRegion variable
summary(df$countryRegion)
str(df$countryRegion)

# Change country names to lower case
df <- df %>% mutate(countryRegion = tolower(countryRegion))

# Strip irrelevant spaces in countryRegion
df <- df %>% mutate(countryRegion = trimws(countryRegion))

# check for unique values of countryRegion
sort(unique(df$countryRegion))

# change "curaçao" to curacao in countryRegion
df <- df %>% mutate(countryRegion = ifelse(countryRegion == 'curaçao', 'curacao', countryRegion))

# testing for curaçao
df %>% filter(countryRegion == 'curaçao') %>% count()

# change méxico to mexico in countryRegion
df <- df %>% mutate(countryRegion = ifelse(countryRegion == 'méxico', 'mexico', countryRegion))

# testing for méxico
df %>% filter(countryRegion == 'méxico') %>% count()

# Update both date columns to Date format
df$startDate <- as.Date(df$startDate, format = "%d/%m/%Y")
df$endDate <- as.Date(df$endDate, format = "%d/%m/%Y")

# Calculate duration of ICO campaign using startDate and endDate
df <- df %>% mutate('ico_duration' = endDate - startDate)

# Convert ico_duration to numeric format
df$ico_duration <- as.numeric(df$ico_duration)

# Check data type of ico_duration
class(df$ico_duration)
summary(df$ico_duration)

# Filter out observations with negative ico_duration
df <- df %>% filter(ico_duration >= 0)

# Understanding teamSize variable
summary(df$teamSize)

# Understanding hasGithub variable
summary(df$hasGithub)

# Understanding hasReddit variable
summary(df$hasReddit)

# Understanding platform variable
sort(unique(df$platform))

# Change text to lower for platform
df <- df %>% mutate(platform = tolower(platform))

# Remove white spaces in platform
df <- df %>% mutate(platform = trimws(platform))

# Recheck platform variable
sort(unique(df$platform))

# Convert btc to bitcoin
df <- df %>% mutate(platform = ifelse(platform == 'btc', "bitcoin", platform))

# Convert eth, ethererum, etherum, to ethereum
df <- df %>% mutate(platform = ifelse(platform == 'eth' | platform == 'ethererum' | platform == 'etherum', "ethereum", platform))

# Convert x11 blockchain to x11
df <- df %>% mutate(platform = ifelse(platform == 'x11 blockchain', 'x11', platform))

# Convert stellar protocol to stellar
df <- df %>% mutate(platform = ifelse(platform == 'stellar protocol', 'stellar', platform))

# Convert "pos + pow", "pos,pow", "pow/pos" to "pos_pow"
df <- df %>% mutate(platform = ifelse(platform == "pos + pow" | platform == "pos,pow" | platform == "pos/pow" | platform == "pow/pos", "pos_pow", platform))

# Recheck platform variable
sort(unique(df$platform))

# Understanding coinNum variable
summary(df$coinNum)

# Understanding minInvestment variable
summary(df$minInvestment)

# Understanding distributedPercentage variable
summary(df$distributedPercentage)

# Filter out values that are greater than 1 in distributedPercentage
df <- df %>% filter(!distributedPercentage > 1)
summary(df$distributedPercentage)

# Create 'social' variable based on hasVideo, hasGithub, & hasReddit
df$social <- ifelse(df$hasVideo == 0 & df$hasGithub == 0 & df$hasReddit == 0, 0, 1)
df <- df %>% select(-hasVideo, -hasGithub, -hasReddit)

# Fetching External Data - GDP
gdp <- WDI(country="all", indicator=c("NY.GDP.MKTP.CD"), start=2010, end=2020)
colnames(gdp) <- c("country", "iso2c", "iso3c", "year", "gdp")
gdp <- subset(gdp, select = -c(country, iso2c))

# Fetching External Data - Unemployment
unemp <- WDI(country="all", indicator=c("SL.UEM.TOTL.ZS"), start=2010, end=2020)
colnames(unemp) <- c("country", "iso2c", "iso3c", "year", "unemployment")
unemp <- subset(unemp, select = -c(country, iso2c))

# Fetching External Data - CPI Inflation
inflation <- WDI(country="all", indicator=c("FP.CPI.TOTL.ZG"), start=2010, end=2020)
colnames(inflation) <- c("country", "iso2c", "iso3c", "year", "inflation")
inflation <- subset(inflation, select = -c(country, iso2c))

# Reading External Data - Bitcoin Prices from coinmarketcap
btc <- read.csv("Bitcoin_historical_data_coinmarketcap.csv", header = FALSE)
head(btc)

# Split the single column into multiple columns using semicolons as separators
btc <- data.frame(do.call(rbind, strsplit(as.character(btc$V1), ";")))

# Set the first row as column names, then drop the redundant first row
colnames(btc) <- btc[1, ]
btc <- btc[-1, ]

# Remove irrelevant columns
btc <- btc[, !(names(btc) %in% c("timeOpen", "timeClose", "timeHigh", "timeLow", "name", "high", "low", "close", "volume", "marketCap"))]

# Rename open column to btc
names(btc)[names(btc) == "open"] <- "btc"

# Update "timestamp" to a date column in the format of "%d/%m/%Y"
btc$timestamp <- as.Date(btc$timestamp, origin="1970-01-01")

# Format "timestamp" to the desired format "%d/%m/%Y"
btc$timestamp <- as.Date(btc$timestamp, format = "%d/%m/%Y")
btc$btc <- as.numeric(btc$btc)

# Add ISO 3-letter country codes to original dataset based on countryRegion
df$iso3c <- countrycode(df$countryRegion, origin = "country.name", destination = "iso3c")

# Extract year from startDate column in original dataset
df$year <- as.numeric(substr(df$startDate, 1, 4))

# Merge BTC, GDP, Inflation & Unemployment data
df <- merge(df, btc, by.x = "startDate", by.y = "timestamp", all.x = TRUE)
df <- merge(df, gdp, by.x = c("iso3c", "year"), by.y = c("iso3c", "year"), all.x = TRUE)
df <- merge(df, inflation, by.x = c("iso3c", "year"), by.y = c("iso3c", "year"), all.x = TRUE)
df <- merge(df, unemp, by.x = c("iso3c", "year"), by.y = c("iso3c", "year"), all.x = TRUE)

# Dropping the year and iso3 column
df <- df[, !(names(df) %in% c("year", "iso3c"))]

# Summary of the cleaned dataset
summary(df)
str(df)
missing_values(df)
dim(df)
# Saving the cleaned dataset to csv file
write.csv(df, file = "Final Data/cleaned_data.csv", row.names = FALSE)

# DATA EXPLORATION-------------

# Clear Environment
rm(list = ls())

# Read the data
df <- read.csv("Final Data/cleaned_data.csv", header = TRUE)
str(df)
missing_values(df)

# rating
df %>%
  ggplot(aes(x = 1.0, y = rating, fill = success)) +
  geom_boxplot() +
  stat_summary(fun = median, geom = "text", aes(label = round(after_stat(y), 2)), vjust = -0.5) +
  facet_wrap(~ success) +
  labs(x = "", y = "ICO Rating", title = "Distribution of rating variable") +
  scale_fill_manual(values = c("Yes" = "skyblue", "No" = "gold"))

# social
df$social <- factor(df$social, levels = c("1", "0"), labels = c("Yes", "No"))
CrossTable(df$success, df$social,  dnn=c('Sucess', 'Social'),
           prop.chisq = FALSE, prop.t = FALSE, prop.r = FALSE)
percentage_data <- df %>%
  group_by(social, success) %>%
  summarise(count = n()) %>%
  mutate(percentage = count / sum(count) * 100)
# df %>% ggplot(aes(x = social, fill = success)) + 
#   geom_bar() + 
#   labs(y="Number of ICO Projects", title = "Distribution of social variable") +
#   scale_fill_manual(values = c("Yes" = "magenta", "No" = "khaki"))
ggplot(percentage_data, aes(x = social, y = count, fill = success)) + 
  geom_bar(stat = "identity") + 
  labs(y = "Number of ICO Projects", title = "Distribution of social variable") +
  scale_fill_manual(values = c("Yes" = "skyblue", "No" = "gold")) +
  geom_text(aes(label = paste0(round(percentage), "%")),
            position = position_stack(vjust = 0.5), size = 2.5)


# priceUSD
df %>% ggplot(aes(x = 1.0, y = log(priceUSD, 10), fill = success)) + 
  geom_boxplot() + 
  stat_summary(fun = median, geom = "text", aes(label = round(after_stat(y), 2)), vjust = -0.5) +
  labs(x="", y="Log of priceUSD", title = "Distribution of log value of priceUSD variable") + 
  facet_wrap(~ success) +
  scale_fill_manual(values = c("Yes" = "skyblue", "No" = "gold"))

# BTC
df %>% ggplot(aes(x = 1.0, y = btc, fill = success)) + 
  geom_boxplot() + 
  stat_summary(fun = median, geom = "text", aes(label = round(after_stat(y), 2)), vjust = -0.5) +
  facet_wrap(~ success) + 
  labs(y="Bitcoin Price (USD)", title = "Distribution of btc variable") +
  scale_fill_manual(values = c("Yes" = "skyblue", "No" = "gold"))

# GDP
df %>% ggplot(aes(x = 1.0, y = log(gdp, 10), fill = success)) + 
  geom_boxplot() + 
  stat_summary(fun = median, geom = "text", aes(label = round(after_stat(y), 2)), vjust = -0.5) +
  labs(x="", y="Log of GDP", title = "Distribution of log value of gdp variable") + 
  facet_wrap(~ success) +
  scale_fill_manual(values = c("Yes" = "skyblue", "No" = "gold"))

# Inflation
df %>% ggplot(aes(x = 1.0, y = log(inflation, 10), fill = success)) + 
  geom_boxplot() + 
  stat_summary(fun = median, geom = "text", aes(label = round(after_stat(y), 2)), vjust = +1.5) +
  labs(x="", y="Log of Inflation", title = "Distribution of log value of inflation variable") + 
  facet_wrap(~ success) +
  scale_fill_manual(values = c("Yes" = "skyblue", "No" = "gold"))

# Unemployment
df %>% ggplot(aes(x = 1.0, y = log(unemployment, 10), fill = success)) + 
  geom_boxplot() + 
  stat_summary(fun = median, geom = "text", aes(label = round(after_stat(y), 2)), vjust = -0.5) +
  labs(x="", y="Log of Unemployment", title = "Distribution of log value of unemployment variable") + 
  facet_wrap(~ success) +
  scale_fill_manual(values = c("Yes" = "skyblue", "No" = "gold"))

# countryRegion
df$countryRegion <- factor(df$countryRegion)
sort(summary(df$countryRegion), decreasing = TRUE)

# Create a frequency table of countryRegion and save as data frame
freq_table <- table(df$countryRegion)

# Sort in descending order
freq_df <- as.data.frame(table(countryRegion = df$countryRegion))
freq_df <- freq_df[order(-freq_df$Freq), ]

# Filter the top 6 countries
top_countries <- freq_df[1:6, ]

# Plot a bar graph for top 6 countries
ggplot(top_countries, aes(x = reorder(countryRegion, -Freq), y = Freq, fill = countryRegion)) + 
  geom_bar(stat = "identity") + 
  geom_text(aes(label = Freq), vjust = 2, size = 3, color = "white") +
  labs(x = "Country", y = "Number of ICO Projects", title = "Top 6 Countries for ICO Projects") + 
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) + 
  scale_fill_discrete(name = "Country") + 
  guides(fill = "none")

# platform
df$platform <- factor(df$platform)
sort(summary(df$platform), decreasing = TRUE)

# Create a frequency table of platform and save as data frame
freq_table <- table(df$platform)

# Sort in descending order
freq_df <- as.data.frame(table(platform = df$platform))
freq_df <- freq_df[order(-freq_df$Freq), ]

# Filter the top 5 platforms
top_platforms <- freq_df[1:5, ]

# Plot a bar graph for top 5 platforms
ggplot(top_platforms, aes(x = reorder(platform, -Freq), y = Freq, fill = platform)) + 
  geom_bar(stat = "identity") + 
  geom_text(aes(label = Freq), vjust = 0.3, size = 3, color = "black") +
  labs(x = "Platform", y = "Number of ICO Projects", title = "Top 5 Platforms used by ICO Projects") + 
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) + 
  scale_fill_discrete(name = "Platform") + 
  guides(fill = "none")

# Number of coins
df %>% ggplot(aes(x = 1.0, y = log(coinNum, 10), fill = success)) + 
  geom_boxplot() + 
  labs(x= "", y="Log of coinNum", title = "Distribution of log value of coinNum variable") + 
  facet_wrap(~ success) + 
  theme(axis.text.x=element_blank(), axis.ticks.x=element_blank())

# Team Size
df %>% ggplot(aes(x = 1.0, y = teamSize, fill = success)) + 
  geom_boxplot() + 
  stat_summary(fun = median, geom = "text", aes(label = round(after_stat(y), 2)), vjust = -0.5) +
  labs(x= "", y="Team Size", title = "Distribution of teamSize variable") + 
  facet_wrap(~ success) + 
  theme(axis.text.x=element_blank(), axis.ticks.x=element_blank()) +
  scale_fill_manual(values = c("Yes" = "skyblue", "No" = "gold"))

# Minimum investment
df$minInvestment <- factor(df$minInvestment, levels = c("1", "0"), labels = c("Yes", "No"))
CrossTable(df$success, df$minInvestment,  dnn=c('Sucess', 'Minimum Investment'),
           prop.chisq = FALSE, prop.t = FALSE, prop.r = FALSE)
df %>% ggplot(aes(x = minInvestment, fill = success)) + 
  geom_bar() + 
  labs(x= "Minimum Investment", y="Number of ICO Projects", title = "Distribution of minInvestment variable")

# Distributed Percentage
summary(df$distributedPercentage)
df %>% ggplot(aes(x = 1.0, y = distributedPercentage, fill = success)) + 
  geom_boxplot() + 
  stat_summary(fun = median, geom = "text", aes(label = round(..y.., 2)), vjust = -0.5) +
  labs(x="", y="Distributed Percentage", title = "Distribution of the distributedPercentage variable") + 
  facet_wrap(~ success)

# Analysing words used in ICO campaigns.
# Create a corpus and preprocessing the text data from the brandSlogan column
corpus <- Corpus(VectorSource(df$brandSlogan)) 
corpus$content[11:20]
corpus <- tm_map(corpus, content_transformer(function(x) gsub("(f|ht)tps?://\\S+|www\\.\\S+", "", x))) # Remove URLs
corpus <- tm_map(corpus, content_transformer(function(x) gsub("([a-z])([A-Z])", "\\1 \\2", x))) # Remove joined words
corpus <- tm_map(corpus, content_transformer(remove_special_chars)) # Remove special characters including punctuation
corpus <- tm_map(corpus, content_transformer(tolower)) # Convert to lowercase
corpus <- tm_map(corpus, removePunctuation) # Remove punctuation
corpus <- tm_map(corpus, removeNumbers) # Remove numbers
corpus <- tm_map(corpus, removeWords, stopwords("english")) # Remove stop words
corpus <- tm_map(corpus, content_transformer(replace_accented_chars)) # Replace accented characters
corpus <- tm_map(corpus, stripWhitespace) # Strip whitespace
corpus <- tm_map(corpus, lemmatize_strings) # lemmatization
corpus$content[11:20]

# Create a document-term matrix
dtm <- TermDocumentMatrix(corpus)
matrix <- as.matrix(dtm)
words <- sort(rowSums(matrix),decreasing=TRUE)
df_words <- data.frame(word = names(words),freq=words)
df_words[1:20,]

# Create a word cloud
set.seed(467)
dev.off()
wordcloud(words = df_words$word, freq = df_words$freq, 
          min.freq = 10, max.words=200, random.order=FALSE, 
          rot.per=0.35, colors=brewer.pal(8, "Dark2"))


# DATA PREPROCESSING-------------

# Clear Environment
rm(list = ls())

# Read cleaned data
df <- read.csv("Final Data/cleaned_data.csv")

# Check missing values
missing_summary <- missing_values(df)
missing_summary

# Create a horizontal bar plot of variables with missing values
ggplot(missing_summary, aes(x = MissingValues, y = reorder(row.names(missing_summary), MissingValues))) +
  geom_bar(stat = "identity", fill = "skyblue", width = 0.5) +
  geom_text(aes(label = ifelse(MissingValues > 0, MissingValues, "")), hjust = -0.2, size = 3, color = "black") + 
  labs(title = "Variables with Count of Missing Values",
       x = "Total Missing Values",
       y = "Variable") +
  theme_minimal()

# Visualizing observations with missing values
missing_cols <- c("unemployment", "inflation", "priceUSD", "gdp", "teamSize", "countryRegion", "platform", "btc")
missing_df <- df[, missing_cols]
aggr(missing_df, numbers = TRUE, prop = FALSE, combined = TRUE, cex.axis = 0.7)

## HANDLING MISSING DATA-------------
# SI method
data_si <- df
# priceUSD
data_si$priceUSD[is.na(data_si$priceUSD)] <- mean(data_si$priceUSD, na.rm = TRUE)
# teamSize
data_si$teamSize[is.na(data_si$teamSize)] <- round(mean(data_si$teamSize, na.rm = TRUE))
missing_values(data_si)

# Check correlation between variables
corrgram(df)
sum(!complete.cases(df))
missdata <- df
missdata$missing <- as.numeric(!complete.cases(df)) 
corrgram(missdata)

# Multiple imputation

# Perform multiple imputation on Numeric variables
mi_df <- df %>% 
  mutate("is_success" = ifelse(success == "Yes", 1, 0))
datami <- mice(subset(mi_df, select = c('social', 'rating', 'priceUSD', 
                                        'teamSize', 'coinNum', 'minInvestment',
                                        'distributedPercentage', 'ico_duration', 
                                        'btc', 'gdp', 'inflation', 
                                        'unemployment', 'is_success')), 
               method = "cart", m = 5, maxit = 50, seed = 467)

# Pooling imputed datasets
data_mi <- complete(datami)
missing_values(data_mi)

# Summary of teamSize
summary(data_si$teamSize)
summary(data_mi$teamSize)

# Summary of priceUSD
summary(data_si$priceUSD) 
summary(data_mi$priceUSD)

# Saving data from the two imputation methods
df_si <- data_si 
df_mi <- cbind(df[!names(df) %in% names(data_mi)], data_mi)

missing_values(df_mi)

# Summary of both SI & MI
summary(df_si)
summary(df_mi)


# Boxplot of teamSize
df %>% ggplot(aes(x = 1.0, y = teamSize)) + 
  geom_boxplot() + 
  labs(title = "Distribution of teamSize before imputation") + 
  xlab("")

df_si %>% ggplot(aes(x = 1.0, y = teamSize)) + 
  geom_boxplot() + 
  labs(title = "Distribution of teamSize after simple imputation") + 
  xlab("")

df_mi %>% ggplot(aes(x = 1.0, y = teamSize)) + 
  geom_boxplot() + 
  labs(title = "Distribution of teamSize after multiple imputation") + 
  xlab("")


# Histograms of teamSize
df %>% ggplot(aes(x = teamSize)) + 
  geom_histogram(binwidth = 3) + 
  labs(title = "Distribution of teamSize before imputation")

df_si %>% ggplot(aes(x = teamSize)) + 
  geom_histogram(binwidth = 3) + 
  labs(title = "Distribution of teamSize after simple imputation method")

df_mi %>% ggplot(aes(x = teamSize)) + 
  geom_histogram(binwidth = 3) + 
  labs(title = "Distribution of teamSize after multiple imputation method")

# Boxplot of priceUSD
df %>% ggplot(aes(x = 1.0, y = log(priceUSD))) + 
  geom_boxplot() + 
  stat_summary(fun = median, geom = "text", aes(label = round(..y.., 2)), vjust = -0.5) +
  labs(title = "Distribution of priceUSD before Imputation") + 
  xlab("")

df_si %>% ggplot(aes(x = 1.0, y = log(priceUSD))) + 
  geom_boxplot() + 
  stat_summary(fun = median, geom = "text", aes(label = round(..y.., 2)), vjust = -0.5) +
  labs(title = "Distribution of priceUSD after Simple Imputation") + 
  xlab("")

df_mi %>% ggplot(aes(x = 1.0, y = log(priceUSD))) + 
  geom_boxplot() + 
  stat_summary(fun = median, geom = "text", aes(label = round(..y.., 2)), vjust = -0.5) +
  labs(title = "Distribution of priceUSD after Multiple Imputation") + 
  xlab("")

# Imputation on categorical variables
missing_values(df_mi)
df_2 <- df_mi
df_2[df_2 == "" | df_2 == " "] <- NA

# Replace missing country with Unknown
df_2 <- df_2 %>% mutate(countryRegion = ifelse(is.na(countryRegion),"Unknown", countryRegion))

# Replace missing platform with Unknown
df_2 <- df_2 %>% mutate(platform = ifelse(is.na(platform), "Unknown", platform))

missing_values(df_2)

# Saving the cleaned dataset to csv file
write.csv(df_2, file = "Final Data/imputed_data.csv", row.names = FALSE)

## HANDLING OUTLIERS-------------

summary(df_2)

### priceUSD-------------
boxplot(df_2$priceUSD, main = "Boxplot of priceUSD", ylab = "Price (USD)")
z_score <- scale(df_2$priceUSD)

# identify the data points with Z-score greater than 3 or less than -3
outliers <- which(abs(z_score) > 3) 
length(outliers) 
df_outliers <- df_2[outliers, ] 
summary(df_2$priceUSD)

# Calculate the percentile values for Winsorization (e.g., 1st and 99th percentile)
winsor_lower <- quantile(df_2$priceUSD, 0.01) 
winsor_upper <- quantile(df_2$priceUSD, 0.99)

# Apply Winsorization
df_2$priceUSD[df_2$priceUSD < winsor_lower] <- winsor_lower
df_2$priceUSD[df_2$priceUSD > winsor_upper] <- winsor_upper

# Check the summary statistics of priceUSD after Winsorization
summary(df_2$priceUSD)


### coinNum-------------
boxplot(df_2$coinNum, main = "Boxplot of coinNum", ylab = "Number of coins")
z_score <- scale(df_2$coinNum)

# identify the data points with Z-score greater than 3 or less than -3
outliers <- which(abs(z_score) > 3) 
length(outliers) 
df_outliers <- df_2[outliers, ] 
summary(df_2$coinNum)

# Calculate the percentile values for Winsorization (e.g., 1st and 99th percentile)
winsor_lower <- quantile(df_2$coinNum, 0.01) 
winsor_upper <- quantile(df_2$coinNum, 0.99)

# Apply Winsorization
df_2$coinNum[df_2$coinNum < winsor_lower] <- winsor_lower
df_2$coinNum[df_2$coinNum > winsor_upper] <- winsor_upper

# Check the summary statistics of coinNum after Winsorization
summary(df_2$coinNum)

### inflation-------------
boxplot(df_2$inflation, main = "Boxplot of inflation", ylab = "GDP")
z_score <- scale(df_2$inflation)

# identify the data points with Z-score greater than 3 or less than -3
outliers <- which(abs(z_score) > 3) 
length(outliers) 
df_outliers <- df_2[outliers, ] 
summary(df_2$inflation)

# Calculate the percentile values for Winsorization (e.g., 1st and 99th percentile)
winsor_lower <- quantile(df_2$inflation, 0.01) 
winsor_upper <- quantile(df_2$inflation, 0.99)

# Apply Winsorization
df_2$inflation[df_2$inflation < winsor_lower] <- winsor_lower
df_2$inflation[df_2$inflation > winsor_upper] <- winsor_upper

# Check the summary statistics of inflation after Winsorization
summary(df_2$inflation)

### unemployment-------------
boxplot(df_2$unemployment, main = "Boxplot of unemployment", ylab = "GDP")
z_score <- scale(df_2$unemployment)

# identify the data points with Z-score greater than 3 or less than -3
outliers <- which(abs(z_score) > 3) 
length(outliers) 
df_outliers <- df_2[outliers, ] 
summary(df_2$unemployment)

# Calculate the percentile values for Winsorization (e.g., 1st and 99th percentile)
winsor_lower <- quantile(df_2$unemployment, 0.01) 
winsor_upper <- quantile(df_2$unemployment, 0.99)

# Apply Winsorization
df_2$unemployment[df_2$unemployment < winsor_lower] <- winsor_lower
df_2$unemployment[df_2$unemployment > winsor_upper] <- winsor_upper

# Check the summary statistics of unemployment after Winsorization
summary(df_2$unemployment)


### ico_duration------------
boxplot(df_2$ico_duration, main = "Boxplot of ICO duration", ylab = "ICO duration")
z_score <- scale(df_2$ico_duration)

# identify the data points with Z-score greater than 3 or less than -3
outliers <- which(abs(z_score) > 3) 
length(outliers) 
df_outliers <- df_2[outliers, ] 
summary(df_2$ico_duration)

# Calculate the percentile values for Winsorization (e.g., 1st and 99th percentile)
winsor_lower <- quantile(df_2$ico_duration, 0.01) 
winsor_upper <- quantile(df_2$ico_duration, 0.99)

# Apply Winsorization
df_2$ico_duration[df_2$ico_duration < winsor_lower] <- winsor_lower
df_2$ico_duration[df_2$ico_duration > winsor_upper] <- winsor_upper

# Convert startDate and endDate columns to Date format
df_2$startDate <- as.Date(df_2$startDate) 
df_2$endDate <- as.Date(df_2$endDate)

# Update endDate based on Winsorized ico_duration
df_2$endDate <- df_2$startDate + df_2$ico_duration

# Check the summary statistics of ico_duration after Winsorization
summary(df_2$ico_duration)

# Save Imputed file
write.csv(df_2, file = "Final Data/winsorized_data.csv", row.names = FALSE)

# DATA ENCODING-------------

# Clear Environment
rm(list = ls())

df <- read.csv("Final Data/winsorized_data.csv")
summary(df)

data1 <- df %>% mutate("is_success" = ifelse(success == "Yes", 1, 0))
numeric_data <- df[, sapply(df, is.numeric)]
ggcorrplot(cor(numeric_data)) + 
  labs(title = "Correlation Plot of Numerical Variables") + 
  theme(plot.title = element_text(hjust = 0, size = 10, face = "bold")) + 
  scale_fill_gradient2(low = "red", mid = "white", high = "blue", 
                       midpoint = 0, limits = c(-1, 1), name = "Correlation")

# Dropping irrelevant columns
df <- df %>% select(-startDate, -endDate, -is_success)
str(df)

# Factoring character variables
df$success <- as.factor(df$success)
df$countryRegion <- as.factor(df$countryRegion)
df$platform <- as.factor(df$platform)

# Separating categorical, text, numerical, and target variables
df_numerical <- df %>% select(-brandSlogan, -countryRegion, -platform, -success)
df_categorical <- df %>% select(countryRegion, platform)
df_text <- df %>% select(brandSlogan)
df_target <- df %>% select(success)

# ENCODING CATEGORICAL VARIABLES
# Calculate frequency of each categorical variable
platform_freq <- table(df_categorical$platform) / nrow(df_categorical)
country_freq <- table(df_categorical$countryRegion) / nrow(df_categorical)

# Replace the platform categories with their frequencies
df_categorical$platform <- platform_freq[df_categorical$platform]
df_categorical$countryRegion <- country_freq[df_categorical$countryRegion]

# Rename 'platform' column to avoid repetition 
colnames(df_categorical)[2] <- "PlatformCategory"

# ENCODING NUMERICAL VARIABLES
# Checking correlation among numerical variables
correlationMatrix <- cor(df_numerical)
correlationMatrix
ggcorrplot(cor(df_numerical)) + 
  labs(title = "Correlation Plot of Numerical Variables") + 
  theme(plot.title = element_text(hjust = 0, size = 10, face = "bold")) + 
  scale_fill_gradient2(low = "red", mid = "white", high = "blue", midpoint = 0, limits = c(-1, 1), name = "Correlation")

# Find attributes that are highly correlated (ideally >0.75)
highlyCorrelated <- findCorrelation(correlationMatrix, cutoff=0.75)
highlyCorrelated

# Dropping highly correlated columns
#df_numerical <- df_numerical %>% select(-hasVideo, -hasGithub, -hasReddit)

# Scaling all values
df_numerical <- scale(df_numerical)

# ENCODING TEXT VARIABLE
corpus <- Corpus(VectorSource(df_text$brandSlogan)) 
corpus$content[13:23]
corpus <- Corpus(VectorSource(df_text$brandSlogan)) 
corpus <- tm_map(corpus, content_transformer(function(x) gsub("(f|ht)tps?://\\S+|www\\.\\S+", "", x))) # Remove URLs
corpus <- tm_map(corpus, content_transformer(function(x) gsub("([a-z])([A-Z])", "\\1 \\2", x))) # Remove joined words
corpus <- tm_map(corpus, content_transformer(remove_special_chars)) # Remove special characters including punctuation
corpus <- tm_map(corpus, content_transformer(tolower))  # Convert to lowercase
corpus <- tm_map(corpus, removePunctuation)             # Remove punctuation [not reqd]
corpus <- tm_map(corpus, removeNumbers)                 # Remove numbers
corpus <- tm_map(corpus, removeWords, stopwords("english"))  # Remove stop words
corpus <- tm_map(corpus, content_transformer(replace_accented_chars)) # Replace accented characters
corpus <- tm_map(corpus, stripWhitespace)               # Strip whitespace
corpus <- tm_map(corpus, lemmatize_strings)       # lemmatization
corpus$content[13:23]

dtm <- DocumentTermMatrix(corpus)
#tfidf <- weightTfIdf(dtm)

# Get the indices of terms with highest frequency or importance
top_terms_dtm <- findFreqTerms(dtm, lowfreq = 100)
#top_terms_tfidf <- findFreqTerms(tfidf, lowfreq = 100)
top_terms_dtm
#top_terms_tfidf

# Subset the dtm and tfidf matrices to include only top terms
dtm_subset <- dtm[, top_terms_dtm]
#tfidf_subset <- tfidf[, top_terms_tfidf]
dtm_subset
#tfidf_subset

# Convert the dtm to a data frame
df_text_dtm <- as.data.frame(as.matrix(dtm_subset))
#df_text_tfidf <- as.data.frame(as.matrix(tfidf_subset))

# COMBINE ALL ENCODED DATA
final_data <- cbind(df_target, df_categorical, df_numerical, df_text_dtm)
final_data$success <- as.factor(final_data$success)  # Ensure the target variable is factor type
str(final_data)

# Saving the encoded dataset to csv file
write.csv(final_data, file = "Final Data/encoded_data.csv", row.names = FALSE)

# FEATURE SELECTION----
rm(list = ls())
# Read data
final_data <- read.csv("Final Data/encoded_data.csv")
final_data$success <- as.factor(final_data$success)

data1 <- final_data %>% mutate("is_success" = ifelse(success == "Yes", 1, 0))
numeric_data <- data1[, sapply(data1, is.numeric)]

ggcorrplot(cor(numeric_data)) + 
  labs(title = "Correlation Plot of Encoded Variables") + 
  theme(plot.title = element_text(hjust = 0, size = 10, face = "bold")) + 
  scale_fill_gradient2(low = "red", mid = "white", high = "blue", 
                       midpoint = 0, limits = c(-1, 1), name = "Correlation")

# Split data into training and testing sets
set.seed(467)  # for reproducibility
# train_indices <- createDataPartition(final_data$success, p = 0.8, list = FALSE)
# train_data <- final_data[train_indices, ]
# test_data <- final_data[!(1:nrow(final_data) %in% train_indices), ]

# Train Random Forest model
rf_model <- randomForest(success ~ ., data = final_data, ntree = 500)
summary(rf_model)
plot(rf_model)

# Feature Importance
varImpPlot(rf_model, main = "Variable Importance Plot")

# Extract importance scores from the rf_model$importance list
importance_scores <- rf_model$importance[,"MeanDecreaseGini"]
importance_percentage <- (importance_scores / sum(importance_scores)) * 100

# Sort the importance scores in descending order
sorted_score <- sort(importance_scores, decreasing = TRUE)
sorted_importance <- sort(importance_percentage, decreasing = TRUE)

# Calculate cumulative importance
cumulative_importance <- cumsum(sorted_importance)

# Create a data frame with variables, sorted scores, and cumulative scores
importance_df <- data.frame(
  Variable = names(sorted_importance),
  Importance = sorted_importance,
  Cumulative_Importance = cumulative_importance
)

# Print resulting table
print(importance_df)

# Plot the curved line graph
importance_df$Variable <- factor(importance_df$Variable, levels = importance_df$Variable)

ggplot(importance_df, aes(x = Variable, y = Cumulative_Importance, group = 1)) +
  geom_smooth(method = "loess", se = FALSE) +  # Don't plot standard error ribbon
  geom_ribbon(data = importance_df[1:12, ], aes(x = Variable, ymax = Cumulative_Importance, ymin = 0), fill = "lightgreen", alpha = 0.5) +
  geom_point() +
  geom_segment(x = importance_df$Variable[12], y = 0, xend = importance_df$Variable[12], yend = importance_df$Cumulative_Importance[12], color = "red") +
  geom_text(aes(x = importance_df$Variable[12], y = importance_df$Cumulative_Importance[12], label = round(importance_df$Cumulative_Importance[12], 2)), vjust = -0.9, color = "red") +
  labs(x = "Variables", y = "Cumulative Importance", title = "Cumulative Importance Diagram of Variables impacting ICO Success") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))  # Rotate x-axis labels for better readability

# Select the top 12 features based on importance scores
top_features <- importance_df[1:12, ]
top_features$Variable <- as.character(top_features$Variable)
df_filtered <- final_data[, c("success", top_features$Variable)]
df_filtered <- df_filtered %>% rename(platform = PlatformCategory) # change name back to 'platform'

# save data
write.csv(df_filtered, file = "Final Data/filtered_data.csv", row.names = FALSE)

# DATA MODELLING-------------

# Clear Environment
rm(list = ls())

# Load dataset
data <- read.csv("Final Data/filtered_data.csv")

# Testing Class Imbalance
table(data$success)
round(prop.table(table(data$success)) * 100, digits = 1)

# Check for correlation of numeric variables
data1 <- data %>% mutate("is_success" = ifelse(success == "Yes", 1, 0))
numeric_data <- data1[, sapply(data1, is.numeric)]
ggcorrplot(cor(numeric_data)) + 
  labs(title = "Correlation Plot of Feature Selected Variables") + 
  theme(plot.title = element_text(hjust = 0, size = 10, face = "bold")) + 
  scale_fill_gradient2(low = "red", mid = "white", high = "blue", 
                       midpoint = 0, limits = c(-1, 1), name = "Correlation")

# Factoring target variable
data$success <- factor(data$success) 
round(prop.table(table(data$success)) * 100, digits = 1)

# Creating training and test data sets
set.seed(467) 
smp_size <- floor(0.8 * nrow(data)) 
training_indices <- sample(nrow(data), smp_size) 

# Storing train and test data without target variable
train_data <- data[training_indices, -1] 
test_data <- data[-training_indices, -1]
# Storing the train and test target variable
train_labels <- data[training_indices, 1] 
test_labels <- data[-training_indices, 1]

# Storing train and test data with target variable
train_data_full <- data[training_indices,] 
test_data_full <- data[-training_indices,]
class(train_data_full)
class(test_data_full)
# train_data_full <- cbind(train_labels, train_data)
# names(train_data_full)[names(train_data_full) == "train_labels"] <- "success"
# test_data_full <- cbind(test_labels, test_data)
# names(test_data_full)[names(test_data_full) == "test_labels"] <- "success"

# Convert to data frame if it's a list
if (!is.data.frame(train_data_full)) {
  train_data_full <- as.data.frame(train_data_full)
}

if (!is.data.frame(test_data_full)) {
  test_data_full <- as.data.frame(test_data_full)
}

# Define cross-validation control
ctrl <- trainControl(method = "cv",  # Cross-validation method
                     number = 5)      # Number of folds

## APPLYING GBM-------------

# Define the search grid for hyperparameters
# gbm_param_grid <- expand.grid(
#   n.trees = c(50, 100, 200),
#   interaction.depth = c(3, 5, 7),
#   shrinkage = c(0.01, 0.05, 0.1),
#   n.minobsinnode = c(10, 20, 30)
# )
# 
# # Train GBM model with hyper-parameter tuning
# gbm_tuned <- train(success ~ ., data = train_data_full, 
#                    method = "gbm",
#                    trControl = ctrl,
#                    tuneGrid = gbm_param_grid,
#                    distribution = "bernoulli",  # For binary classification
#                    verbose = FALSE)
# 
# # Print best tuning parameters
# print(gbm_tuned$bestTune)
# 
# gbm_pred <- predict(gbm_tuned, newdata = test_data_full)
# 
# # Evaluate the model
# gbm_confusion <- confusionMatrix(gbm_pred, test_data_full$success)
# 
# # Extract confusion matrix metrics
# gbm_accuracy <- gbm_confusion$overall['Accuracy']
# gbm_precision <- gbm_confusion$byClass['Precision']
# gbm_sensitivity <- gbm_confusion$byClass['Sensitivity']
# gbm_specificity <- gbm_confusion$byClass['Specificity']

## APPLYING XGBOOST-------------

# Define parameter grid for hyperparameter tuning
# xgb_param_grid <- expand.grid(
#   nrounds = c(200, 300, 400, 500),           # Number of boosting rounds (iterations)
#   max_depth = c(1, 2),                # Maximum tree depth
#   eta = c(0.3, 0.4),               # Learning rate
#   gamma = c(0),                # Minimum loss reduction required to make a further partition
#   colsample_bytree = c(1),     # Subsample ratio of columns when constructing each tree
#   min_child_weight = c(1),        # Minimum sum of instance weight (hessian) needed in a child
#   subsample = c(1)             # Subsample ratio of the training instance
# )
# 
# xgb_model <- train(train_data, train_labels, 
#                    method = "xgbTree",           # Use XGBoost method
#                    trControl = ctrl,             # Use defined training control
#                    tuneGrid = xgb_param_grid,        # Grid of hyperparameters to search
#                    metric = "Accuracy",          # Evaluation metric (accuracy for model selection)
#                    verbose = FALSE)              # Suppress verbose output
# 
# xgb_model$bestTune
# 
# # Make predictions on test data
# xgb_pred <- predict(xgb_model, newdata = test_data)
# 
# # Confusion matrix
# xgb_confusion <- confusionMatrix(data = xgb_pred, reference = test_labels)
# 
# # Calculate measures
# xgb_accuracy <- xgb_confusion$overall["Accuracy"]
# xgb_precision <- xgb_confusion$byClass["Precision"]
# xgb_sensitivity <- xgb_confusion$byClass["Sensitivity"]
# xgb_specificity <- xgb_confusion$byClass["Specificity"]

## APPLYING ADABOOST-------------

# ada_param_grid <- expand.grid(
#   mfinal = c(100, 200, 300, 400, 500),  # Number of iterations
#   interaction.depth = c(1, 2, 3),         # Maximum depth of each tree
#   shrinkage = c(0.01, 0.05, 0.1)          # Shrinkage parameter
# )

# adaboost_model <- boosting(success ~ ., 
#                            data = train_data_full, 
#                            boos = TRUE, 
#                            mfinal = 500)
# 
# # Make predictions on test data
# adaboost_pred <- predict(adaboost_model, newdata = test_data_full)
# 
# # CrossTable(adaboost_pred$class, test_data_full$success, dnn=c('Predict', 'Actual'),
# #            prop.chisq = FALSE, prop.t = FALSE, prop.r = FALSE)
# 
# # Create confusion matrix
# ada_confusion <- confusionMatrix(factor(adaboost_pred$class, levels = c("No", "Yes")), 
#                                   factor(test_data_full$success, levels = c("No", "Yes")))
# 
# # Extract metrics
# ada_accuracy <- ada_confusion$overall['Accuracy']
# ada_precision <- ada_confusion$byClass['Precision']
# ada_sensitivity <- ada_confusion$byClass['Sensitivity']
# ada_specificity <- ada_confusion$byClass['Specificity']

## ---APPLYING LOGISTIC REGRESSION-------------

lr_model <- glm(success ~ ., data = train_data_full, family = "binomial")
lr_pred <- predict(lr_model, newdata = test_data_full, type = "response")
lr_pred_binary <- ifelse(lr_pred > 0.5, "Yes", "No")
lr_pred_binary <- as.factor(lr_pred_binary)
lr_confusion <- confusionMatrix(lr_pred_binary, test_data_full$success)

lr_accuracy <- lr_confusion$overall['Accuracy']
lr_precision <- lr_confusion$byClass['Precision']
lr_sensitivity <- lr_confusion$byClass['Sensitivity']
lr_specificity <- lr_confusion$byClass['Specificity']


# Optimise Logistic Regression Model

# Define a parameter grid for tuning
lr_param_grid <- expand.grid(
  alpha = seq(0, 1, by = 0.1),  # Values for the alpha hyperparameter
  lambda = 10^seq(-3, 3, by = 0.1)  # Values for the lambda hyperparameter
)

# Define the control parameters for cross-validation
lr_ctrl <- trainControl(method = "cv",  # Cross-validation method (e.g., "cv" for k-fold cross-validation)
                        number = 5,     # Number of folds for cross-validation
                        verboseIter = TRUE,  # Display progress during tuning
                        returnData = FALSE,  # Don't return training data
                        returnResamp = "all",  # Return resampled results for all models
                        classProbs = TRUE)  # Calculate class probabilities

# Perform grid search for parameter tuning
lr_tuned <- train(
  success ~ .,  # Formula for the model
  data = train_data_full,  # Training dataset
  method = "glmnet",  # Machine learning method
  trControl = lr_ctrl,  # Cross-validation control parameters
  tuneGrid = lr_param_grid,  # Parameter grid for tuning
  family = "binomial"  # Family for logistic regression
)

best_alpha <- lr_tuned$bestTune$alpha
best_lambda <- lr_tuned$bestTune$lambda

train_matrix <- as.matrix(train_data_full[, -1])
test_matrix <- as.matrix(test_data_full[, -1])

lr_model_opt <- glmnet(x = train_matrix,
                       y = train_data_full$success,
                       alpha = best_alpha,
                       lambda = best_lambda,
                       family = "binomial")

lr_pred_opt <- predict(lr_model_opt, newx = test_matrix, type = "response")
lr_pred_binary <- ifelse(lr_pred_opt > 0.5, "Yes", "No")
lr_pred_binary <- as.factor(lr_pred_binary)

lr_confusion_opt <- confusionMatrix(lr_pred_binary, test_data_full$success)

lr_accuracy_opt <- lr_confusion_opt$overall['Accuracy']
lr_precision_opt <- lr_confusion_opt$byClass['Precision']
lr_sensitivity_opt <- lr_confusion_opt$byClass['Sensitivity']
lr_specificity_opt <- lr_confusion_opt$byClass['Specificity']

## APPLYING KNN MODEL-------------

# knn_pred <- knn(train = train_data, test = test_data, cl = train_labels, k=25) 
# # CrossTable(knn_pred, test_labels, prop.chisq = FALSE, 
# #            prop.t = FALSE, prop.r = FALSE, 
# #            dnn = c('Predicted', 'Actual'))
# 
# # Calculate confusion matrix
# knn_confusion_1 <- confusionMatrix(knn_pred, test_labels)
# 
# # Extract metrics
# knn_accuracy_1 <- knn_confusion_1$overall['Accuracy'] 
# knn_precision_1 <- knn_confusion_1$byClass['Precision'] 
# knn_sensitivity_1 <- knn_confusion_1$byClass['Sensitivity'] 
# knn_specificity_1 <- knn_confusion_1$byClass['Specificity']
# 
# # Optimizing KNN model with k=17
# knn_opt_pred <- knn(train = train_data, test = test_data, cl = train_labels, k=35) 
# # CrossTable(knn_opt_pred, test_labels, prop.chisq = FALSE, 
# #            prop.t = FALSE, prop.r = FALSE, 
# #            dnn = c('Predicted', 'Actual'))
# 
# # Calculate confusion matrix
# knn_confusion_2 <- confusionMatrix(knn_opt_pred, test_labels)
# 
# # Extract metrics
# knn_accuracy_2 <- knn_confusion_2$overall['Accuracy'] 
# knn_precision_2 <- knn_confusion_2$byClass['Precision'] 
# knn_sensitivity_2 <- knn_confusion_2$byClass['Sensitivity'] 
# knn_specificity_2 <- knn_confusion_2$byClass['Specificity']

## APPLYING NAIVE BAYES MODEL-------------
# nb_model <- naiveBayes(train_data, train_labels) 
# nb_test_pred <- predict(nb_model, test_data)
# # CrossTable(nb_test_pred, test_labels, prop.chisq = FALSE, 
# #            prop.t = FALSE, prop.r = FALSE, 
# #            dnn = c('Predicted', 'Actual'))
# 
# # Calculate confusion matrix
# nb_confusion <- confusionMatrix(nb_test_pred, test_labels)
# 
# # Extract metrics
# nb_accuracy <- nb_confusion$overall['Accuracy'] 
# nb_precision <- nb_confusion$byClass['Precision'] 
# nb_sensitivity <- nb_confusion$byClass['Sensitivity'] 
# nb_specificity <- nb_confusion$byClass['Specificity']
# 
# # Adding laplace smoothing for optimization
# nb_model_laplace <- naiveBayes(train_data, train_labels, laplace=1) 
# nb_test_laplace_pred <- predict(nb_model_laplace, test_data)
# # CrossTable(nb_test_laplace_pred, test_labels, prop.chisq = FALSE, 
# #            prop.t = FALSE, prop.r = FALSE, 
# #            dnn = c('Predicted', 'Actual'))
# 
# # Calculate laplace confusion matrix
# nb_confusion_opt <- confusionMatrix(nb_test_laplace_pred, test_labels)
# 
# # Extract laplace metrics
# nb_accuracy_opt <- nb_confusion_opt$overall['Accuracy'] 
# nb_precision_opt <- nb_confusion_opt$byClass['Precision'] 
# nb_sensitivity_opt <- nb_confusion_opt$byClass['Sensitivity'] 
# nb_specificity_opt <- nb_confusion_opt$byClass['Specificity']

## ---APPLYING DECISION TREE MODEL-------------

dt_model <- C5.0(train_data, train_labels) 
dt_model 
summary(dt_model) 
plot(dt_model)
dt_pred <- predict(dt_model, test_data) 
# CrossTable(dt_pred, test_labels, prop.chisq = FALSE, 
#            prop.c = FALSE, prop.r = FALSE, 
#            dnn = c('Predicted', 'Actual' ))

# Calculate confusion matrix
dt_confusion <- table(dt_pred, test_labels)

# Calculate metrics
dt_accuracy <- sum(diag(dt_confusion)) / sum(dt_confusion)
dt_precision <- dt_confusion[2, 2] / sum(dt_confusion[, 2]) 
dt_sensitivity <- dt_confusion[2, 2] / sum(dt_confusion[2, ]) 
dt_specificity <- dt_confusion[1, 1] / sum(dt_confusion[1, ])

# Optimise Decision Tree Model
dt_param_grid <- expand.grid(
  model = "tree",              # Specify the model type
  winnow = FALSE,              # Disable winnowing
  trials = 1:50                # Number of boosting iterations (trials)
)

# Train Decision Tree model with hyperparameter tuning
dt_model_opt <- train(
  success ~ .,                 # Formula for the model
  data = train_data_full,      # Training dataset
  method = "C5.0",             # Decision Tree method
  trControl = ctrl,            # Cross-validation control
  tuneGrid = dt_param_grid,    # Hyperparameter grid
  metric = "Accuracy"          # Evaluation metric
)

dt_model_opt$bestTune
# Make predictions on test data
dt_pred_opt <- predict(dt_model_opt, newdata = test_data_full)

# Create confusion matrix
dt_confusion_opt <- confusionMatrix(dt_pred_opt, test_data_full$success)

# Extract confusion matrix metrics
dt_accuracy_opt <- dt_confusion_opt$overall['Accuracy']
dt_precision_opt <- dt_confusion_opt$byClass['Precision']
dt_sensitivity_opt <- dt_confusion_opt$byClass['Sensitivity']
dt_specificity_opt <- dt_confusion_opt$byClass['Specificity']

## ---APPLYING SVM MODEL-------------

svm_model <- ksvm(as.matrix(train_data_full[-1]), train_data_full$success)
# Make predictions on test data
svm_pred <- predict(svm_model, as.matrix(test_data_full[-1]))
#CrossTable(svm_pred, test_labels, prop.chisq = FALSE, 
#           prop.c = FALSE, prop.r = FALSE, 
#           dnn = c('Predicted', 'Actual' ))

# Calculate confusion matrix
svm_confusion <- table(svm_pred, test_data_full$success)

# Calculate metrics
svm_accuracy <- sum(diag(svm_confusion)) / sum(svm_confusion) 
svm_precision <- svm_confusion[2, 2] / sum(svm_confusion[, 2]) 
svm_sensitivity <- svm_confusion[2, 2] / sum(svm_confusion[2, ]) 
svm_specificity <- svm_confusion[1, 1] / sum(svm_confusion[1, ])

# Optimizing SVM model using kernel type 'polydot'
svm_model_opt <- ksvm(as.matrix(train_data_full[-1]), train_data_full$success, 
                      kernel = "polydot")

# Make predictions on test data
svm_pred_opt <- predict(svm_model_opt, as.matrix(test_data_full[-1]))
# CrossTable(svm_opt_pred, test_labels, prop.chisq = FALSE, 
#            prop.c = FALSE, prop.r = FALSE, 
#            dnn = c('Predicted', 'Actual' ))

# Calculate confusion matrix
svm_confusion_opt <- table(svm_pred_opt, test_data_full$success)

# Calculate metrics
svm_accuracy_opt <- sum(diag(svm_confusion_opt)) / sum(svm_confusion_opt) 
svm_precision_opt <- svm_confusion_opt[2, 2] / sum(svm_confusion_opt[, 2]) 
svm_sensitivity_opt <- svm_confusion_opt[2, 2] / sum(svm_confusion_opt[2, ]) 
svm_specificity_opt <- svm_confusion_opt[1, 1] / sum(svm_confusion_opt[1, ])

## APPLYING NN MODEL-------------
# library(neuralnet)
# library(dplyr)
# #nn_model <- neuralnet(train_labels ~ ., data = train_data, linear.output = FALSE) 
# nn_model <- neuralnet(formula = success ~ ., data = train_data_full, linear.output = FALSE)
# plot(nn_model)
# 
# # Make predictions on test data
# nn_pred <- predict(nn_model, select(test_data_full, -success))
# summary(nn_pred)
# # Convert predictions to class labels (binary classification)
# nn_pred_labels <- apply(nn_pred, 1, function(row) { ifelse(row[1] > row[2], "No", "Yes") })
# 
# # CrossTable(nn_pred_labels, test_data_full$success, prop.chisq = FALSE,
# #            prop.c = FALSE, prop.r = FALSE, dnn = c('Predicted', 'Actual' ))
# 
# # Calculate confusion matrix
# nn_confusion <- table(nn_pred_labels, test_data_full$success)
# 
# # Calculate metrics
# nn_accuracy <- sum(diag(nn_confusion)) / sum(nn_confusion)
# nn_precision <- nn_confusion[2, 2] / sum(nn_confusion[, 2])
# nn_sensitivity <- nn_confusion[2, 2] / sum(nn_confusion[2, ])
# nn_specificity <- nn_confusion[1, 1] / sum(nn_confusion[1, ])

# Optimizing NN model
# nn_opt_model <- neuralnet(train_labels ~ ., data = train_data,
#                           linear.output = FALSE, hidden = c(5,2))
# plot(nn_opt_model)
# nn_opt_pred <- predict(nn_opt_model, test_data)
#
# # Binary classification
# nn_opt_pred_labels <- apply(nn_opt_pred, 1, function(row) { ifelse(row[1] > row[2], "No", "Yes") })
#
# CrossTable(nn_opt_pred_labels, test_labels, prop.chisq = FALSE,
#            prop.c = FALSE, prop.r = FALSE, dnn = c('Predicted', 'Actual' ))
#
# # Calculate confusion matrix
# nn_opt_confusion <- table(nn_opt_pred_labels, test_labels)
# # Calculate metrics
# nn_opt_accuracy <- sum(diag(nn_opt_confusion)) / sum(nn_opt_confusion)
# nn_opt_precision <- nn_opt_confusion[2, 2] / sum(nn_opt_confusion[, 2])
# nn_opt_sensitivity <- nn_opt_confusion[2, 2] / sum(nn_opt_confusion[2, ])
# nn_opt_specificity <- nn_opt_confusion[1, 1] / sum(nn_opt_confusion[1, ])

# nn_opt_accuracy <- 0.677595628415301
# nn_opt_precision <- 0.424242424242424
# nn_opt_sensitivity <- 0.571428571428571
# nn_opt_specificity <- 0.716417910447761

## ---APPLYING RANDOM FOREST MODEL-------------
# Train Random Forest model
rf_model <- randomForest(success ~ ., data = train_data_full)
summary(rf_model)
plot(rf_model)

# Make predictions on the test dataset
rf_pred <- predict(rf_model, newdata = test_data_full)

# CrossTable(rf_pred, test_data_full$success, prop.chisq = FALSE, 
#            prop.c = FALSE, prop.r = FALSE, 
#            dnn = c('Predicted', 'Actual' ))

# Create confusion matrix
rf_confusion <- table(rf_pred, test_data_full$success)

# Evaluate metrics
rf_accuracy <- sum(diag(rf_confusion)) / sum(rf_confusion)
rf_precision <- rf_confusion[2, 2] / sum(rf_confusion[, 2])
rf_sensitivity <- rf_confusion[2, 2] / sum(rf_confusion[2, ])
rf_specificity <- rf_confusion[1, 1] / sum(rf_confusion[1, ])


# Optimise Random Forest model

# Create model with default paramters
tunegrid <- expand.grid(mtry=c(1:20))

rf_default <- train(success ~ ., 
                    data = train_data_full, 
                    method="rf", 
                    metric = "Accuracy", 
                    tuneGrid = tunegrid, 
                    trControl = ctrl)
print(rf_default)
plot(rf_default)

rf_pred_opt <- predict(rf_default, newdata = test_data_full)
rf_confusion_opt <- table(rf_pred_opt, test_data_full$success)

rf_accuracy_opt <- sum(diag(rf_confusion_opt)) / sum(rf_confusion_opt)
rf_precision_opt <- rf_confusion_opt[2, 2] / sum(rf_confusion_opt[, 2])
rf_sensitivity_opt <- rf_confusion_opt[2, 2] / sum(rf_confusion_opt[2, ])
rf_specificity_opt <- rf_confusion_opt[1, 1] / sum(rf_confusion_opt[1, ])

# EVALUATE MODELS-------------

# values <- c(lr_accuracy, lr_precision, lr_sensitivity, lr_specificity,
#             lr_accuracy_opt, lr_precision_opt, lr_sensitivity_opt, lr_specificity_opt,
#             knn_accuracy_1, knn_precision_1, knn_sensitivity_1, knn_specificity_1,
#             knn_accuracy_2, knn_precision_2, knn_sensitivity_2, knn_specificity_2,
#             nb_accuracy, nb_precision, nb_sensitivity, nb_specificity,
#             nb_accuracy_opt, nb_precision_opt, nb_sensitivity_opt, nb_specificity_opt,
#             dt_accuracy, dt_precision, dt_sensitivity, dt_specificity,
#             dt_accuracy_opt, dt_precision_opt, dt_sensitivity_opt, dt_specificity_opt,
#             svm_accuracy, svm_precision, svm_sensitivity, svm_specificity,
#             svm_accuracy_opt, svm_precision_opt, svm_sensitivity_opt, svm_specificity_opt,
#             nn_accuracy, nn_precision, nn_sensitivity, nn_specificity,
#             nn_opt_accuracy, nn_opt_precision, nn_opt_sensitivity, nn_opt_specificity,
#             gbm_accuracy, gbm_precision, gbm_sensitivity, gbm_specificity,
#             xgb_accuracy, xgb_precision, xgb_sensitivity, xgb_specificity,
#             ada_accuracy, ada_precision, ada_sensitivity, ada_specificity,
#             rf_accuracy, rf_precision, rf_sensitivity, rf_specificity,
#             rf_accuracy_opt, rf_precision_opt, rf_sensitivity_opt, rf_specificity_opt)

values <- c(lr_accuracy, lr_precision, lr_sensitivity, lr_specificity,
            lr_accuracy_opt, lr_precision_opt, lr_sensitivity_opt, lr_specificity_opt,
            dt_accuracy, dt_precision, dt_sensitivity, dt_specificity,
            dt_accuracy_opt, dt_precision_opt, dt_sensitivity_opt, dt_specificity_opt,
            svm_accuracy, svm_precision, svm_sensitivity, svm_specificity,
            svm_accuracy_opt, svm_precision_opt, svm_sensitivity_opt, svm_specificity_opt,
            rf_accuracy, rf_precision, rf_sensitivity, rf_specificity,
            rf_accuracy_opt, rf_precision_opt, rf_sensitivity_opt, rf_specificity_opt)

# Convert values to percentages
values_percent <- scales::percent(as.numeric(values), accuracy = 0.001)

# Create matrix
result_matrix <- matrix(values_percent, nrow = 8, ncol = 4, byrow = TRUE)

# Add row and column names
colnames(result_matrix) <- c("Accuracy", "Precision", "Sensitivity", "Specificity") 
# rownames(result_matrix) <- c("Logistic Regression", "Logistic Regression - Optimised",
#                              "Decision Tree", "Decision Tree - Optimised", 
#                              "SVM", "SVM - Optimized", 
#                              "Random Forest", "Random Forest - Optimised")

# rownames(result_matrix) <- c("Logistic Regression", "Logistic Regression - Optimised",
#                              "KNN", "KNN - Optimised",
#                              "Naive Bayes", "Naive Bayes - Optimised",
#                              "Decision Tree", "Decision Tree - Optimised",
#                              "SVM", "SVM - Optimized",
#                              "Neural Network", "Neural Network - Optimised",
#                              "Gradient Boost",
#                              "XGBoost",
#                              "Adaptive Boost",
#                              "Random Forest", "Random Forest - Optimised")

rownames(result_matrix) <- c("Logistic Regression", "Logistic Regression - Optimised",
                             "Decision Tree", "Decision Tree - Optimised",
                             "SVM", "SVM - Optimized",
                             "Random Forest", "Random Forest - Optimised")

# Show the result matrix
print(result_matrix)
result_matrix
# Save Measures
write.csv(result_matrix, file = "Final Data/measures.csv", row.names = TRUE)


# Reshape the data for plotting
library(tidyr)
class(result_matrix)
result_matrix <- as.data.frame(result_matrix)
rownames(result_matrix)

model_metrics_long <- pivot_longer(result_matrix, cols = -Model, names_to = "Metric", values_to = "Value")

# Bar plot
ggplot(model_metrics_long, aes(x = Model, y = Value, fill = Metric)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(title = "Model Performance Comparison",
       y = "Metric Value") +
  theme_minimal() +
  theme(legend.position = "top")