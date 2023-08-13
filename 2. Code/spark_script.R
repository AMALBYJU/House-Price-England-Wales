# SPARK SCRIPT

install.packages('aws.s3')
install.packages('sparklyr')
install.packages('dplyr')
install.packages('forecast')
install.packages('ggplot2')

library(sparklyr)
library(dplyr)
library(forecast)
library(ggplot2)
library(aws.s3)

Sys.setenv("AWS_ACCESS_KEY_ID" = "",
           "AWS_SECRET_ACCESS_KEY" = "")

# Connect to spark
sc <- spark_connect(master = "yarn-client", 
                    spark_home = "/usr/lib/spark/")

# Read the house prices csv file
pp_2002_2021 <- spark_read_csv(sc, name = 'pp_2002_2021', 
                               path = 's3://<bucket_name>/<path>')

time_series_price_income <- 
  spark_read_csv(sc, name = 'time_series_price_income', 
                 path = 's3://<bucket_name>/<path>')

# ------------------------------------------------------------------------------------------------
# DATA CLEANING 
house_data <-
  pp_2002_2021 %>%
  mutate(
    logPrice = log(Price),
    Date_of_Transfer_num = as.numeric(Date_of_Transfer),
    Property_Type = case_when(
      Property_Type == 'D' ~ 'Detached',
      Property_Type == 'F' ~ 'Flats/Maisonettes',
      Property_Type == 'O' ~ 'Other',
      Property_Type == 'S' ~ 'Semi-Detached',
      Property_Type == 'T' ~ 'Terraced'
    ),
    Old_New = case_when(
      Old_New == 'N' ~ 'Old',
      Old_New == 'Y' ~ 'New'
    )) %>%
  select(logPrice, Date_of_Transfer, Date_of_Transfer_num, Price, Property_Type, 
         Old_New, Duration, County) %>%
  copy_to(sc, ., name = "house_data")

house_data_cluster <- 
  house_data %>%
  mutate(
    Date_of_Transfer = as.numeric(Date_of_Transfer)
  ) %>%
  ft_string_indexer("Property_Type", "Property_Type_Int") %>%
  ft_string_indexer("Old_New", "Old_New_Int") %>%
  ft_string_indexer("Duration", "Duration_Int") %>%
  ft_string_indexer("County", "County_Int") %>%
  select(Price, Date_of_Transfer, Property_Type_Int, Old_New_Int, Duration_Int, County_Int)

# -----------------------------------------------------------------------------------------------
# EXPLORATORY DATA ANALYSIS

# Line chart - mean price vs date
house_data %>%
  mutate(year = year(Date_of_Transfer)) %>%
  group_by(year) %>%
  summarize(Mean_Price = mean(Price)) %>%
  ggplot(aes(x = year, y = Mean_Price)) +
  geom_line() +
  scale_y_continuous(labels=scales::dollar_format(prefix = '£')) +
  labs(x = "Year", 
       y = "Mean Price",
       title = "Line chart - Mean Price vs Date")

# Line chart - Number of Transactions vs Date
house_data %>%
  group_by(year(Date_of_Transfer)) %>%
  summarize(Num_Transactions = n()) %>%
  ggplot(aes(x = `year(Date_of_Transfer)`, y = Num_Transactions)) +
  geom_line() +
  labs(x = 'Number of Transactions',
       y = 'Year',
       title = 'Line chart - Number of Transactions vs Date')

# Bar chart
house_data %>%
  filter(Property_Type != 'Other') %>%
  group_by(Old_New, Property_Type) %>%
  summarize(Mean_Price = mean(Price)) %>%
  ggplot(aes(x = Property_Type, y = Mean_Price)) +
  geom_col() +
  scale_y_continuous(labels=scales::dollar_format(prefix = '£')) +
  facet_wrap(~Old_New) +
  labs(x = "Property Type",
       y = "Mean Price")

county_mean_prices <-
  house_data %>%
  group_by(County) %>%
  summarize(MeanPrice = mean(Price))

# Top 5 and bottom 5 counties with mean prices
rbind(
  county_mean_prices %>%
    slice_max(MeanPrice, n = 5),
  county_mean_prices %>%
    slice_min(MeanPrice, n = 5)
) %>%
  arrange(desc(MeanPrice)) %>%
  ggplot(aes(x = reorder(County, MeanPrice), y = MeanPrice)) +
  geom_col() +
  scale_y_continuous(labels=scales::dollar_format(prefix = '£')) +
  coord_flip() +
  labs(x = 'County',
       y = 'Mean Price',
       title = 'Top 5 and Bottom 5 counties with mean prices')

# Line plot of increasing prices for each property type
house_data %>%
  filter(Property_Type != 'Other') %>%
  group_by(Property_Type, year(Date_of_Transfer)) %>%
  summarize(Mean_Price = mean(Price)) %>%
  ggplot(aes(x = `year(Date_of_Transfer)`, y = Mean_Price, color = Property_Type)) +
  geom_line() +
  scale_y_continuous(labels=scales::dollar_format(prefix = '£')) +
  labs(x = 'Year',
       y = 'Mean Price',
       title = 'Increase in mean price for each property type')

# Pie chart of number of transactions for each property type
house_data %>%
  filter(Property_Type != 'Other') %>%
  ggplot(aes(x = 1, fill = Property_Type)) +
  geom_bar(width = 0.1)+
  coord_polar(theta = "y") +
  theme(line = element_blank(),
        axis.ticks = element_blank(),
        axis.text = element_blank(),
        axis.title = element_blank())

# Line chart of house price vs income
colors <- c("Mean Equivalised Disposable Income" = "red", "Mean House Price" = "blue")
ggplot(time_series_price_income, aes(x = year, y = meanPrice)) +
  geom_line(aes(color = "Mean House Price")) +
  geom_line(data = time_series_price_income, aes(x = year, y = meanEquivalisedDisposableIncome, color = "Mean Equivalised Disposable Income")) +
  labs(x = "Year",
       y = "Pounds",
       color = "Legend") +
  scale_y_continuous(labels=scales::dollar_format(prefix = '£')) +
  scale_color_manual(values = colors)

# ----------------------------------------------------------------------------------------------
# EXPLANATORY MODEL

# Elbow analysis for k-means
cost_vector = rep(0, 10)
cost_vector[1] = NA

for(i in 2:10) {
  model <- ml_kmeans(house_data_cluster, features = colnames(house_data_cluster), k = i)
  cost_vector[i] <- model$cost
}

plot(1:10, cost_vector, type = 'b', xlab = 'Number of clusters', ylab = 'Cost')

# k = 3 as the elbow occurs here
k_means_model <- ml_kmeans(house_data_cluster, features = colnames(house_data_cluster), k = 3)
k_means_model

# ----------------------------------------------------------------------------------------------
# REGRESSION MODELS

# train-test split
partitions <-
  house_data %>%
  sdf_random_split(training = 0.7, testing = 0.3)

train <- partitions$training %>% copy_to(sc, ., name = "train")
test <- partitions$testing %>% copy_to(sc, ., name = "test")

# Random Forest Model
(formula_rf <- logPrice ~ Date_of_Transfer_num+Property_Type+Old_New+Duration+County)
rf_model <- ml_random_forest(train, 
                             formula_rf, 
                             num_trees = 100, 
                             max_depth = 3,
                             type = "regression")
glance(rf_model)

# Gradient Boosted Tree
(formula_gbt <- logPrice ~ Date_of_Transfer_num+Property_Type+Old_New+Duration+County)
gbt_model <- ml_gbt_regressor(train, 
                              formula_gbt, 
                              max_iter = 100, 
                              max_depth = 3)
glance(gbt_model)

# Model Evaluation
test_y <- test['logPrice'] %>% pull()
sd_test_y <- sd(test_y)

predicted_y_rf <- predict(rf_model, test)
resid_rf <- predicted_y_rf - test_y
rf_rmse <- sqrt(mean(resid_rf ^ 2, na.rm = TRUE)) # Random Forest RMSE

predicted_y_gbt <- predict(gbt_model, test)
resid_gbt <- predicted_y_gbt - test_y
gbt_rmse <- sqrt(mean(resid_gbt ^ 2, na.rm = TRUE)) # Gradient Boosted Tree RMSE

tribble(
  ~model, ~RMSE, ~sd,
  "Random Forest", rf_rmse, sd_test_y,
  "Gradient Boosted Tree", gbt_rmse, sd_test_y       # RMSE vs standard deviation
)

ml_tree_feature_importance(gbt_model) # Feature importance

# ----------------------------------------------------------------------------------------------
# TIME SERIES FORECASTING

time_series <-
  house_data %>%
  mutate(year = year(Date_of_Transfer)) %>%
  group_by(year) %>%
  summarize(meanPrice = mean(Price),
            transaction_count = n()) %>%
  arrange(year) %>%
  copy_to(sc, ., name = "time_series")

train_time_series <-
  time_series %>%
  filter(year <= 2019) %>%
  arrange(year) %>%
  copy_to(sc, ., name = "train_time_series")

test_time_series <-
  time_series %>%
  filter(year > 2019) %>%
  arrange(year) %>%
  copy_to(sc, ., name = "test_time_series")

ts_meanPrice <- time_series['meanPrice'] %>% 
  pull() %>%
  ts(., start = c(2002), end = c(2021), frequency = 1)

train_ts_meanPrice <- train_time_series['meanPrice'] %>% 
  pull() %>%
  ts(., start = c(2002), end = c(2019), frequency = 1)

test_ts_meanPrice <- test_time_series['meanPrice'] %>% 
  pull() %>%
  ts(., start = c(2020), end = c(2021), frequency = 1)

# TBATS model for forecasting future average price
tbats_model <- tbats(train_ts_meanPrice)
summary(tbats_model)

# Forecast on test data
fore_tbats = forecast::forecast(tbats_model, h=2)
df_tbats = as.data.frame(fore_tbats)

# Plot the results
colors <- c("Mean Price (2020 - 2021)" = "red", "Predicted Mean Price (2020-2021)" = "blue")
ggplot(as.data.frame(ts_meanPrice), aes(time(ts_meanPrice), x)) +
  geom_line() +
  geom_line(data = as.data.frame(test_ts_meanPrice), aes(time(test_ts_meanPrice), x, color = "Mean Price (2020 - 2021)")) +
  geom_line(data = df_tbats, aes(time(test_ts_meanPrice), `Point Forecast`, color = "Predicted Mean Price (2020-2021)")) +
  labs(x = "Year",
       y = "Mean Price",
       color = "Legend",
       title = "Time Series Forecasting of Mean House Price"
  ) +
  scale_y_continuous(labels=scales::dollar_format(prefix = '£')) +
  scale_color_manual(values = colors)

residual <- df_tbats$`Point Forecast` - test_ts_meanPrice
sqrt(mean(residual^2))         # RMSE of the TBATS model

# Disconnect from spark
spark_disconnect(sc)

