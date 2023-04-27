# Setup # =======================================
rm(list = ls())

if (!require(caret)) {install.packages("caret"); library(caret)}
if (!require(dplyr)) {install.packages("dplyr"); library(dplyr)}
if (!require(ggplot2)) {install.packages("ggplot2"); library(ggplot2)}
if (!require(ModelMetrics)) {install.packages("ModelMetrics"); library(ModelMetrics)}
if (!require(xgboost)) {install.packages("xgboost"); library(xgboost)}
if (!require(glue)) {install.packages("glue"); library(glue)}
if (!require(doParallel)) {install.packages("doParallel"); library(doParallel)}
if (!require(lubridate)) {install.packages("lubridate"); library(lubridate)}
if (!require(bit64)) {install.packages("bit64"); library(bit64)}
if (!require(data.table)) {install.packages("data.table"); library(data.table)}
if (!require(bst)) {install.packages("bst"); library(bst)}
if (!require(geosphere)) {install.packages("geosphere"); library(geosphere)}
#library(numbers)

set.seed(404)
cl = makePSOCKcluster(8)
registerDoParallel(cl)

taxi_data = read.csv("train.csv", nrows = 5000000) #5,000,000
str(taxi_data)

taxi_data$pickup_datetime = as.POSIXct(taxi_data$pickup_datetime)
taxi_data$hour = ymd_hms(taxi_data$pickup_datetime) |> lubridate::hour()
taxi_data$weekday = taxi_data$pickup_datetime |> weekdays() |> factor() |> as.numeric()

taxi_data$hour_cos = cos(2*pi*taxi_data$hour/24)
taxi_data$hour_sin = sin(2*pi*taxi_data$hour/24)

taxi_data$pickup_longitude = sapply(taxi_data$pickup_longitude, function(x) min(max(x,-180),180))
taxi_data$pickup_latitude = sapply(taxi_data$pickup_latitude, function(x) min(max(x,-90),90))
taxi_data$dropoff_longitude = sapply(taxi_data$dropoff_longitude, function(x) min(max(x,-180),180))
taxi_data$dropoff_latitude = sapply(taxi_data$dropoff_latitude, function(x) min(max(x,-90),90))

taxi_data$haversine = distHaversine(
    as.matrix(taxi_data[c("pickup_longitude", "pickup_latitude")]),
    as.matrix(taxi_data[c("dropoff_longitude", "dropoff_latitude")])
    )

taxi_y = taxi_data$fare_amount
taxi_x = taxi_data[c("pickup_longitude", "pickup_latitude",
                "dropoff_longitude", "dropoff_latitude",
                "passenger_count", "hour_cos", "hour_sin",
                "weekday", "haversine")]

train_index = sample(1:nrow(taxi_x), nrow(taxi_x)*0.8)



# Model Training # ==============================
taxi_x_matrix = as.matrix(taxi_x)
taxi_y_matrix = as.matrix(taxi_y)

boosted_tree_mod = xgboost(data = taxi_x_matrix[train_index,],
                           label = taxi_y_matrix[train_index],
                           max_depth = 10, eta = 0.4,
                           nrounds = 12, objective = "reg:squarederror")
# data: dataframe x
# label: dataframe y
# max_depth: (max splits) - 1
#   best depth has chance to use all columns
# eta: counteracts overfitting. higher = more
#   slightly higher than base due to depth
# nrounds: trees to chain
#   increased until plateau
# objective: objective function

pred = predict(boosted_tree_mod, taxi_x_matrix[-train_index,])
rmse(taxi_y_matrix[-train_index], pred) #4.5737


# Cleanup # =====================================
stopCluster(cl)
