# Afroditi Doriti, Ubiqum, December 2018,
# Version 4, Wifi Locationing, Preprocessing + Models

# Model creation for building, floor, longitude and latitude according to
# WiFi signal in 3 buildings of the Universitat Jaume I.
# The data was colected between the 30.05.2013 and 20.06.2013.
# Validation dataset from 19.09.2013 to 08.10.2013.

# Data preprocessing: value for no signal from 100 becomes -105,
# deleted zero variance rows and columns, normalization per row
# Gradient boosted trees used for the predictions

# load libraries----
if (!require("pacman"))
  install.packages("pacman")
pacman::p_load("tidyverse", "plotly", "caret")

# set wd  and read data ----
# set wd
setwd("C:\\Users\\ASUS\\Documents\\Ubiqum\\4.3")
input_file <- "trainingData.csv" # training set
input_file2 <- "validationData.csv"# validation set

# read data
data <- read.csv(input_file) # training
validation <- read.csv(input_file2) # validation

# Clean up training data ----
# remove duplicates
data <- distinct(data)

# change type of data
# building id, user id, floor, space id, phone id, relative position
# become factors
for (i in (ncol(data) - 6):(ncol(data) - 1)) {
  data[, i] <- as.factor(data[, i])
}

# timestamp to time
data$TIMESTAMP <- as.POSIXct(data$TIMESTAMP, origin = "1970-01-01",
                             tz = "Europe/Madrid")

# make no signal value of 100 into -105
# to be comparable to the other values
data[data == 100] <- -105


# delete columns with zero variance
data_no_zero <- data[-as.numeric(which(apply(data, 2, var) == 0))]

# to avoid NaNs, delete all rows with zero variance
data_no_zero <-
  data_no_zero[-as.numeric(which(apply(data_no_zero[, 1:(nrow(data_no_zero) -
                                                           9)], 1, var) == 0)),]

# delete rows for which the value of the WAPS is more than -30
rows <- c(seq(
  from = 0,
  to = 0,
  length.out = nrow(data_no_zero)
))

for (i in 1:nrow(data_no_zero)) {
  for (j in 1:(ncol(data_no_zero) - 9)) {
    if (data_no_zero[i, j] > -30) {
      rows[i] <- 1
    }
  }
}

# rows to keep:
rows_keep <- rows != 1

data_no_zero <- data_no_zero[rows_keep,]

# keep only one of the repeated measurements

# find the duplicates
unique <-
  duplicated(data_no_zero[, (ncol(data_no_zero) - 9):(ncol(data_no_zero) -
                                                        1)])

# keep only the ones that are not duplicates
data_no_zero <- data_no_zero[!unique,]

# normalize per row
data_norm <-
  as.data.frame(t(apply(data_no_zero[, 1:(ncol(data_no_zero) - 9)], 1,
                        function(x)
                          (x - min(x)) / (max(x) - min(x)))))

data_norm <-
  data.frame(data_norm, data_no_zero[, (ncol(data_no_zero) - 8):ncol(data_no_zero)])

# backing up the data:
data_backup <- data_norm

# clean up validation data ----
# change type of data
# building id, user id, floor, space id, phone id, relative position
for (i in (ncol(validation) - 6):(ncol(validation) - 1)) {
  validation[, i] <- as.factor(validation[, i])
}

# timestamp
validation$TIMESTAMP <-
  as.POSIXct(validation$TIMESTAMP, origin = "1970-01-01",
             tz = "Europe/Madrid")

# make 100 into -105
validation[validation == 100] <- -105


# delete columns that we deleted in the original data
validation <-
  validation[-as.numeric(which(apply(data, 2, var) == 0))]

# normalize the validation data
validation_norm <-
  as.data.frame(t(apply(validation[, 1:(ncol(validation) - 9)], 1,
                        function(x)
                          ((x - min(x)) / (max(x) - min(x))
                          ))))
validation_norm <- data.frame(validation_norm,
                              validation[, (ncol(validation) - 8):ncol(validation)])

# save the data:
validation_backup <- validation_norm

# Models for building ----
# Create Data Partition
set.seed(1988)
intrain_buil_norm <-
  createDataPartition(y = data_norm$BUILDINGID,
                      p = 0.7,
                      list = FALSE)
training_buil_norm <- data_norm[intrain_buil_norm,]
testing_buil_norm <- data_norm[-intrain_buil_norm,]

# create the right training group
train_buil_norm <- data.frame(training_buil_norm$BUILDINGID,
                              training_buil_norm[, 1:(nrow(training_buil_norm) -
                                                        9)])


# Gradient boosted machine for building for data_norm
rfctrl <-
  trainControl(method = "repeatedcv",
               number = 3,
               repeats = 1)
set.seed(1987)
gbm_buil_norm <-
  train(
    training_buil_norm.BUILDINGID ~ .,
    data = train_buil_norm,
    method = "gbm",
    trControl = rfctrl,
    # preProcess = c("center", "scale"),
    tuneLength = 3
  )

# prediction of building for gbm
pred_gbm_buil_norm <-
  predict(gbm_buil_norm, newdata = testing_buil_norm)

# # confusion matrix
# confusionMatrix(pred_gbm_buil_norm, testing_buil_norm$BUILDINGID)

# Validation test for building ----

# knn
pred_knn_buil_norm_val <-
  predict(knn_buil_norm, newdata = validation_norm)

# Models for floor ----
# Create Data Partition
# substitute building with the gbm prediction, because
# it had the best accuracy
data_norm$BUILDINGID <- predict(gbm_buil_norm, data_norm)

set.seed(1988)
intrain_fl <-
  createDataPartition(y = data_norm$FLOOR, p = 0.7, list = FALSE)
training_fl <- data_norm[intrain_fl,]
testing_fl <- data_norm[-intrain_fl,]

# create the right training group
train_fl <- cbind(
  select(training_fl, contains("WAP")),
  select(training_fl, BUILDINGID),
  select(training_fl, FLOOR)
)
train_fl <- training_fl %>%
  select(contains("wap"), BUILDINGID, FLOOR)




# Gradient boosted machine for building
rfctrl <-
  trainControl(method = "repeatedcv",
               number = 3,
               repeats = 1)
set.seed(1987)
gbm_fl <- train(
  FLOOR ~ .,
  data = train_fl,
  method = "gbm",
  trControl = rfctrl,
  preProcess = c("center", "scale"),
  tuneLength = 3
)

# prediction of building for rf
pred_gbm_fl <- predict(gbm_fl, newdata = testing_fl)

# # confusion matrix
# confusionMatrix(pred_gbm_fl, testing_fl$FLOOR)

# Validation Floor ----
# substitute prediction for the building with the gbm prediction, because
# it had the best accuracy
validation_norm$BUILDINGID <- pred_gbm_buil_norm_val

# predict floor
# knn
pred_knn_fl_val <- predict(knn_fl, newdata = validation_norm)
confusionMatrix(pred_knn_fl_val, validation_norm$FLOOR)

# random forest
pred_rf_fl_val <- predict(rf_fl, newdata = validation_norm)
confusionMatrix(pred_rf_fl_val, validation_norm$FLOOR)

# gbm
pred_gbm_fl_val <- predict(gbm_fl, newdata = validation_norm)
confusionMatrix(pred_gbm_fl_val, validation_norm$FLOOR)

# Models for LONGITUDE ----

# First substitute the floor with the prediction:
# chose gbm model because when substituting with rf (higher accuracy),
# I got lower accuracy for longitude
data_norm$FLOOR <- predict(gbm_fl, newdata = data_norm)

# Create the right datasets
set.seed(1988)
intrain_lo <-
  createDataPartition(y = data_norm$LONGITUDE,
                      p = 0.7,
                      list = FALSE)
training_lo <- data_norm[intrain_lo,]
testing_lo <- data_norm[-intrain_lo,]

# create the right training group
train_lo <- cbind(
  select(training_lo, contains("WAP")),
  select(training_lo, BUILDINGID),
  select(training_lo, FLOOR),
  select(training_lo, LONGITUDE)
)

# Gradient boosted machine for longitude
rfctrl <-
  trainControl(method = "repeatedcv",
               number = 3,
               repeats = 1)
set.seed(1987)
gbm_lo <- train(
  LONGITUDE ~ .,
  data = train_lo,
  method = "gbm",
  trControl = rfctrl,
  preProcess = c("center", "scale"),
  tuneLength = 3
)

# prediction of longitude for gbm
pred_gbm_lo <- predict(gbm_lo, newdata = testing_lo)

# postResample(pred_gbm_lo, testing_lo$LONGITUDE)

# Validation longitude ----
# substitute prediction for the floor
validation_norm$FLOOR <- pred_gbm_fl_val


# gbm
pred_gbm_lo_val <- predict(gbm_lo, newdata = validation_norm)
# postResample(pred_gbm_lo_val, validation_norm$LONGITUDE)

# Models for latitude ----
# First substitute the longitude with the gbm model (higher accuracy)
data_norm$LONGITUDE <- predict(gbm_lo, newdata = data_norm)

# Create the right datasets
set.seed(1988)
intrain_la <-
  createDataPartition(y = data_norm$LATITUDE,
                      p = 0.7,
                      list = FALSE)
training_la <- data_norm[intrain_la,]
testing_la <- data_norm[-intrain_la,]

# create the right training group
train_la <- cbind(
  select(training_la, contains("WAP")),
  select(training_la, BUILDINGID),
  select(training_la, FLOOR),
  select(training_la, LONGITUDE),
  select(training_la, LATITUDE)
)

# Gradient boosted machine for longitude
rfctrl <-
  trainControl(method = "repeatedcv",
               number = 3,
               repeats = 1)
set.seed(1987)
gbm_la <- train(
  LATITUDE ~ .,
  data = train_la,
  method = "gbm",
  trControl = rfctrl,
  preProcess = c("center", "scale"),
  tuneLength = 3
)

# prediction of longitude for rf
pred_gbm_la <- predict(gbm_la, newdata = testing_la)

# postResample(pred_gbm_la, testing_la$LATITUDE)

# Validation for latitude ----
# substitute prediction for the floor
validation_norm$LONGITUDE <- pred_knn_lo_val

# gbm
pred_gbm_la_val <- predict(gbm_la, newdata = validation_norm)
postResample(pred_gbm_la_val, validation_norm$LATITUDE)

# Visualizations and distance prediction ----

#ggplot
validation_norm$LATITUDE <- pred_knn_la_val
df_val <-
  data.frame(select(validation_backup, contains("WAP")),
             select(validation_norm, contains("WAP")))
ggplot(data = df_val, aes(x = LONGITUDE, y = LATITUDE)) + geom_point() +
  geom_point(aes(x = LONGITUDE.1, y = LATITUDE.1), colour = "red")

# calculating the distance according to pythagoras
df_val$DISTANCE <- ((df_val$LONGITUDE - df_val$LONGITUDE.1) ^ 2 +
                      (df_val$LATITUDE - df_val$LATITUDE.1) ^ 2) ^ (1 /
                                                                      2)


# plotly

# create right df
# create variableto identify prediction and real data, old,
# 1 for real data, 0 for prediciton
validation_backup$old <- c(1)
validation_norm$old <- c(0)

df_val2 <-
  rbind(select(validation_backup, -contains("WAP")),
        select(validation_norm,-contains("WAP")))
df_val2$old <- as.factor(df_val2$old)

PredictedVSReal3D <-
  plot_ly(
    df_val2,
    x = ~ LONGITUDE,
    y = ~ LATITUDE,
    z = ~ FLOOR,
    opacity = 0.5,
    color = ~ old,
    colors = c('#BF382A', '#0C4B8E'),
    size = 0.3
  ) %>%
  add_markers() %>%
  layout(scene = list(
    xaxis = list(title = 'Longitude'),
    yaxis = list(title = 'Latitude'),
    zaxis = list(title = 'Floor')
  ))

PredictedVSReal3D # 3D plot
