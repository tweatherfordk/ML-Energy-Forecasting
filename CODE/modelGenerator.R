library(openxlsx)
library(dplyr)
library(tidyr)

data = read.xlsx("2015_2024_Elec_Net_Gen_Data.xlsx")

### Format Data and prepare data for time series analysis and modelling ###
data_clean = data[,c(6,14:25, 28)]

df_grouped = data_clean %>% group_by(YEAR, Plant.State) %>% summarise(across(c(1:12), sum, na.rm = TRUE), .groups = "keep")  %>% 
  rename(
    Year = YEAR,
    State = Plant.State
  )

colnames(df_grouped) = c("Year", "State", "January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December")

df_long = df_grouped %>% 
  pivot_longer(cols = January:December, 
               names_to = "Month", 
               values_to = "Total.Generation") %>% 
  mutate(Date = as.Date(paste(Year, Month, "1"), format = "%Y %B %d")) %>%
  mutate(Date = format(Date, "%m/%d/%Y")) %>% 
  select(Year, Date, State, Total.Generation)

gendata = df_long[, !(names(df_long) == "Year")]
gendata$Date = as.Date(gendata$Date, format = "%m/%d/%Y")
gendata

#library(tseries)
library(xts)
library(zoo)
gendata_CA = gendata[gendata$State == "CA",]
genCA_ts <- ts(gendata_CA$Total.Generation, start = c(2015, 1), frequency = 12)
plot(genCA_ts, ylab = "Generation Value (MW)", main = "California Electricity Generation Data", col="blue")
acf(genCA_ts, ylab = "ACF Value", main = "ACF - Electricity Generation Data")

genCA_ts_diff = diff(genCA_ts)
plot(genCA_ts_diff)
acf(genCA_ts_diff)
gendata_train_CA = (gendata[gendata$State == "CA" & gendata$Date < '2024-01-01',])[, c(1,3)]
gendata_test_CA = (gendata[gendata$State == "CA" & gendata$Date >= '2024-01-01',])[, c(1,3)]
gendata_train_CA
gendata_test_CA

library(forecast)
p = 5
q = 5
d = 2

# Build ARIMA model for CA state
opt_arima_model.CA = auto.arima(gendata_train_CA$Total.Generation, max.p = p, max.d = d, max.q = q, stepwise = TRUE, method = "ML")
orders.CA = arimaorder(opt_arima_model.CA)
print(orders.CA)

# Forecasting with ARIMA model
forecast_opt_arima_model.CA = predict(opt_arima_model.CA, n.ahead=12)
ubound_opt_arima_model95.CA = forecast_opt_arima_model.CA$pred+1.96*forecast_opt_arima_model.CA$se
lbound_opt_arima_model95.CA = forecast_opt_arima_model.CA$pred-1.96*forecast_opt_arima_model.CA$se

# Plot observed vs predicted with 95% prediction intervals
ts.plot(ts(gendata_test_CA$Total.Generation, start = c(2024, 1), frequency = 12), col = "black", type = "l", lwd = 2, 
        ylim = c(min(lbound_opt_arima_model95.CA), max(ubound_opt_arima_model95.CA)),
        xlab = "Date", ylab = "Electricity Generation", 
        main = "ARIMA Forecast vs Observed Electricity Generation")
points(ts(gendata_test_CA$Total.Generation, start = c(2024, 1), frequency = 12), col = "black")
lines(ts(forecast_opt_arima_model.CA$pred, start = c(2024, 1), frequency = 12), col = "red",lwd = 2)
points(ts(forecast_opt_arima_model.CA$pred, start = c(2024, 1), frequency = 12), col = "red")
lines(ts(ubound_opt_arima_model95.CA, start = c(2024, 1), frequency = 12), col = "blue",lwd = 2)
lines(ts(lbound_opt_arima_model95.CA, start = c(2024, 1), frequency = 12), col = "blue",lwd = 2)
legend('topright', legend = c("Observed E-Gen", "Predicted E-Gen", 
                              "Confidence Interval"), lwd = 2, 
       col = c("black", "red", "blue"), cex=0.50)

# Prediction accuracy
mape.ARIMA.CA = mean(abs(gendata_test_CA$Total.Generation-forecast_opt_arima_model.CA$pred)/gendata_test_CA$Total.Generation)
pm.ARIMA.CA = sum((gendata_test_CA$Total.Generation-forecast_opt_arima_model.CA$pred)^2) /
  sum((gendata_test_CA$Total.Generation - mean(gendata_test_CA$Total.Generation))^2)
rmse.ARIMA.CA = sqrt(mean((gendata_test_CA$Total.Generation-forecast_opt_arima_model.CA$pred)^2))
mae.ARIMA.CA = mean(abs(gendata_test_CA$Total.Generation-forecast_opt_arima_model.CA$pred))
me.ARIMA.CA = mean(gendata_test_CA$Total.Generation-forecast_opt_arima_model.CA$pred)
cat("Error Values for CA E-Gen ARIMA model: \n")
cat("MAPE: ", mape.ARIMA.CA)
cat("\tPM: ", pm.ARIMA.CA)
cat("\tRMSE: ", rmse.ARIMA.CA)
cat("\tMAE: ", mae.ARIMA.CA)
cat("\tME: ", me.ARIMA.CA)

# 4 Years prediction into future for CA
forecast5_ARIMA.CA = predict(opt_arima_model.CA, n.ahead=60)
n = length(forecast5_ARIMA.CA$pred)
start_date = as.Date("2024-01-01")
dates = seq(from=start_date, by="month", length.out=n)
forecast5_df.CA = data.frame(Date=dates, CA=forecast5_ARIMA.CA$pred)
write.csv(forecast5_df.CA, "Forecast_5Years.csv", row.names=FALSE)

stList = as.list(unique(gendata$State))
gendata_train = list()
gendata_test = list()
for(st in stList) {
  gendata_train[[st]] = (gendata[gendata$State == st & 
                                   gendata$Date < '2024-01-01',])[, c(1,3)]
  gendata_test[[st]] = (gendata[gendata$State == st & 
                                  gendata$Date >= '2024-01-01',])[, c(1,3)]
}

opt_arima_model = list()
for(st in stList) {
  opt_arima_model[[st]] = auto.arima(gendata_train[[st]]$Total.Generation, 
                                     max.p = p, max.d = d, max.q = q, 
                                     stepwise = TRUE, method = "ML")
}

# Plot observed vs predicted with 95% prediction intervals & Prediction accuracy
pdf("PredictionPlot_AllStates.pdf", width = 8, height = 6)
pred_acc_df = data.frame("State"=character(), "MAPE"=numeric(), "PM"=numeric(),
                         "RMSE"=numeric(), "MEA"=numeric(), "ME"=numeric(),
                         stringsAsFactors = FALSE)
forecast1_ARIMA = list()
for(st in stList) {
  forecast1_ARIMA[[st]] = predict(opt_arima_model[[st]], n.ahead=12)
  
  ubound_opt_arima_model95 = forecast1_ARIMA[[st]]$pred+1.96*forecast1_ARIMA[[st]]$se
  lbound_opt_arima_model95 = forecast1_ARIMA[[st]]$pred-1.96*forecast1_ARIMA[[st]]$se
  
  heading = paste("ARIMA Forecast vs Observed Electricity Generation - ", st)
  
  ts.plot(ts(gendata_test[[st]]$Total.Generation, start = c(2024, 1), 
             frequency = 12), col = "black", type = "l", lwd = 2, 
          ylim = c(min(lbound_opt_arima_model95), 
                   max(ubound_opt_arima_model95)),
          xlab = "Date", ylab = "Electricity Generation", 
          main = heading)
  points(ts(gendata_test[[st]]$Total.Generation, start = c(2024, 1), 
            frequency = 12), col = "black")
  lines(ts(forecast1_ARIMA[[st]]$pred, start = c(2024, 1), 
           frequency = 12), col = "red",lwd = 2)
  points(ts(forecast1_ARIMA[[st]]$pred, start = c(2024, 1), 
            frequency = 12), col = "red")
  lines(ts(ubound_opt_arima_model95, start = c(2024, 1), 
           frequency = 12), col = "blue",lwd = 2)
  lines(ts(lbound_opt_arima_model95, start = c(2024, 1), 
           frequency = 12), col = "blue",lwd = 2)
  legend('topright', legend = c("Observed E-Gen", "Predicted E-Gen", 
                                "Confidence Interval"), lwd = 2, 
         col = c("black", "red", "blue"), cex=0.50)
  
  mape.ARIMA = mean(abs(gendata_test[[st]]$Total.Generation-forecast1_ARIMA[[st]]$pred)/
                      gendata_test[[st]]$Total.Generation)
  pm.ARIMA = sum((gendata_test[[st]]$Total.Generation-forecast1_ARIMA[[st]]$pred)^2) /
    sum((gendata_test[[st]]$Total.Generation - mean(gendata_test[[st]]$Total.Generation))^2)
  rmse.ARIMA = sqrt(mean((gendata_test_CA$Total.Generation-forecast1_ARIMA[[st]]$pred)^2))
  mae.ARIMA = mean(abs(gendata_test[[st]]$Total.Generation-forecast1_ARIMA[[st]]$pred))
  me.ARIMA = mean(gendata_test[[st]]$Total.Generation-forecast1_ARIMA[[st]]$pred)
  pred_acc_df = rbind(pred_acc_df, data.frame(st, mape.ARIMA, pm.ARIMA, 
                                              rmse.ARIMA, mae.ARIMA,
                                              me.ARIMA))
}

names(pred_acc_df)[names(pred_acc_df) == "st"] = "State"
names(pred_acc_df)[names(pred_acc_df) == "mape.ARIMA"] = "MAPE"
names(pred_acc_df)[names(pred_acc_df) == "pm.ARIMA"] = "PM"
names(pred_acc_df)[names(pred_acc_df) == "rmse.ARIMA"] = "RMSE"
names(pred_acc_df)[names(pred_acc_df) == "mae.ARIMA"] = "MAE"
names(pred_acc_df)[names(pred_acc_df) == "me.ARIMA"] = "ME"

dev.off()
write.csv(pred_acc_df, "Prediction_Accuracy_AllStates.csv", row.names=FALSE)

start_date = as.Date("2024-01-01")
dates = seq(from=start_date, by="month", length.out=60)
forecast5_df = data.frame(Date=dates)

forecast5_ARIMA = list()
for(st in stList) {
  forecast5_ARIMA[[st]] = predict(opt_arima_model[[st]], n.ahead=60)
  forecast5_df = cbind(forecast5_df, data.frame(temp=round(forecast5_ARIMA[[st]]$pred)))
  names(forecast5_df)[names(forecast5_df) == "temp"] = st
}
write.csv(forecast5_df, "Forecast_5Years_AllStates.csv", row.names=FALSE)