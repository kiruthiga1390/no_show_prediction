# No_show Prediction
Kaggle data set consists of appointment information and whether the patient showed up for the appointment. There is huge loss in revenue and resources if the patient didn't show up for the medical appointment they booked. The aim of this project is to predict whether a patient will show up for the medical appointment or not. The prediction is done using python and model generated for prediction is Random Forest.

## Pre processing
1.Changed columns into numerical values ex: (Male,Female) (no,yes) to(0,1)
2.Removed outliers in age(-1 and 115)
3.Added new columns for better insights waiting time (appointment day - scheduled day),risk score (no.of appointments patient didnt show up/ total appointments patient booked), appointment day of week(monday to friday as 0 to 5)

## Visulaization
Visualization of every column with show and no show count using histograms

## Modelling
1.Split data into train and test set
2.Random forest is used for prediction

## Metrics
1.Accuracy
2.Classfication report
3.Confusion matrix
4.Feature importance