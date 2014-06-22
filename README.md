MLJH
====
This is a repository for the Johns Hopkins Pracctical Machine Learning Coursera course project in June of 2014. 
The entire project is presented in writeup.md and all the other files exist to supprt this writeup.

The dataset is the WLE dataset courtesy of:

Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. Qualitative Activity Recognition of Weight Lifting Exercises. 
Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13) . 
Stuttgart, Germany: ACM SIGCHI, 2013.
Read more: http://groupware.les.inf.puc-rio.br/har#ixzz35OFtlFsQ

In brief, the WLE dataset was taken from sensors attached to six volunteers' bodies (on the arm, forearm, and belt) and 
one sensor on the dumbbell while the volunteers were asked to perform dumbbell lifting exercises.  These sensor measured 
the x, y, and z dimensions of accelerometer, gyroscope, and magnetometer at 45 Hz. The observations were labelled 
A, B, C, D, or E which corresponds to identifying the proper movement (A) and movement in improper form (B through E).

The project is to predict the proper movement (dependent variable classe = A, B, C, D, E) using the observed variables.

As with most machine learning projects, much of the effort was spent cleaning the data which entails a good amount of
exploratory plotting and testing by other means.  I was particularly concerned with leakage in the feature set.

Since the project was to try to predict proper movement of an exercise (i.e. the proper lifting of a dumbbell) using
sensor data, it makes sense to exclude non-sensor data such as indices, names, time-stamps, and data collection "book-keeping"
indices.  These types of variables pose the danger of leakage whch allows the machine learner to train and test very well
in a laboratory setting but will run very poorly when deployed as these leakage variables will not be available to the
machine.

A good portion of the data cleaning effort was put into exploring the effects of the indexing and house-keeping variables
for leakage effects.  Eventually I decided to do away with all of them just to be safe.

In addition, there were a good number of variables that were summary statistics (e.g. mean, kurtosis, skew) of a set of 
observations -- each row of observation was a part of a sliding window of time series sensor data.  These variables were 
sparse with most rows being blank or NA, some had #Div/0! warnings.  I decided to NOT impute the blanks, NA's, and 
division by 0's, since they can pose leakage effects as well.

Finally, the machine learner was trained using Random Forest (tm) of 1000 trees.  It worked well, with prediction accuracy
of over 99% in two different forests of different training and testing sets.

I also talked about stationarity of the data (since it was a time series) and scaling, centering, and normalising the 
feature set.  None of these were deemed necesssary (random forest does not require too much of scaling, centering and 
normalising.  The time series data did not seem to have non-mean reverting trends.

Finally I explored the idea that feature set selection can be done by reading the *importance* tables -- this is similar
to ussing a PCA method of feature selection for regression learners.  I used the variables with the biggest decrease 
in mean gini effects and could construct tree with just six (6) most important variables and a forest of just 100 trees 
and still achieved over 98% accuracy.  The savings in training speed and economy of data requirement was significant.  
