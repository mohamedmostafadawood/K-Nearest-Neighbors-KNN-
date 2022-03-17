# K-Nearest-Neighbors(KNN)

Consider the data given in the file “EPL_Training.xlsx” and “EPL_Testing.xlsx”. The data represents the
outcome of all games in the English Premier League in the season 2017/2018 that ended in the win of one
of the two competing teams. The attributes given in the files for each game are as follows:
Home Team Name
Away Team Name
HS: Home Team Shots
AS: Away Team Shots
HST: Home Team Shots on Target
AST: Away Team Shots on Target
HF: Home Team Fouls Committed
AF: Away Team Fouls Committed
HC: Home Team Corners
AC: Away Team Corners
FTR: Full-time Result
The home team and away team names are given for reference only. They should be removed from the
analysis. The FTR field represents who won the game (H if the home team won and A if the away team
won). It should be used to identify which team won the game. Therefore, this data has 8 dimensions
excluding the teams names and the FTR field.

I am using K-Nearest Neighbors (KNN) classification to predict who wins the games given in
the file “EPL_Testing.xlsx” based on the 8 dimensions given. Use the data given in the file
“EPL_Training.xlsx” as training data. In order to determine the best value of K to be used to classify the test
data, implement the cross-validation approach given in the lecture with 95% training and 5% validation.
I have repeated this partitioning process 1000 times until I find the best K based on the average error.

Also: I have provided a plot of the average error obtained during cross-validation versus the value of K.
