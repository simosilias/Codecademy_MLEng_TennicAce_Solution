import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# load and investigate the data here:
df = pd.read_csv('tennis_stats.csv')
print(df.head())
print(df.columns)
#-----------------------------------------------------------------------------------------------------------------
#Offensive: Aces, DoubleFaults, FirstServe, FirstServePointsWon, SecondServePointsWon, BreakPointsFaced,
#, BreakPointsSaved, ServiceGamesWon, TotalServicePointsWon
#Defensive: FirstServeReturnPointsWon, SecondServeReturnPointsWon, BreakPointsOpportunities, BreakPointsConverted,
#, ReturnGamesPlayed, ReturnGamesWon, ReturnPointsWon, TotalPointsWon
#Outcomes: Wins, Losses, Ranking, Winnings
#-----------------------------------------------------------------------------------------------------------------
# perform exploratory analysis here:
plt.figure()
plt.scatter(df['FirstServePointsWon'], df['Wins'], alpha = 0.4) #No
plt.xlabel("FirstServePointsWon")
plt.ylabel("Wins")
plt.figure()
plt.scatter(df['BreakPointsOpportunities'], df['Winnings'], alpha = 0.4)
plt.xlabel("BreakPointsOpportunities")
plt.ylabel("Winnings")
plt.figure()
plt.scatter(df['DoubleFaults'], df['Losses'], alpha = 0.4)
plt.xlabel("DoubleFaults")
plt.ylabel("Losses")
plt.figure()
plt.scatter(df['ReturnGamesWon'], df['Wins'], alpha = 0.4) #No
plt.xlabel("ReturnGamesWon")
plt.ylabel("Wins")
plt.figure()
plt.scatter(df['Aces'], df['Wins'], alpha = 0.4)
plt.xlabel("Aces")
plt.ylabel("Wins")
plt.figure()
plt.scatter(df['Aces'], df['Winnings'], alpha = 0.4)
plt.xlabel("Aces")
plt.ylabel("Winnings")
plt.figure()
plt.scatter(df['FirstServeReturnPointsWon'], df['Winnings'], alpha = 0.4) #No
plt.xlabel("FirstServeReturnPointsWon")
plt.ylabel("Winnings")
plt.figure()
plt.scatter(df['FirstServePointsWon'], df['Winnings'], alpha = 0.4) #No
plt.xlabel("FirstServePointsWon")
plt.ylabel("Winnings")
plt.figure()
plt.scatter(df['TotalPointsWon'], df['Winnings'], alpha = 0.4)
plt.xlabel("TotalPointsWon")
plt.ylabel("Winnings")
plt.figure()
plt.scatter(df['ServiceGamesWon'], df['Winnings'], alpha = 0.4)
plt.xlabel("ServiceGamesWon")
plt.ylabel("Winnings")

plt.show()
#-----------------------------------------------------------------------------------------------------------------
#Relationships:
#BreakPointsOpportunities-Winnings
#DoubleFautls-Losses
#Aces-Wins
#Aces-Winnings
#------------------------------------------------------------------------------------------------------------------

## perform single feature linear regressions here:

print("------ One feature Linear Regression Models ------")

#1st Model (FirstServeReturnPointsWon-Winnings)
features = df[['FirstServeReturnPointsWon']]
outcomes = df[['Winnings']]
#Split the dataset
features_train, features_test, outcomes_train, outcomes_test = train_test_split(features,outcomes, train_size = 0.8)
#Create and train the model
model = LinearRegression()
model.fit(features_train, outcomes_train)
#Model Performance
print("1st Model (FirstServeReturnPointsWon-Winnings) Score: " + str(model.score(features_test, outcomes_test)))
model_predict = model.predict(features_test)
plt.figure()
plt.scatter(outcomes_test, model_predict, alpha = 0.4)
plt.xlabel("Outcomes")
plt.ylabel("Predictions")
plt.title("1st Model (FirstServeReturnPointsWon-Winnings)")

#2nd Model (BreakPointsOpportunities-Winnings)
features = df[['BreakPointsOpportunities']]
outcomes = df[['Winnings']]
#Split the dataset
features_train, features_test, outcomes_train, outcomes_test = train_test_split(features,outcomes, train_size = 0.8)
#Create and train the model
model = LinearRegression()
model.fit(features_train, outcomes_train)
#Model Performance
print("2nd Model (BreakPointsOpportunities-Winnings) Score: " + str(model.score(features_test, outcomes_test)))
model_predict = model.predict(features_test)
plt.figure()
plt.scatter(outcomes_test, model_predict, alpha = 0.4)
plt.xlabel("Outcomes")
plt.ylabel("Predictions")
plt.title("2nd Model (BreakPointsOpportunities-Winnings)")

#3rd Model (DoubleFaults-Losses)
features = df[['DoubleFaults']]
outcomes = df[['Losses']]
#Split the dataset
features_train, features_test, outcomes_train, outcomes_test = train_test_split(features,outcomes, train_size = 0.8)
#Create and train the model
model = LinearRegression()
model.fit(features_train, outcomes_train)
#Model Performance
print("3rd Model (DoubleFaults-Losses) Score: " + str(model.score(features_test, outcomes_test)))
model_predict = model.predict(features_test)
plt.figure()
plt.scatter(outcomes_test, model_predict, alpha = 0.4)
plt.xlabel("Outcomes")
plt.ylabel("Predictions")
plt.title("3rd Model (DoubleFaults-Losses)")

#4th Model (Aces-Wins)
features = df[['Aces']]
outcomes = df[['Wins']]
#Split the dataset
features_train, features_test, outcomes_train, outcomes_test = train_test_split(features,outcomes, train_size = 0.8)
#Create and train the model
model = LinearRegression()
model.fit(features_train, outcomes_train)
#Model Performance
print("4th Model (Aces-Wins) Score: " + str(model.score(features_test, outcomes_test)))
model_predict = model.predict(features_test)
plt.figure()
plt.scatter(outcomes_test, model_predict, alpha = 0.4)
plt.xlabel("Outcomes")
plt.ylabel("Predictions")
plt.title("4th Model (Aces-Wins)")

plt.show()

## perform two feature linear regressions here:

print("------ Two feature Linear Regression Models ------")

#1st Model (FirstServeReturnPointsWon,BreakPointOpportunities-Winnings)
features = df[['FirstServeReturnPointsWon', 'BreakPointsOpportunities']]
outcomes = df[['Winnings']]
#Split the dataset
features_train, features_test, outcomes_train, outcomes_test = train_test_split(features,outcomes, train_size = 0.8)
#Create and train the model
model = LinearRegression()
model.fit(features_train, outcomes_train)
#Model Performance
print("1st Model (FirstServeReturnPointsWon,BreakPointOpportunities-Winnings) Score: " + str(model.score(features_test, outcomes_test)))
model_predict = model.predict(features_test)
plt.figure()
plt.scatter(outcomes_test, model_predict, alpha = 0.4)
plt.xlabel("Outcomes")
plt.ylabel("Predictions")
plt.title("1st Model (FirstServeReturnPointsWon,BreakPointOpportunities-Winnings)")

#2nd Model (FirstServePointsWon, SecondServePointsWon - Winnings)
features = df[['FirstServePointsWon', 'SecondServePointsWon']]
outcomes = df[['Winnings']]
#Split the dataset
features_train, features_test, outcomes_train, outcomes_test = train_test_split(features,outcomes, train_size = 0.8)
#Create and train the model
model = LinearRegression()
model.fit(features_train, outcomes_train)
#Model Performance
print("2nd Model (FirstServePointsWon, SecondServePointsWon - Winnings) Score: " + str(model.score(features_test, outcomes_test)))
model_predict = model.predict(features_test)
plt.figure()
plt.scatter(outcomes_test, model_predict, alpha = 0.4)
plt.xlabel("Outcomes")
plt.ylabel("Predictions")
plt.title("2nd Model (FirstServePointsWon, SecondServePointsWon - Winnings)")

#3rd Model (BreakPointsOpportunities, BreakPointsConverted - Winnings)
features = df[['BreakPointsOpportunities', 'BreakPointsConverted']]
outcomes = df[['Winnings']]
#Split the dataset
features_train, features_test, outcomes_train, outcomes_test = train_test_split(features,outcomes, train_size = 0.8)
#Create and train the model
model = LinearRegression()
model.fit(features_train, outcomes_train)
#Model Performance
print("3rd Model (BreakPointsOpportunities, BreakPointsConverted - Winnings) Score: " + str(model.score(features_test, outcomes_test)))
model_predict = model.predict(features_test)
plt.figure()
plt.scatter(outcomes_test, model_predict, alpha = 0.4)
plt.xlabel("Outcomes")
plt.ylabel("Predictions")
plt.title("3rd Model (BreakPointsOpportunities, BreakPointsConverted - Winnings)")

plt.show()

## perform multiple feature linear regressions here:

print("------ Multiple feature Linear Regression Models ------")

#1st Model (All Columns - Winnings)
features = df[['FirstServe','FirstServePointsWon','FirstServeReturnPointsWon',
'SecondServePointsWon','SecondServeReturnPointsWon','Aces',
'BreakPointsConverted','BreakPointsFaced','BreakPointsOpportunities',
'BreakPointsSaved','DoubleFaults','ReturnGamesPlayed','ReturnGamesWon',
'ReturnPointsWon','ServiceGamesPlayed','ServiceGamesWon','TotalPointsWon',
'TotalServicePointsWon']]
outcome = df[['Winnings']]
#Split the dataset
features_train, features_test, outcomes_train, outcomes_test = train_test_split(features,outcomes, train_size = 0.8)
#Create and train the model
model = LinearRegression()
model.fit(features_train, outcomes_train)
#Model Performance
print("1st Model (All Columns - Winnings) Score: " + str(model.score(features_test, outcomes_test)))
model_predict = model.predict(features_test)
plt.figure()
plt.scatter(outcomes_test, model_predict, alpha = 0.4)
plt.xlabel("Outcomes")
plt.ylabel("Predictions")
plt.title("1st Model (All Columns - Winnings)")

#2nd Model (Service Game Columns (Offensive) - Winnings)
features = df[['Aces', 'DoubleFaults', 'FirstServe', 'FirstServePointsWon', 'SecondServePointsWon', 'BreakPointsFaced',
               'BreakPointsSaved', 'ServiceGamesPlayed', 'ServiceGamesWon', 'TotalServicePointsWon']]
outcomes = df[['Winnings']]
#Split the dataset
features_train, features_test, outcomes_train, outcomes_test = train_test_split(features,outcomes, train_size = 0.8)
#Create and train the model
model = LinearRegression()
model.fit(features_train, outcomes_train)
#Model Performance
print("2nd Model (Service Game Columns (Offensive) - Winnings) Score: " + str(model.score(features_test, outcomes_test)))
model_predict = model.predict(features_test)
plt.figure()
plt.scatter(outcomes_test, model_predict, alpha = 0.4)
plt.xlabel("Outcomes")
plt.ylabel("Predictions")
plt.title("2nd Model (Service Game Columns (Offensive) - Winnings)")

#3rd Model (Return Game Columns (Defensive) - Winnings)
features = df[['FirstServeReturnPointsWon', 'SecondServeReturnPointsWon', 'BreakPointsOpportunities',
               'BreakPointsConverted', 'ReturnGamesPlayed', 'ReturnGamesWon', 'ReturnPointsWon', 'TotalPointsWon']]
outcomes = df[['Winnings']]
#Split the dataset
features_train, features_test, outcomes_train, outcomes_test = train_test_split(features,outcomes, train_size = 0.8)
#Create and train the model
model = LinearRegression()
model.fit(features_train, outcomes_train)
#Model Performance
print("3rd Model (Return Game Columns (Defensive) - Winnings) Score: " + str(model.score(features_test, outcomes_test)))
model_predict = model.predict(features_test)
plt.figure()
plt.scatter(outcomes_test, model_predict, alpha = 0.4)
plt.xlabel("Outcomes")
plt.ylabel("Predictions")
plt.title("3rd Model (Return Game Columns (Defensive) - Winnings)")





