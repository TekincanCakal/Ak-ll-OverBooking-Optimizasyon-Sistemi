import os
from joblib import dump, load
import pandas as pd
import numpy as np
from sklearn.calibration import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def train():
    dataset = pd.read_csv('dataset_new.csv')
    X_train, X_test, y_train, y_test = train_test_split(
        dataset.iloc[:, :-1], dataset['Result'], test_size=0.3)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(accuracy_score(y_test, y_pred))
    answer = int(input("Save Model 1 / 0\n"))
    if (answer == 1):
        dump(model, 'RandomForestClassifier.joblib')


def generate_dataset():
    dataset = pd.read_csv('dataset.csv')
    dataset = dataset.drop(columns=['id', 'Type of Travel', 'Flight Distance', 'Age', 'Gender', 'Inflight wifi service', 'Ease of Online booking', 'Departure/Arrival time convenient', 'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort',
                                    'Inflight entertainment', 'On-board service', 'Leg room service', 'Baggage handling', 'Checkin service', 'Inflight service', 'Cleanliness', 'Departure Delay in Minutes', 'Arrival Delay in Minutes', 'satisfaction'])
    class_type = dict()
    class_type["Eco"] = 1
    class_type["Eco Plus"] = 1
    class_type["Business"] = 2
    dataset['Class'] = dataset['Class'].replace(class_type)
    dataset["Weather Type"] = np.random.randint(1, 4, size=len(dataset))
    customer_type = dict()
    customer_type["Loyal Customer"] = 1
    customer_type["disloyal Customer"] = 2
    dataset['Customer Type'] = dataset['Customer Type'].replace(customer_type)

    def percentage_generator(x):
        te = np.random.randint(0, 2)
        if te == 0:
            return 0
        else:
            return np.random.random_sample()
    dataset["Canceled Ticket Percentage"] = dataset.apply(
        percentage_generator, axis=1)
    dataset["Miss Ticket Percentage"] = dataset.apply(
        percentage_generator, axis=1)
    dataset["Flight Season"] = np.random.randint(1, 4, size=len(dataset))
    dataset["Traffic Volume Percentage"] = np.random.random_sample(
        size=len(dataset))
    dataset["Is Promotional Ticket"] = np.random.randint(
        1, 3, size=len(dataset))

    def resulter(x):
        risk_percentage = 0
        if (float(x["Customer Type"]) == 2):
            risk_percentage -= 20
        else:
            risk_percentage += 10

        if (float(x["Class"]) == 2): # bussiness
            risk_percentage -= 10

        if (float(x["Weather Type"]) == 3): # karlı
            risk_percentage += 20
        elif (float(x["Weather Type"]) == 2):  # yağmurlu
            risk_percentage += 10

        if (float(x["Miss Ticket Percentage"]) >= 0.7):
            risk_percentage -= 30
        elif (float(x["Miss Ticket Percentage"]) >= 0.3):
            risk_percentage -= 10
        elif (float(x["Miss Ticket Percentage"]) >= 0.1):
            risk_percentage -= 5
        else:
            risk_percentage += 20

        if (float(x["Canceled Ticket Percentage"]) >= 0.85):
            risk_percentage -= 20
        elif (float(x["Canceled Ticket Percentage"]) >= 0.65):
            risk_percentage -= 10
        elif (float(x["Canceled Ticket Percentage"]) >= 0.4):
            risk_percentage -= 5
        else:
            risk_percentage += 10

        if (x["Flight Season"] == 3):  # kış
            risk_percentage += 20
        elif (x["Flight Season"] == 2):  # bahar
            risk_percentage += 10
        elif(x["Flight Season"] == 1): #yaz
            risk_percentage -= 10

        if (float(x["Traffic Volume Percentage"]) >= 0.7):
            risk_percentage += 20
        elif (float(x["Traffic Volume Percentage"]) >= 0.5):
            risk_percentage += 10
        elif (float(x["Traffic Volume Percentage"]) >= 0.3):
            risk_percentage += 5
        else:
            risk_percentage -= 10
        
        if (float(x["Is Promotional Ticket"]) == 2):
            risk_percentage += 30
        
        if (risk_percentage <= 20):
            return 0  # risksiz
        if (risk_percentage <= 40):
            return 1  # az riskli
        elif (risk_percentage <= 60):
            return 2  # orta riskli
        else:
            return 3  # çok riskli
        
    dataset["Result"] = dataset.apply(resulter, axis=1)
    dataset = dataset.sample(frac=1).reset_index(drop=True)
    dataset.to_csv("dataset_new.csv")


def test_model():
    if (os.path.isfile("./RandomForestClassifier.joblib")):
        model = load('RandomForestClassifier.joblib')
        customer_type = int(
            input("Enter Customer Type, FFP = 2 / Normal = 1\n"))
        class_type = int(input("Class Type, Bussiness = 2 / Eco = 1\n"))
        weather_type = int(
            input("Weather Type, Snow = 3 / Rain = 2 / Normal = 1\n"))
        canceled_ticket_percentage = float(
            input("Canceled Ticket Percentage\n"))
        mis_ticket_percentage = float(input("Miss Ticket Percentage\n"))
        flight_season = int(
            input("Flight Season, Winter = 3 / Summer = 2 / Spring = 1\n"))
        traffic_volume_percentage = float(input("Traffic Volume Percentage\n"))
        promotional_ticket = int(
            input("Enter Ticket Type, Promotional = 2 / Normal = 1\n"))
        inp = {"Unnamed: 0.1": 0, "Unnamed: 0": 0, "Customer Type": customer_type, "Class": class_type, "Weather Type": weather_type, "Canceled Ticket Percentage": canceled_ticket_percentage,
               "Miss Ticket Percentage": mis_ticket_percentage, "Flight Season": flight_season, "Traffic Volume Percentage": traffic_volume_percentage, "Is Promotional Ticket": promotional_ticket}
        y_pred = model.predict(pd.DataFrame([inp]))
        print(y_pred)
        if (y_pred[0] == 0):
            print("risksiz")
        elif (y_pred[0] == 1):
            print("az riskli")
        elif (y_pred[0] == 2):
            print("orta riskli")
        else:
            print("çok riskli")
    else:
        print("Model not found")


print("1- Generate Dataset")
print("2- Train a Model")
print("3- Test the Model")
answer = int(input("Select the one of them\n"))
if (answer == 1):
    generate_dataset()
if (answer == 2):
    train()
if (answer == 3):
    test_model()
