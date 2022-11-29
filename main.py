# -------------------------Setting Up-------------------

import pandas as pd
import sklearn as sk
import numpy as np
import tkinter as tk
from tkinter import *
from tkinter import ttk
from tkinter import messagebox

# import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import classification_report,confusion_matrix
from sklearn import metrics

# Importing the CSV file
mydata = pd.read_csv("CustomersDataset.csv")

# Since customerID column contains unique data for each entry and we won't need it in our calculations (not an independent variable). So we dropped it from our DataFrame
mydata.drop(columns=["customerID"], inplace=True, axis=1)


# -----------------------------PreProcessing----------------------------
# Since we need to perform calculations and train our models on the dataset we need to have it in numeric form, so we converted each column to numeric values.
df = pd.read_csv("CustomersDataset.csv")
df.replace(
    {
        'gender': {'Female': 1, 'Male': 0},
        'Partner': {'Yes': 1, 'No': 0},
        'Dependents': {'Yes': 1, 'No': 0},
        'PhoneService': {'Yes': 1, 'No': 0},
        'MultipleLines': {'Yes': 1, 'No': 0, 'No phone service': 2},
        'InternetService': {'DSL': 1, 'Fiber optic': 2, 'No': 0},
        'OnlineSecurity': {'Yes': 1, 'No': 0, 'No internet service': 2},
        'OnlineBackup': {'Yes': 1, 'No': 0, 'No internet service': 2},
        'DeviceProtection': {'Yes': 1, 'No': 0, 'No internet service': 2},
        'TechSupport': {'Yes': 1, 'No': 0, 'No internet service': 2},
        'StreamingTV': {'Yes': 1, 'No': 0, 'No internet service': 2},
        'StreamingMovies': {'Yes': 1, 'No': 0, 'No internet service': 2},
        'Contract': {'Month-to-month': 0, 'One year': 1, 'Two year': 2},
        'PaperlessBilling': {'Yes': 1, 'No': 0},
        'PaymentMethod': {'Bank transfer (automatic)': 0, 'Credit card (automatic)': 1, 'Electronic check': 2,
                          'Mailed check': 3},
        'Churn': {'Yes': 1, 'No': 0}
    }, inplace=True
)
# Remove any Duplicated entries
df.drop_duplicates(inplace=True)
# Remove any row that contain any Empty cells
df.dropna(subset=['customerID', 'gender', 'SeniorCitizen',
                  'Partner', 'Dependents', 'tenure', 'PhoneService', 'MultipleLines',
                  'InternetService', 'OnlineSecurity', 'OnlineBackup',
                  'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                  'Contract', 'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges', 'TotalCharges', 'Churn'],
          inplace=True)

mm = df

# Defining our Dependent and Independent Variables in separate DataFrames

X = df.drop(columns=['customerID', 'Churn'])
Y = df['Churn']

# Checking the Data Type of each column in the X DataFrame (Independent Variable)
# print(X.dtypes)

# Converting each column to any numeric type to be able to train our Machine
# model=DecisionTreeClassifier()
X['gender'] = X['gender'].astype(float)
X['SeniorCitizen'] = X['SeniorCitizen'].astype(float)
X['Partner'] = X['Partner'].astype(float)
X['Dependents'] = X['Dependents'].astype(float)
X['tenure'] = X['tenure'].astype(float)
X['PhoneService'] = X['PhoneService'].astype(float)
X['MultipleLines'] = X['MultipleLines'].astype(float)
X['InternetService'] = X['InternetService'].astype(float)
X['OnlineSecurity'] = X['OnlineSecurity'].astype(float)
X['OnlineBackup'] = X['OnlineBackup'].astype(float)
X['DeviceProtection'] = X['DeviceProtection'].astype(float)
X['TechSupport'] = X['TechSupport'].astype(float)
X['StreamingTV'] = X['StreamingTV'].astype(float)
X['StreamingMovies'] = X['StreamingMovies'].astype(float)
X['Contract'] = X['Contract'].astype(float)
X['PaperlessBilling'] = X['PaperlessBilling'].astype(float)
X['PaymentMethod'] = X['PaymentMethod'].astype(float)
X['MonthlyCharges'] = X['MonthlyCharges'].astype(float)
pd.to_numeric(Y)

# model=SVC()
# model.fit(X,Y)
# Error in the following line
# Error in the column of TotalCharges contains multiple data types so gives out Object type that cannot be converted to a numeric type
# model.fit(X,Y)

# -----------Debuging--------------
# Not finding any empty cells
# mm.dropna(subset=['TotalCharges'],inplace=True)
# Finding empty cells that contain a space ' '. Getting the number of entries that contain ' '
mm["TotalCharges"].isnull().sum()
c = (mm['TotalCharges'] == ' ').sum()
#print(c) = 11
# So our relevant data= 7032 rows x 21 columns

# ---------------------Preprocessing Stage---------------------
# The irrelevent values found were in the TotalCharges Column. The irrelevent values were (' ')(space)
for x in mm.index:
    if mm.loc[x, "TotalCharges"] == ' ':
        mm.drop(x, inplace=True)
        # continue

# Creating a new DataFrame after preprocessing column "TotalCharges"
mm['gender'] = mm['gender'].astype(float)
mm['SeniorCitizen'] = mm['SeniorCitizen'].astype(float)
mm['Partner'] = mm['Partner'].astype(float)
mm['Dependents'] = mm['Dependents'].astype(float)
mm['tenure'] = mm['tenure'].astype(float)
mm['PhoneService'] = mm['PhoneService'].astype(float)
mm['MultipleLines'] = mm['MultipleLines'].astype(float)
mm['InternetService'] = mm['InternetService'].astype(float)
mm['OnlineSecurity'] = mm['OnlineSecurity'].astype(float)
mm['OnlineBackup'] = mm['OnlineBackup'].astype(float)
mm['DeviceProtection'] = mm['DeviceProtection'].astype(float)
mm['TechSupport'] = mm['TechSupport'].astype(float)
mm['StreamingTV'] = mm['StreamingTV'].astype(float)
mm['StreamingMovies'] = mm['StreamingMovies'].astype(float)
mm['Contract'] = mm['Contract'].astype(float)
mm['PaperlessBilling'] = mm['PaperlessBilling'].astype(float)
mm['PaymentMethod'] = mm['PaymentMethod'].astype(float)
mm['MonthlyCharges'] = mm['MonthlyCharges'].astype(float)
mm['TotalCharges'] = mm['TotalCharges'].astype(float)

# Defining Our Dependent and Independent Variables in DataFrames
XX = mm.drop(columns=['customerID', 'Churn'])
YY = mm['Churn']
# Splitting the data to train and test data
X_train, X_test, Y_train, Y_test = train_test_split(XX, YY, test_size=0.2, shuffle=False)
# Applying the Decision Tree Classifier
model = DecisionTreeClassifier()
model.fit(X_train, Y_train)
prediction = model.predict(X_test)
# Getting the Accuracy
score = accuracy_score(Y_test, prediction)
print('Decision Tree: ' + str(score))

# Applying the SVM Classifier
modelsvm = SVC()
modelsvm.fit(X_train, Y_train)
prediction_svm = modelsvm.predict(X_test)
# Getting the Accuracy
score_svm = accuracy_score(Y_test, prediction_svm)
print('SVM : ' + str(score_svm))

# Applying Logistic Regression Classifier
model_lr = LogisticRegression()
model_lr.fit(X_train, Y_train)

# proba_predict=model_lr.predict_proba(X_test)
Y_predict_lr = model_lr.predict(X_test)
# Getting the Accuracy
score_lr = model_lr.score(X_test, Y_test)
print("Logistic Regression : " + str(score_lr))
# cnf_matrix = metrics.confusion_matrix(Y_test, Y_predict)
# metrics.accuracy_score(Y_test, Y_predict)

#Applying Naive Bayes Classifier
gnb = KNeighborsClassifier()
gnb.fit(X_train, Y_train)
Y_pred_nb = gnb.predict(X_test)
score_nb = accuracy_score(Y_test, Y_pred_nb)
print("Naive Bayes : " + str(score_nb))
# -------------------GUI Codes--------------------

# Creating The Root or Frame of the GUI
root = Tk()
root.title("Service Cancellation Predictor")

# Creating all of the labels used in the GUI

label1 = Label(root, text="Mehtolodgy")
label2 = Label(root, text="Customer Data")
label3 = Label(root, text="CustomerID")
label4 = Label(root, text="Partner")
label5 = Label(root, text="Phone Service")
label6 = Label(root, text="Online Security")
label7 = Label(root, text="Tech Support")
label8 = Label(root, text="Contract")
label9 = Label(root, text="Monthly Charges")
label10 = Label(root, text="Gender")
label11 = Label(root, text="Dependent")
label12 = Label(root, text="Multiple lines")
label13 = Label(root, text="Online Backup")
label14 = Label(root, text="Streaming TV")
label15 = Label(root, text="Paperless Billing")
label16 = Label(root, text="Total Charges")
label17 = Label(root, text="Senior Citizen")
label18 = Label(root, text="Tenure")
label19 = Label(root, text="Internet Service")
label20 = Label(root, text="Device Protection")
label21 = Label(root, text="Streaming Movies")
label22 = Label(root, text="Payment Method")
label23 = Label(root, text="Churn")

# Placing all of our Labels in their specific placement
label1.grid(row=0, column=0)

label2.grid(row=3, column=0)
label3.grid(row=4, column=0)
label4.grid(row=5, column=0)
label5.grid(row=6, column=0)
label6.grid(row=7, column=0)
label7.grid(row=8, column=0)
label8.grid(row=9, column=0)
label9.grid(row=10, column=0)

label10.grid(row=4, column=2)
label11.grid(row=5, column=2)
label12.grid(row=6, column=2)
label13.grid(row=7, column=2)
label14.grid(row=8, column=2)
label15.grid(row=9, column=2)
label16.grid(row=10, column=2)

label17.grid(row=4, column=4)
label18.grid(row=5, column=4)
label19.grid(row=6, column=4)
label20.grid(row=7, column=4)
label21.grid(row=8, column=4)
label22.grid(row=9, column=4)
# label23.grid(row=10 , column=4 )


# Creating the Textboxes used in the GUI
txt1 = Entry(root, width=20, borderwidth=5)
txt2 = Entry(root, width=20, borderwidth=5)
txt3 = Entry(root, width=20, borderwidth=5)
txt4 = Entry(root, width=20, borderwidth=5)
txt5 = Entry(root, width=20, borderwidth=5)
txt6 = Entry(root, width=20, borderwidth=5)
txt7 = Entry(root, width=20, borderwidth=5)

txt8 = Entry(root, width=20, borderwidth=5)
txt9 = Entry(root, width=20, borderwidth=5)
txt10 = Entry(root, width=20, borderwidth=5)
txt11 = Entry(root, width=20, borderwidth=5)
txt12 = Entry(root, width=20, borderwidth=5)
txt13 = Entry(root, width=20, borderwidth=5)
txt14 = Entry(root, width=20, borderwidth=5)

txt15 = Entry(root, width=20, borderwidth=5)
txt16 = Entry(root, width=20, borderwidth=5)
txt17 = Entry(root, width=20, borderwidth=5)
txt18 = Entry(root, width=20, borderwidth=5)
txt19 = Entry(root, width=20, borderwidth=5)
txt20 = Entry(root, width=20, borderwidth=5)

# Placing the TextBoxes in their right place
txt1.grid(row=4, column=1)
txt2.grid(row=5, column=1)
txt3.grid(row=6, column=1)
txt4.grid(row=7, column=1)
txt5.grid(row=8, column=1)
txt6.grid(row=9, column=1)
txt7.grid(row=10, column=1)

txt8.grid(row=4, column=3)
txt9.grid(row=5, column=3)
txt10.grid(row=6, column=3)
txt11.grid(row=7, column=3)
txt12.grid(row=8, column=3)
txt13.grid(row=9, column=3)
txt14.grid(row=10, column=3)

txt15.grid(row=4, column=6)
txt16.grid(row=5, column=6)
txt17.grid(row=6, column=6)
txt18.grid(row=7, column=6)
txt19.grid(row=8, column=6)
txt20.grid(row=9, column=6)

# Creating the Checkboxes for retrieving a specific algorithim
var1 = IntVar()
tk.Checkbutton(root, text="Logistic Regression", variable=var1, onvalue=1, offvalue=0).grid(row=0, column=0, padx=15)
var2 = IntVar()
tk.Checkbutton(root, text="SVM", variable=var2, onvalue=1, offvalue=0).grid(row=0, column=1, padx=15)
var3 = IntVar()
tk.Checkbutton(root, text="ID3", variable=var3, onvalue=1, offvalue=0).grid(row=0, column=2, padx=15)
var4 = IntVar()
tk.Checkbutton(root, text="Naive Bayes", variable=var4, onvalue=1, offvalue=0).grid(row=0, column=3, padx=15)

global checkbox
global performTesting
performTesting = [0, 0, 0, 0]
global performTraining
performTraining = [0, 0, 0, 0]


# Creating the buttons command user defined functions
def train():
    checkbox = [var1.get(), var2.get(), var3.get(), var4.get()]

    cnt = 0
    for k in checkbox:
        if (k == 0):
            cnt = cnt + 1

    if (cnt == 4):
        messagebox.showerror("No classifier selected", "Please choose a classifier")
        return

    for h in range(0, 4):
        if (h == 0 and checkbox[h] == 1):
            model_lr.fit(X_train, Y_train)
            performTraining[h] = 1
        elif (h == 1 and checkbox[h] == 1):
            modelsvm.fit(X_train, Y_train)
            performTraining[h] = 1
        elif (h == 2 and checkbox[h] == 1):
            model.fit(X_train, Y_train)
            performTraining[h] = 1
        elif (h == 3 and checkbox[h] == 1):
            gnb.fit(X_train, Y_train)
            performTraining[h] = 1

    index = 0
    for i in performTraining:
        if (index == 0):
            my_label = "Logistic Regression: "
        elif (index == 1):
            my_label = my_label + "SVM: "
        elif (index == 2):
            my_label = my_label + "Decision Tree: "
        elif (index == 3):
            my_label = my_label + "Naive Bayes: "
        if (i == 0):
            my_label = my_label + "Model not trained yet "
        elif (i == 1):
            my_label = my_label + "Model trained successfully "
        index = index + 1
        my_label = my_label + " \n "

    messagebox.showinfo("Training Process", my_label)
    return


def test():
    checkbox = [var1.get(), var2.get(), var3.get(),var4.get()]

    cnt = 0
    for k in checkbox:
        if (k == 0):
            cnt = cnt + 1

    if (cnt == 4):
        messagebox.showerror("No classifier selected", "Please choose a classifier")
        return

    j = 0
    for l in performTraining:
        if (checkbox[j] == 1 and l == 0):
            messagebox.showerror("Model not Trained yet", "Please train your classifier first")
            return
        j = j + 1

    for h in range(0, 4):
        if (h == 0 and checkbox[h] == 1):
            Y_predict = model_lr.predict(X_test)
            #score_lr = model_lr.score(Y_test, Y_predict)
            performTesting[h] = round(score_lr, 3)
        elif (h == 1 and checkbox[h] == 1):
            Y_predict = modelsvm.predict(X_test)
            #score_svm = modelsvm.score(Y_test, Y_predict)
            performTesting[h] = round(score_svm, 3)
        elif (h == 2 and checkbox[h] == 1):
            Y_predict = model.predict(X_test)
            #score = model.score(Y_test, Y_predict)
            performTesting[h] = round(score, 3)
        elif (h == 3 and checkbox[h] == 1):
            Y_predict = gnb.predict(X_test)
            # score = gnb.score(Y_test, Y_predict)
            performTesting[h] = round(score_nb, 3)

    index = 0
    for i in performTesting:
        if (index == 0):
            my_label = "Logistic Regression: "
        elif (index == 1):
            my_label = my_label + "SVM: "
        elif (index == 2):
            my_label = my_label + "Decision Tree: "
        elif (index == 3):
            my_label = my_label + "Naive Bayes: "
        if (i == 0):
            my_label = my_label + "Model no tested yet "
        elif (i > 0):
            my_label = my_label + str(i)
        index = index + 1
        my_label = my_label + " \n "
    messagebox.showinfo("Testing Process", my_label)
    return

def popup():
    alldependentvars = [txt1.get(), txt8.get(), txt15.get(), txt2.get(), txt9.get(), txt16.get(), txt3.get(),
                        txt10.get(), txt17.get(), txt4.get(), txt11.get(), txt18.get(), txt5.get(), txt12.get(),
                        txt19.get(), txt6.get(), txt13.get(), txt20.get(), txt7.get(), txt14.get()]
    flag = 1
    for y in alldependentvars:
        if (y == ''):
            flag = 0
            break

    if (flag == 0):
        messagebox.showerror("Incomplete Data", "Please enter all fileds of data")
        return

    if (alldependentvars[3] == 'Yes'):
        alldependentvars[3] = 1
    elif (alldependentvars[3] == 'No'):
        alldependentvars[3] = 0

    if (alldependentvars[4] == 'Yes'):
        alldependentvars[4] = 1
    elif (alldependentvars[4] == 'No'):
        alldependentvars[4] = 0

    if (alldependentvars[9] == 'No internet service'):
        alldependentvars[9] = 2
    elif (alldependentvars[9] == 'Yes'):
        alldependentvars[9] = 1
    elif (alldependentvars[9] == 'No'):
        alldependentvars[9] = 0

    if (alldependentvars[10] == 'No internet service'):
        alldependentvars[10] = 2
    elif (alldependentvars[10] == 'Yes'):
        alldependentvars[10] = 1
    elif (alldependentvars[10] == 'No'):
        alldependentvars[10] = 0

    if (alldependentvars[15] == 'Month-to-month'):
        alldependentvars[15] = 0
    elif (alldependentvars[15] == 'One year'):
        alldependentvars[15] = 1
    elif (alldependentvars[15] == 'Two year'):
        alldependentvars[15] = 2

    if (alldependentvars[1] == 'Female'):
        alldependentvars[1] = 1
    elif (alldependentvars[1] == 'Male'):
        alldependentvars[1] = 0

    if (alldependentvars[6] == 'Yes'):
        alldependentvars[6] = 1
    elif (alldependentvars[6] == 'No'):
        alldependentvars[6] = 0

    if (alldependentvars[7] == 'No phone service'):
        alldependentvars[7] = 2
    elif (alldependentvars[7] == 'Yes'):
        alldependentvars[7] = 1
    elif (alldependentvars[7] == 'No'):
        alldependentvars[7] = 0

    if (alldependentvars[11] == 'No internet service'):
        alldependentvars[11] = 2
    elif (alldependentvars[11] == 'Yes'):
        alldependentvars[11] = 1
    elif (alldependentvars[11] == 'No'):
        alldependentvars[11] = 0

    if (alldependentvars[12] == 'No internet service'):
        alldependentvars[12] = 2
    elif (alldependentvars[12] == 'Yes'):
        alldependentvars[12] = 1
    elif (alldependentvars[12] == 'No'):
        alldependentvars[12] = 0

    if (alldependentvars[6] == 'Yes'):
        alldependentvars[6] = 1
    elif (alldependentvars[6] == 'No'):
        alldependentvars[6] = 0

    if (alldependentvars[8] == 'Fiber optic'):
        alldependentvars[8] = 2
    elif (alldependentvars[8] == 'DSL'):
        alldependentvars[8] = 1
    elif (alldependentvars[8] == 'No'):
        alldependentvars[8] = 0

    if (alldependentvars[14] == 'No internet service'):
        alldependentvars[14] = 2
    elif (alldependentvars[14] == 'Yes'):
        alldependentvars[14] = 1
    elif (alldependentvars[14] == 'No'):
        alldependentvars[14] = 0

    if (alldependentvars[13] == 'No internet service'):
        alldependentvars[13] = 2
    elif (alldependentvars[13] == 'Yes'):
        alldependentvars[13] = 1
    elif (alldependentvars[13] == 'No'):
        alldependentvars[13] = 0

    if (alldependentvars[16] == 'Yes'):
        alldependentvars[16] = 1
    elif (alldependentvars[16] == 'No'):
        alldependentvars[16] = 0

    if (alldependentvars[17] == 'Mailed check'):
        alldependentvars[17] = 3
    elif (alldependentvars[17] == 'Electronic check'):
        alldependentvars[17] = 2
    elif (alldependentvars[17] == 'Credit card (automatic)'):
        alldependentvars[17] = 1
    elif (alldependentvars[17] == 'Bank transfer (automatic)'):
        alldependentvars[17] = 0


    alldependentvars.pop(0)
    # float(alldependentvars[0])
    float(alldependentvars[1])
    float(alldependentvars[2])
    float(alldependentvars[3])
    float(alldependentvars[4])
    float(alldependentvars[5])
    float(alldependentvars[6])
    float(alldependentvars[7])
    float(alldependentvars[8])
    float(alldependentvars[9])
    float(alldependentvars[10])
    float(alldependentvars[11])
    float(alldependentvars[12])
    float(alldependentvars[13])
    float(alldependentvars[14])
    float(alldependentvars[15])
    float(alldependentvars[16])
    float(alldependentvars[17])
    float(alldependentvars[18])

    checkbox = [var1.get(), var2.get(), var3.get(),var4.get()]

    h = 0
    for l in performTraining:
        if (checkbox[h] != l):
            messagebox.showerror("Model not Trained yet", "Please train your classifier first")
            return
        h = h + 1

    id3prediction = model.predict([alldependentvars])
    svmprediction = modelsvm.predict([alldependentvars])
    lrprediction = model_lr.predict([alldependentvars])
    nbprediction = gnb.predict([alldependentvars])
    predictionvals = [lrprediction, svmprediction, id3prediction, nbprediction]
    if (predictionvals[0] == 0):
        print("Logistic Regression: NO")
    elif (predictionvals[0] == 1):
        print("Logistic Regression: YES")
    if (predictionvals[1] == 0):
        print("SVM: NO")
    elif (predictionvals[1] == 1):
        print("SVM: YES")
    if (predictionvals[2] == 0):
        print("ID3: NO")
    elif (predictionvals[2] == 1):
        print("ID3: YES")
    if (predictionvals[3] == 0):
            print("Naive Bayes: NO")
    elif (predictionvals[3] == 1):
            print("Naive Bayes: YES")

    cnt = 0
    for j in checkbox:
        if (j == 0):
            cnt = cnt + 1
    # predictionvals = np.array(predictionvals)
    j = 0
    for i in checkbox:
        if (i == 0):
            predictionvals[j] = -1
        j = j + 1
    # for k in predictionvals:
    #     if(predictionvals[k]==-1):
    #         predictionvals=np.delete(predictionvals,[k])

    if (cnt == 4):
        messagebox.showerror("No classifier selected", "Please choose a classifier")
        return

    index = 0
    for i in predictionvals:
        if (index == 0):
            my_label = "Logistic Regression: "
        elif (index == 1):
            my_label = my_label + "SVM: "
        elif (index == 2):
            my_label = my_label + "Decision Tree: "
        elif (index == 3):
            my_label = my_label + "Naive Bayes: "
        if (i == 0):
            my_label = my_label + "No "
        elif (i == 1):
            my_label = my_label + "Yes "
        elif (i == -1):
            my_label = my_label + "Classifier not selected "
        index = index + 1
        my_label = my_label + " \n "

    # predictionvals=[str(x) for x in predictionvals]
    # messagelist='\n'.join(predictionvals)
    messagelist = my_label
    messagebox.showinfo("Your Predictions", messagelist)
    return


ttk.Button(root, text="Train", command=train).grid(column=0, row=1)
ttk.Button(root, text="Test", command=test).grid(column=1, row=1)
ttk.Button(root, text="Predict", command=popup).grid(column=2, row=18)

root.mainloop()
