# 1/ Import the necessary libraries


import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics


# 2/ Load the iris dataset
iris = load_iris()
data = pd.DataFrame({'sepal length':iris.data[:,0],
                     'sepal width':iris.data[:,1],
                     'petal length':iris.data[:,2],
                      'petal width':iris.data[:,3],
                      'species': iris.target })
X = data[['sepal length','sepal width', 'petal length','petal width']]
Y = data['species']
X_train, X_test, Y_train , Y_test = train_test_split(X,Y, test_size = 0.2, random_state=4)

# 3/ Train a Random Forest Classifier on the iris dataset
clf = RandomForestClassifier(n_estimators = 10)
clf.fit(X_train,Y_train)
Y_pred = clf.predict(X_test)
print('Accuracy:' , metrics.accuracy_score(Y_test,Y_pred))

# 4 /Pour créer l'application Streamlit, nous utilisons les fonctions streamlit.title() et streamlit.header()

st.title("Iris Flower Prediction App")
st.header("Enter the values below to predict the type of iris flower")

# 5/
sepal_length = st.slider("sepal length", float(X.iloc[:, 0].min()),float(X.iloc[:, 0].max()), float(X.iloc[:, 0].mean()))
sepal_width = st.slider("sepal width", float(X.iloc[:, 1].min()),float(X.iloc[:, 1].max()), float(X.iloc[:, 1].mean()))
petal_length = st.slider("petal length", float(X.iloc[:, 2].min()),float(X.iloc[:, 2].max()), float(X.iloc[:, 2].mean()))
petal_width = st.slider("petal width", float(X.iloc[:, 3].min()),float(X.iloc[:, 3].max()), float(X.iloc[:, 3].mean()))

# Affichage
# affichage des valeurs sélectionnées
st.write("Vous avez choisi les valeurs suivantes:")
st.write("- Sepal Length:", sepal_length)
st.write("- Sepal Width:", sepal_width)
st.write("- Petal Length:", petal_length)
st.write("- Petal Width:", petal_width)

# création du bouton de prédiction
if st.button("Prédire"):
    # chargement du modèle de classification

    # création d'un DataFrame avec les valeurs d'entrée
    new_data = pd.DataFrame({'sepal length': sepal_length,
                             'sepal width': sepal_width,
                             'petal length': petal_length,
                             'petal width': petal_width},
                            index=[0])
    # prédiction de la classe de la fleur
    prediction = clf.predict(new_data)
    # affichage de la prédiction
    st.write("La fleur iris est de type", iris.target_names[prediction[0]])

