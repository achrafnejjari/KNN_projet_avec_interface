# tu peux run interface avec : streamlit run app.py 
import streamlit as st
import pandas as pd
import joblib

# Charger le modèle et le scaler
model = joblib.load('knn_model.pkl')
scaler = joblib.load('scaler.pkl')

def predire_satisfaction_raw(model, scaler, infos_passager):
    colonnes = ['Gender', 'Age', 'Type of Travel', 'Class', 'Flight Distance', 
                'Inflight service', 'Cleanliness', 'Departure Delay in Minutes', 'Arrival Delay in Minutes']
    
    data = pd.DataFrame([infos_passager], columns=colonnes)

    # Encoder les variables catégorielles
    data['Gender'] = data['Gender'].map({'Male': 1, 'Female': 0})
    data['Type of Travel'] = data['Type of Travel'].map({'Business travel': 1, 'Personal Travel': 0})
    data['Class'] = data['Class'].map({'Business': 2, 'Eco Plus': 1, 'Eco': 0})

    # Normaliser les colonnes numériques
    numerical_columns = ['Age', 'Flight Distance', 'Inflight service', 'Cleanliness', 
                         'Departure Delay in Minutes', 'Arrival Delay in Minutes']
    data[numerical_columns] = scaler.transform(data[numerical_columns])

    prediction = model.predict(data)[0]
    return "✅ Le passager est Satisfait" if prediction == 1 else "❌ Le passager n'est pas satisfait"

# Interface Streamlit
st.title("🔍 Prédiction de la satisfaction des passagers")

gender = st.selectbox("Genre", ['Male', 'Female'])
age = st.slider("Âge", 10, 100, 30)
type_travel = st.selectbox("Type de voyage", ['Business travel', 'Personal Travel'])
travel_class = st.selectbox("Classe", ['Business', 'Eco Plus', 'Eco'])
flight_distance = st.number_input("Distance du vol", min_value=0)
inflight_service = st.slider("Service à bord", 0, 5, 3)
cleanliness = st.slider("Propreté", 0, 5, 3)
dep_delay = st.number_input("Retard au départ (en minutes)", min_value=0)
arr_delay = st.number_input("Retard à l’arrivée (en minutes)", min_value=0)

if st.button("Prédire la satisfaction"):
    infos = [gender, age, type_travel, travel_class, flight_distance, 
             inflight_service, cleanliness, dep_delay, arr_delay]
    
    resultat = predire_satisfaction_raw(model, scaler, infos)
    st.success(resultat)
