#🖥️ Codice completo per app.py

import streamlit as st
import joblib

import pandas as pd
from datetime import datetime
import os
import plotly.graph_objects as go

# Carica i modelli
lr_model = joblib.load('model_lr.pkl')
rf_model = joblib.load('model_rf.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Titolo
st.title("Analisi del Sentiment in Tempo Reale")
st.write("Inserisci una frase e scopri se il sentiment è positivo o negativo.")

# Selettore del modello
model_choice = st.radio("Scegli il modello:", ["Logistic Regression", "Random Forest"])

#prediction=""
#confidence=0
# Input utente
user_input = st.text_area("✏️ Scrivi qui la tua recensione:")

# Analisi
#NOTA:ho assegnato dei valori altrimenti va in errore nella generazione del grafico

if st.button("Analizza Sentiment"):
    if user_input.strip() == "":
        st.warning("Inserisci del testo prima di analizzare.")
    else:
        # Analisi del testo
        X_input = vectorizer.transform([user_input])
        model = lr_model if model_choice == "Logistic Regression" else rf_model
        prediction = model.predict(X_input)[0]
        proba = model.predict_proba(X_input)[0]
        class_index = list(model.classes_).index("positivo")
        confidence = proba[class_index]

        # Mostra risultato
        st.success(f"Sentiment previsto: **{prediction.upper()}**")
        st.progress(confidence)
        st.write(f"Probabilità che sia positivo: **{confidence:.2%}**")

        # Salva su CSV — dentro il blocco!
        data = {
            "timestamp": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            "testo": [user_input],
            "sentiment": [prediction],
            "probabilità_positivo": [confidence]
        }

        df_new = pd.DataFrame(data)

        if os.path.exists("storico_sentiment.csv"):
            df_existing = pd.read_csv("storico_sentiment.csv")
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df_combined = df_new

            df_combined.to_csv("storico_sentiment.csv", index=False)

    fig = go.Figure(go.Indicator(
    mode = "gauge+number",
    value = confidence * 100,
    title = {'text': "Confidenza Sentiment Positivo (%)"},
    gauge = {
		 'axis': {'range': [0, 100]},
		 'bar': {'color': "green" if prediction == "positivo" else "red"},
		'steps': [
    {'range': [0, 50], 'color': "lightcoral"},
    {'range': [50, 100], 'color': "lightgreen"}
				 ]
	   }
							 ))
    st.plotly_chart(fig)

