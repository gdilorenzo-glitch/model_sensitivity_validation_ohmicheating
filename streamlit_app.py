"""
Digital twin del riscaldamento ohmico: punto di ingresso dell'app.
Definisce la navigazione tra le pagine (titoli mostrati nel menu laterale).

Avvio locale:  streamlit run streamlit_app.py
"""

import streamlit as st

st.set_page_config(page_title="Digital twin del riscaldamento ohmico", layout="wide")

pagine = st.navigation([
    st.Page("pagine/modello_base.py", title="Modello base per il riscaldamento ohmico",
            url_path="modello-base", default=True),
    st.Page("pagine/multifase.py", title="Riscaldamento ohmico del sistema multifase bicomponente",
            url_path="multifase"),
    st.Page("pagine/sensitivita_mesh.py", title="Analisi di sensitività alla mesh",
            url_path="sensitivita-mesh"),
    st.Page("pagine/calibrazione_validazione.py", title="Calibrazione e validazione",
            url_path="calibrazione-validazione"),
])
pagine.run()
