"""
Pagina: Calibrazione e validazione del modello fisico con dati sperimentali (file Excel caricato dall'utente).
"""

import copy
import io
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import streamlit as st

import multifase_engine as me
import confronto_esperimenti as ce

st.title("Calibrazione e validazione")
st.caption("Carica il file Excel delle prove: l'app traduce gli attributi di ciascuna prova (configurazione, tensione, "
           "concentrazione salina, temperature iniziali) nei parametri del modello, simula e confronta con le sonde.")

LARG = 560
COL_SONDE = ['tab:blue', 'tab:orange', 'tab:green']
S_NODO_PASSO = 5e-5

if 'ce' not in st.session_state:
    st.session_state.ce = {'calib': {}}
S = st.session_state.ce

TIPO = {1: 'RPF', 2: 'RPF', 3: 'RPFNB', 4: 'RPFNB', 5: 'RPFNB', 6: 'RPFNB'}
ETICHETTE_COMP = {i: f"Comp. {i} - {TIPO[i]}: burro {me.COMPOSIZIONI[i]['burro']:g}%, sale {me.COMPOSIZIONI[i]['sale']:g}%, "
                     f"acqua {me.COMPOSIZIONI[i]['acqua']:g}%" for i in range(1, 7)}
CUSTOM = 'Personalizzata (burro e acqua a scelta; predefinita: ricetta sperimentale con burro)'


# =============================================================================
# Barra laterale
# =============================================================================
@st.cache_data(show_spinner=False)
def carica(contenuto: bytes):
    return ce.leggi_dati(io.BytesIO(contenuto))


with st.sidebar:
    st.header("Dati sperimentali")
    file = st.file_uploader("File Excel con le prove (.xlsx)", type=['xlsx'], key="ce_file")

df, avvisi, el, configs, tensioni, sali = None, [], None, [], [], []
if file is not None:
    try:
        df, avvisi = carica(file.getvalue())
        el = ce.elenco_prove(df)
        configs = sorted(df['configurazione'].unique())
        tensioni = sorted(df['tensione'].unique())
        sali = sorted(df['sale'].unique())
    except Exception as e:
        st.sidebar.error(f"Impossibile leggere il file: {e}")
        df = None

with st.sidebar:
    st.header("Impostazioni del modello")
    with st.expander("Cella e tensione", expanded=True):
        cella = st.selectbox("Cella", list(me.CELLE.keys()), key="ce_cella")
        Lx_mm = st.number_input("Distanza tra gli elettrodi [mm]", value=me.CELLE[cella]['Lx'] * 1000, min_value=10.0,
                                key=f"ce_Lx_{cella}")
        modo_V = st.selectbox("La tensione del file è", ["tensione efficace (rms)", "tensione di picco"], key="ce_modoV")
    with st.expander("Ricetta della purea", expanded=True):
        sel_ric = st.selectbox("Ricetta (Tabella 4.3)", [ETICHETTE_COMP[i] for i in range(1, 7)] + [CUSTOM],
                               index=6, key="ce_ricetta")
        if sel_ric == CUSTOM:
            ricetta_id = None
            burro = st.number_input("Burro [% w/w]", value=5.8, min_value=0.0, key="ce_burro")
            acqua = st.number_input("Acqua [% w/w]", value=72.5, min_value=1.0, key="ce_acqua")
            sale_da_file = True
            st.caption("Fiocchi = 100 - burro - acqua - sale; il sale viene dal file.")
        else:
            ricetta_id = [i for i in range(1, 7) if ETICHETTE_COMP[i] == sel_ric][0]
            burro, acqua = me.COMPOSIZIONI[ricetta_id]['burro'], me.COMPOSIZIONI[ricetta_id]['acqua']
            uso_sale = st.radio("Concentrazione salina", ["Dal file (burro e acqua della ricetta)", "Esatta della ricetta (ignora il file)"],
                                key="ce_usosale",
                                help="Con la prima opzione ogni prova usa il proprio sale e i fiocchi si adattano (fiocchi = resto).")
            sale_da_file = uso_sale.startswith("Dal file")
        opz_reg = ["Automatica (più vicina per rapporto sale/acqua)"] + [
            f"Comp. {i} ({TIPO[i]}, sale {me.COMPOSIZIONI[i]['sale']:g}%)" for i in range(1, 7)]
        sel_reg = st.selectbox("Regressione σ(T) di partenza per la purea", opz_reg, key="ce_reg")
        regressione_partenza = None if sel_reg.startswith("Automatica") else int(sel_reg.split()[1])
    occ_edit = None
    if df is not None:
        with st.expander("Configurazioni → posizioni delle polpette", expanded=True):
            st.caption("Spunta le posizioni occupate da una polpetta. 'Middle' e 'Side' sono dedotte dalle temperature "
                       "iniziali (la sonda nella polpetta parte più fredda): **da confermare**. Una riga senza nessuna spunta è una configurazione "
                       "**solo purea (RP)**: le tre sonde sono tutte in purea.")
            occ_df = pd.DataFrame({'P1': [ce.occupazione_default(c)[0] for c in configs],
                                   'P2': [ce.occupazione_default(c)[1] for c in configs],
                                   'P3': [ce.occupazione_default(c)[2] for c in configs]}, index=configs)
            occ_edit = st.data_editor(occ_df, key="ce_occ", width='stretch')
    with st.expander("Scambio e proprietà termofisiche"):
        T_amb = st.number_input("Temperatura ambiente [°C]", value=21.0, key="ce_Tamb")
        h_lbl = {'h(T): convezione naturale + irraggiamento': 'naturale_irraggiamento', 'h costante 10 W/m²/K': 'costante'}
        h_sel = st.selectbox("Scambio sulla superficie libera", list(h_lbl.keys()), key="ce_h")
        prop_lbl = {'Choi–Okos dipendenti da T': 'choi_okos_T', 'Choi–Okos costanti a 50 °C': 'choi_okos_Tref'}
        prop_sel = st.selectbox("Proprietà termofisiche", list(prop_lbl.keys()), key="ce_prop")
    with st.expander("Mesh e passo temporale", expanded=True):
        mm = st.session_state.get('mesh_minima')
        h_def = float(mm['h_mm']) if mm else 5.0
        if mm:
            st.success(f"Dall'analisi di sensitività: {mm['h_mm']:g} mm (tolleranza {mm['tol']:g} °C).")
        else:
            st.info("Analisi di sensitività alla mesh non ancora eseguita: predefinito 5 mm. "
                    "Eseguila nella pagina dedicata per scegliere la mesh minima.")
        h_mesh = st.slider("Passo di mesh [mm]", 1.5, 10.0, h_def, 0.1, key=f"ce_hmesh_{h_def}",
                           help="Usato per simulazione, calibrazione e validazione.")
        dt_run = st.selectbox("Passo temporale: simulazione singola [s]", [5.0, 10.0], key="ce_dtrun")
        dt_cal = st.selectbox("Passo temporale: calibrazione e validazione [s]", [10.0, 5.0, 15.0], key="ce_dtcal")

imp = ce.Impostazioni(cella=cella, Lx_mm=Lx_mm, tensione_rms=modo_V.startswith("tensione efficace"),
                      ricetta_id=ricetta_id, burro=burro, acqua=acqua, sale_da_file=sale_da_file,
                      regressione_partenza=regressione_partenza, h_modo=h_lbl[h_sel], T_amb=T_amb,
                      prop_modo=prop_lbl[prop_sel])
if df is not None:
    imp.occupazione = {c: tuple(bool(occ_edit.loc[c, p]) for p in ('P1', 'P2', 'P3')) for c in configs}


# =============================================================================
# Utilità e figure
# =============================================================================
def id_prova(c, V, s):
    return f"{c} | {V:g} V | sale {s:g}"


def chiave_sale(s):
    return round(float(s), 4)


def nodi_stimati(d):
    geo = ce._geometria_prova(imp, d['configurazione'].iloc[0])[0]
    return me.stima_nodi(geo, h_mesh / 1000)[0]


def fig_curve_dati(lista):
    fig, ax = plt.subplots(figsize=(7, 4.3))
    for (d, etich) in lista:
        for i in range(3):
            ax.plot(d['tempo'], d[f'T{i + 1}'], color=COL_SONDE[i], alpha=0.8, lw=1.5,
                    label=f'{etich} - P{i + 1}' if len(lista) <= 3 else None)
    ax.set_xlabel('tempo [s]'); ax.set_ylabel('T [°C]'); ax.grid(alpha=.3)
    if len(lista) <= 3:
        ax.legend(fontsize=8)
    ax.set_title('Temperature misurate')
    plt.tight_layout()
    return fig


def fig_sovrapposizione(ris, d, titolo):
    fig, ax = plt.subplots(figsize=(7, 4.6))
    for i, n in enumerate(('P1', 'P2', 'P3')):
        fase = 'polpetta' if ris['in_polpetta'][i] else 'purea'
        ax.plot(d['tempo'], d[f'T{i + 1}'], 'o', ms=3.5, color=COL_SONDE[i], alpha=0.7, label=f'{n} ({fase}) misurata')
        ax.plot(ris['tempo'], ris['T_sonde'][n], '-', color=COL_SONDE[i], lw=2, label=f'{n} simulata')
    ax.set_xlabel('tempo [s]'); ax.set_ylabel('T [°C]'); ax.grid(alpha=.3); ax.legend(fontsize=7, ncol=2)
    ax.set_title(titolo, fontsize=10)
    plt.tight_layout()
    return fig


def fig_errori(ris, d):
    fig, ax = plt.subplots(figsize=(6, 4.6))
    t = d['tempo'].values.astype(float)
    for i, n in enumerate(('P1', 'P2', 'P3')):
        e = np.interp(t, ris['tempo'], ris['T_sonde'][n]) - d[f'T{i + 1}'].values
        ax.plot(t, e, color=COL_SONDE[i], lw=2, label=n)
    ax.axhline(0, color='k', lw=0.8)
    ax.set_xlabel('tempo [s]'); ax.set_ylabel('simulata - misurata [°C]'); ax.grid(alpha=.3); ax.legend(fontsize=8)
    ax.set_title('Errore del modello nel tempo')
    plt.tight_layout()
    return fig


def fig_parity(risultati):
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    colori = {'purea': 'tab:orange', 'polpetta': 'tab:brown'}
    mostrati = set()
    for r in risultati:
        for i in range(3):
            f = r['fasi'][i]
            ax.plot(r['exp'][i], r['sim'][i], '.', ms=3, alpha=0.45, color=colori[f], label=f if f not in mostrati else None)
            mostrati.add(f)
    lim = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lim, lim, 'k--', lw=1)
    ax.set_xlabel('T misurata [°C]'); ax.set_ylabel('T simulata [°C]'); ax.grid(alpha=.3); ax.legend(fontsize=8)
    ax.set_title('Simulata vs misurata (tutti i punti)')
    plt.tight_layout()
    return fig


def fig_rmse_config(tab):
    g = tab.groupby('configurazione')['RMSE_tot'].mean().sort_values()
    fig, ax = plt.subplots(figsize=(5.6, 4))
    ax.bar(g.index, g.values, color='tab:purple', alpha=0.8)
    ax.set_ylabel('RMSE medio [°C]'); ax.grid(alpha=.3, axis='y'); ax.set_title('Errore per configurazione')
    plt.tight_layout()
    return fig


def tabella_prop(prop, prop_lett, p=None):
    """Parametri calibrabili: valore di letteratura, valore usato e, per i fattori su k e rho*cp, il valore effettivo a 20 °C
    (relazione di Choi-Okos moltiplicata per il fattore)."""
    v, v0 = ce.valori_da_prop(prop), ce.valori_da_prop(prop_lett)
    righe = []
    for q in ce.PARAMETRI:
        eff = ''
        if p is not None and p.get('prop_modo') != 'manuali' and q['id'] in ('fk_m', 'fk_p', 'fc_m', 'fc_p'):
            comp = p['comp_matrice'] if q['id'].endswith('_m') else p['comp_polpetta']
            k_, r_, c_ = me.proprieta_choi_okos(comp, 20.0)
            eff = (f"k = {v[q['id']] * float(k_):.3f} W/(m K)" if q['id'].startswith('fk') else
                   f"ρ·cp = {v[q['id']] * float(r_) * float(c_) / 1e6:.3f} MJ/(m³ K)")
        righe.append({'Parametro': q['nome'], 'Unità': q['unita'], 'Letteratura': v0[q['id']], 'Usato': v[q['id']],
                      'Variazione': (v[q['id']] / v0[q['id']] - 1) if v0[q['id']] else np.nan, 'Valore effettivo a 20 °C': eff})
    return pd.DataFrame(righe)


def scelta_proprieta(sale, chiave_widget, calib_disponibile):
    """Radio per scegliere le proprieta' del modello. Restituisce (prop, etichetta)."""
    lett = ce.prop_letteratura(imp, sale)
    modi = (["Calibrate sui dati (per questa concentrazione)"] if calib_disponibile else []) + ["Letteratura (punto di partenza)", "Manuale (solo σ purea)"]
    modo = st.radio("Proprietà del modello", modi, horizontal=True,
                    key=f"{chiave_widget}_{'cal' if calib_disponibile else 'rif'}")
    if modo.startswith("Calibrate"):
        return S['calib'][chiave_sale(sale)]['prop'], "calibrate sui dati", lett
    if modo.startswith("Manuale"):
        m1, m2 = st.columns(2)
        s20 = m1.number_input("σ purea a 20 °C [S/m]", value=1.5, min_value=0.01, key=f"{chiave_widget}_s20")
        al = m2.number_input("α purea [1/K]", value=0.03, min_value=0.0, max_value=0.1, format="%.4f", key=f"{chiave_widget}_al")
        return ce.applica_valori(lett, dict(sig_m_s20=s20, sig_m_alpha=al)), f"manuali (σ purea 20 °C = {s20:.2f} S/m)", lett
    return lett, lett['etichetta'], lett


tab_lett, tab_dati, tab_sim, tab_cal, tab_val = st.tabs(
    ["Proprietà di letteratura", "1. Dati", "2. Simula una prova", "3. Calibrazione", "4. Validazione"])

# =============================================================================
# 0. Proprietà di letteratura
# =============================================================================
with tab_lett:
    st.markdown("Tutte le proprietà di letteratura (e i dati di base) usate dal modello. Sono il **punto di partenza** della "
                "calibrazione: nella scheda «3. Calibrazione» si sceglie quali di esse stimare dai dati.")
    for titolo, contenuto, nota in me.tabelle_letteratura():
        st.subheader(titolo)
        if isinstance(contenuto, tuple) and contenuto[0] == 'latex':
            for r_ in contenuto[1]:
                st.latex(r_)
        elif isinstance(contenuto, tuple) and contenuto[0] == 'md':
            st.markdown(contenuto[1])
        else:
            st.dataframe(contenuto, hide_index=True)
        if nota:
            st.caption(nota)

if df is None:
    for t in (tab_dati, tab_sim, tab_cal, tab_val):
        with t:
            st.info("Carica nella barra laterale un file con le colonne: **Configurazione, Tensione, Concentrazione salina, "
                    "Ttempo, Temperatura 1, Temperatura 2, Temperatura 3** (una riga per istante, più prove nello stesso foglio).")
    st.stop()

# =============================================================================
# 1. Dati
# =============================================================================
with tab_dati:
    for a in avvisi:
        st.warning(a)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Prove", len(el)); c2.metric("Configurazioni", len(configs))
    c3.metric("Tensioni [V]", ", ".join(f"{v:g}" for v in tensioni))
    c4.metric("Concentrazioni saline", ", ".join(f"{s:g}" for s in sali))
    st.dataframe(el[['id', 'punti', 't_max', 'dt', 'T_ini', 'T_fin_media']].rename(columns={
        'id': 'Prova', 'punti': 'Punti', 't_max': 'Durata [s]', 'dt': 'Passo [s]', 'T_ini': 'T iniziale media [°C]',
        'T_fin_media': 'T finale media [°C]'}).round(2), hide_index=True, height=300)
    sel = st.multiselect("Mostra le curve delle prove", list(el['id']), default=list(el['id'])[:1], key="ce_selcurve")
    if sel:
        lista = []
        for idp in sel:
            r = el[el['id'] == idp].iloc[0]
            lista.append((ce.estrai_prova(df, r['configurazione'], r['tensione'], r['sale']), idp))
        st.pyplot(fig_curve_dati(lista), width=LARG + 100)

# =============================================================================
# 2. Simula una prova
# =============================================================================
with tab_sim:
    s1, s2, s3 = st.columns(3)
    conf_s = s1.selectbox("Configurazione", configs, key="ce_conf_s")
    V_opz = sorted(el[el['configurazione'] == conf_s]['tensione'].unique())
    V_s = s2.selectbox("Tensione [V]", V_opz, key=f"ce_V_s_{conf_s}")
    sali_opz = sorted(el[(el['configurazione'] == conf_s) & (el['tensione'] == V_s)]['sale'].unique())
    sale_s = s3.selectbox("Concentrazione salina", sali_opz, key=f"ce_sale_s_{conf_s}_{V_s}")
    d_s = ce.estrai_prova(df, conf_s, V_s, sale_s)
    solo_purea = st.checkbox("Simula il modello a **sola purea** (configurazione RP, nessuna polpetta) e confrontalo con questa prova",
                             key="ce_solopurea", help="Utile per vedere quanto le polpette modificano il riscaldamento: le tre sonde sono considerate in purea.")
    imp_s = copy.copy(imp)
    imp_s.occupazione = dict(imp.occupazione)
    if solo_purea:
        imp_s.occupazione[conf_s] = (False, False, False)

    prop_s, lbl_prop, prop_lett_s = scelta_proprieta(sale_s, "ce_modoprop", chiave_sale(sale_s) in S['calib'])
    geo_s, p_s, in_pol = ce.prepara_prova(d_s, imp_s, prop_s)
    with st.expander("Come il modello è stato impostato a partire dagli attributi della prova", expanded=True):
        info = pd.DataFrame({'Sonda': ['P1', 'P2', 'P3'], 'Fase': ['polpetta' if x else 'purea' for x in in_pol],
                             'T iniziale misurata [°C]': [d_s['T1'].iloc[0], d_s['T2'].iloc[0], d_s['T3'].iloc[0]]})
        st.dataframe(info.round(2), hide_index=True)
        comp = ce.composizione_prova(imp_s, sale_s)
        st.write(f"- **Configurazione** `{conf_s}` → polpette in {[f'P{i + 1}' for i in range(3) if imp_s.occupazione[conf_s][i]] or 'nessuna posizione'}")
        st.write(f"- **Tensione** {V_s:g} V → {p_s['V_rms']:.1f} V rms su {geo_s['Lx'] * 1000:.0f} mm = {p_s['V_rms'] / (geo_s['Lx'] * 100):.2f} V/cm")
        st.write(f"- **Ricetta** ({'esatta della tabella' if (ricetta_id and not sale_da_file) else 'sale dal file'}): burro {comp['w_burro']:.2f}%, "
                 f"sale {comp['w_sale']:.2f}%, fiocchi {comp['w_fiocchi']:.2f}%, acqua {comp['w_acqua']:.2f}%")
        st.write(f"- **Temperature iniziali del modello**: purea {p_s['T_iniziale']:.1f} °C, polpette {p_s['T_iniziale_polpetta']:.1f} °C "
                 "(medie delle sonde nella rispettiva fase)")
        st.write(f"- **Proprietà**: {lbl_prop}. σ purea = {p_s['matrice_a']:.4f}·T + {p_s['matrice_b']:.4f}; "
                 f"σ polpette = {p_s['polpetta_a']:.4f}·T + {p_s['polpetta_b']:.4f}")
        st.write(f"- **Durata** {p_s['durata']:.0f} s; temperatura ambiente {imp_s.T_amb:.0f} °C; mesh {h_mesh:g} mm")
        st.dataframe(tabella_prop(prop_s, prop_lett_s, p_s).round(4), hide_index=True)
        st.caption("I parametri «fattore» (su $k$ e su $\\rho c_p$ di ciascuna fase) **moltiplicano i valori calcolati con le relazioni di Choi–Okos**: "
                   "valgono 1 finché non vengono calibrati, cioè «nessuna correzione». L'ultima colonna mostra il valore effettivo a 20 °C "
                   "(Choi–Okos × fattore); $\\rho$ e $c_p$ compaiono nel modello solo attraverso il prodotto.")

    n_est = nodi_stimati(d_s)
    st.caption(f"Mesh: ~{n_est:,} nodi; tempo indicativo ~{n_est * S_NODO_PASSO * p_s['durata'] / dt_run:.0f} s.")
    if st.button("Simula questa prova", type="primary", key="ce_run"):
        barra = st.progress(0.0, text="Simulazione in corso...")
        ris = ce.simula_prova(d_s, imp_s, prop_s, h_mesh, dt_run,
                              progress_callback=lambda f: barra.progress(f, text=f"Simulazione in corso... {f * 100:.0f}%"))
        barra.empty()
        S['ultima'] = dict(ris=ris, d=d_s, metr=ce.confronta(ris, d_s), titolo=f"{id_prova(conf_s, V_s, sale_s)}  -  proprietà {lbl_prop}")
    if 'ultima' in S:
        u = S['ultima']
        g1, g2 = st.columns(2)
        g1.pyplot(fig_sovrapposizione(u['ris'], u['d'], u['titolo']), width=LARG + 120)
        g2.pyplot(fig_errori(u['ris'], u['d']), width=LARG)
        m = u['metr']
        st.dataframe(m.rename(columns={'sonda': 'Sonda', 'fase': 'Fase', 'RMSE': 'RMSE [°C]', 'bias': 'Bias medio [°C]',
                                       'errore_finale': 'Errore finale [°C]', 'errore_max': 'Errore max [°C]',
                                       'salto_exp': 'Salto misurato [°C]', 'salto_sim': 'Salto simulato [°C]'}).round(2),
                     hide_index=True)
        st.metric("RMSE complessivo", f"{np.sqrt((m['RMSE'] ** 2).mean()):.2f} °C")
        csv = io.StringIO(); csv.write("tempo_s,exp_P1,exp_P2,exp_P3,sim_P1,sim_P2,sim_P3\n")
        t_ = u['d']['tempo'].values.astype(float)
        simi = [np.interp(t_, u['ris']['tempo'], u['ris']['T_sonde'][n]) for n in ('P1', 'P2', 'P3')]
        for k, t in enumerate(t_):
            csv.write(f"{t:g}," + ",".join(f"{u['d'][f'T{i + 1}'].iloc[k]:.3f}" for i in range(3)) + "," +
                      ",".join(f"{simi[i][k]:.3f}" for i in range(3)) + "\n")
        st.download_button("Scarica il confronto (CSV)", csv.getvalue(), "confronto_prova.csv", "text/csv")

# =============================================================================
# 3. Calibrazione
# =============================================================================
with tab_cal:
    st.markdown("""
Si sceglie **quali proprietà del modello stimare dai dati**; tutte le altre restano ai valori di letteratura. La stima minimizza lo scarto
tra temperature simulate e misurate nelle sonde delle prove selezionate. La conducibilità è parametrizzata come
$\\sigma(T)=\\sigma_{20}\\,[1+\\alpha\\,(T-20)]$; gli altri parametri sono fattori moltiplicativi sui valori di letteratura.
""")
    k1, k2 = st.columns(2)
    sale_c = k1.selectbox("Concentrazione salina da calibrare", sali, index=min(3, len(sali) - 1), key="ce_sale_c")
    el_c = el[np.isclose(el['sale'], sale_c)]
    prop_c0 = ce.prop_letteratura(imp, sale_c)
    k2.info(f"Punto di partenza: {prop_c0['etichetta']}.")

    # --- prove di calibrazione ---
    V_cent = tensioni[len(tensioni) // 2]
    ids_c = list(el_c['id'])
    default_cal = [i for i in ids_c if f"| {V_cent:g} V |" in i]
    sel_cal = st.multiselect("Prove usate per calibrare", ids_c, default=default_cal, key=f"ce_selcal_{sale_c}",
                             help="Conviene calibrare su una tensione e validare sulle altre.")

    # --- tabella dei parametri: quale calibrare ---
    st.markdown("**Parametri da calibrare**")
    v0 = ce.valori_da_prop(prop_c0)
    tab0 = pd.DataFrame({
        'Calibra': [p['id'] == 'sig_m_s20' for p in ce.PARAMETRI],
        'Parametro': [p['nome'] for p in ce.PARAMETRI],
        'Unità': [p['unita'] for p in ce.PARAMETRI],
        'Letteratura': [v0[p['id']] for p in ce.PARAMETRI],
        'Min': [p['min'] for p in ce.PARAMETRI],
        'Max': [p['max'] for p in ce.PARAMETRI],
    })
    chiave_tab = f"ce_ptab_{sale_c}_{ricetta_id}_{sale_da_file}_{regressione_partenza}"
    tab_ed = st.data_editor(
        tab0, key=chiave_tab, hide_index=True, width='stretch',
        disabled=['Parametro', 'Unità', 'Letteratura'],
        column_config={'Calibra': st.column_config.CheckboxColumn('Calibra', width='small'),
                       'Letteratura': st.column_config.NumberColumn(format="%.4g"),
                       'Min': st.column_config.NumberColumn(format="%.4g"),
                       'Max': st.column_config.NumberColumn(format="%.4g")})
    scelti = [ce.PARAMETRI[i]['id'] for i in range(len(ce.PARAMETRI)) if bool(tab_ed.loc[i, 'Calibra'])]
    limiti = {ce.PARAMETRI[i]['id']: (float(tab_ed.loc[i, 'Min']), float(tab_ed.loc[i, 'Max'])) for i in range(len(ce.PARAMETRI))}
    st.caption("ρ e cp compaiono nel modello solo attraverso il prodotto ρ·cp (capacità termica volumetrica), quindi sono "
               "calibrabili solo insieme. Un parametro con poca influenza sulle sonde risulterà non identificabile (incertezza grande).")

    # --- obiettivo e opzioni ---
    ball_param = any(i.startswith(('sig_p', 'fk_p', 'fc_p')) for i in scelti)
    cc1, cc2 = st.columns(2)
    fasi_lbl = cc1.multiselect("Sonde usate nell'obiettivo", ["Sonde in purea", "Sonde nelle polpette"],
                               default=["Sonde in purea"] + (["Sonde nelle polpette"] if ball_param else []),
                               key=f"ce_fasi_{sale_c}_{'p' if ball_param else 'm'}",
                               help="Per calibrare proprietà delle polpette servono le sonde nelle polpette.")
    nfev = cc2.number_input("Valutazioni massime", value=14, min_value=3, max_value=80, key="ce_nfev")
    fasi = tuple(('purea' if 'purea' in f else 'polpetta') for f in fasi_lbl)

    n_par = len(scelti)
    sim_tot = (nfev + 1) * (n_par + 1) * max(1, len(sel_cal))
    if sel_cal:
        sim_s = np.mean([nodi_stimati(ce.estrai_prova(df, *[(el_c[el_c['id'] == i].iloc[0][c]) for c in ('configurazione', 'tensione', 'sale')])) for i in sel_cal]) \
            * S_NODO_PASSO * (270 / dt_cal)
        st.caption(f"Costo massimo: ~{sim_tot} simulazioni da ~{sim_s:.1f} s ≈ {sim_tot * sim_s / 60:.1f} minuti (stima; dipende dal server).")
    if n_par > 1:
        st.warning(f"Stai calibrando {n_par} parametri insieme: confronta il numero di parametri con quello dei dati e controlla "
                   "incertezze e correlazioni nei risultati. Parametri correlati o poco influenti non sono identificabili.")

    if st.button("Calibra", type="primary", key="ce_calibra", disabled=not (sel_cal and scelti and fasi)):
        prove = []
        for idp in sel_cal:
            r = el_c[el_c['id'] == idp].iloc[0]
            prove.append(ce.estrai_prova(df, r['configurazione'], r['tensione'], r['sale']))
        barra = st.progress(0.0, text="Calibrazione in corso...")
        tot = (nfev + 1) * (n_par + 1)
        t0 = time.time()
        try:
            res = ce.calibra(prove, imp, prop_c0, scelti, limiti=limiti, fasi=fasi, h_mm=h_mesh, dt=dt_cal, max_nfev=int(nfev),
                             progress_callback=lambda n: barra.progress(min(1.0, n / tot), text=f"Calibrazione: valutazione {n}"))
        except ValueError as e:
            barra.empty()
            st.error(str(e))
            st.stop()
        barra.empty()
        res.update(prove_usate=list(sel_cal), durata_s=time.time() - t0, h_mesh=h_mesh, dt=dt_cal, sale=float(sale_c))
        S['calib'][chiave_sale(sale_c)] = res
        st.rerun()

    cal = S['calib'].get(chiave_sale(sale_c))
    if cal:
        st.success(f"Calibrazione disponibile per sale {sale_c:g}% ({cal['durata_s']:.0f} s, {cal['n_chiamate']} valutazioni, "
                   f"mesh {cal['h_mesh']:g} mm).")
        q1, q2, q3, q4 = st.columns(4)
        q1.metric("RMSE prima", f"{cal['rmse_prima']:.2f} °C"); q2.metric("RMSE dopo", f"{cal['rmse']:.2f} °C")
        q3.metric("Parametri calibrati", cal['n_parametri']); q4.metric("Residui usati", cal['n_residui'])
        righe = []
        for t in cal['tabella']:
            rel = t['incertezza'] / t['calibrato'] if t['calibrato'] else np.nan
            stato = ('non identificabile' if (not np.isfinite(rel) or rel > 0.5) else ('al limite' if t['al_limite'] else 'ok'))
            if t['al_limite'] and stato == 'ok':
                stato = 'al limite'
            righe.append({'Parametro': t['parametro'], 'Unità': t['unita'], 'Letteratura': t['partenza'], 'Calibrato': t['calibrato'],
                          '± (1σ, approx.)': t['incertezza'], 'Variazione': t['calibrato'] / t['partenza'] - 1, 'Stato': stato})
        st.dataframe(pd.DataFrame(righe).round(4), hide_index=True)
        if cal['corr'] is not None and cal['n_parametri'] > 1:
            with st.expander("Correlazioni tra i parametri stimati"):
                st.dataframe(cal['corr'].round(2))
                forti = [(a, b, cal['corr'].loc[a, b]) for i, a in enumerate(cal['corr'].index) for b in cal['corr'].columns[i + 1:]
                         if abs(cal['corr'].loc[a, b]) > 0.9]
                for a, b, c_ in forti:
                    st.warning(f"{a} e {b} sono fortemente correlati ({c_:+.2f}): i dati non li distinguono, "
                               "conviene calibrarne uno solo.")
        st.write(f"Prove usate ({len(cal['prove_usate'])}): {', '.join(cal['prove_usate'])}  -  sonde: {', '.join(cal['fasi'])}")
        st.caption("I valori calibrati sono parametri *efficaci*: assorbono anche gli errori residui del modello (scambio, posizione "
                   "delle sonde, geometria). Vanno giudicati soprattutto dalla validazione sulle prove NON usate (scheda 4).")
    with st.expander("Tutte le calibrazioni disponibili"):
        if S['calib']:
            righe = []
            for k, v in sorted(S['calib'].items()):
                righe.append({'Sale [%]': k, 'Parametri': ', '.join(v['selezionati']), 'RMSE prima [°C]': v['rmse_prima'],
                              'RMSE dopo [°C]': v['rmse'], 'Prove': len(v['prove_usate'])})
            st.dataframe(pd.DataFrame(righe).round(3), hide_index=True)
        else:
            st.write("Nessuna calibrazione ancora eseguita.")

# =============================================================================
# 4. Validazione
# =============================================================================
with tab_val:
    st.markdown("Esegue **tutte le prove** della concentrazione scelta con le proprietà scelte e le confronta con le misure. "
                "Le prove non usate per calibrare mostrano quanto il modello è davvero predittivo.")
    sale_v = st.selectbox("Concentrazione salina", sali, index=min(3, len(sali) - 1), key="ce_sale_v")
    prop_v, lbl_v, prop_lett_v = scelta_proprieta(sale_v, "ce_modov", chiave_sale(sale_v) in S['calib'])
    el_v = el[np.isclose(el['sale'], sale_v)]
    sel_v = st.multiselect("Prove da eseguire", list(el_v['id']), default=list(el_v['id']), key=f"ce_selv_{sale_v}")
    if sel_v:
        n_m = np.mean([nodi_stimati(ce.estrai_prova(df, el_v[el_v['id'] == i].iloc[0]['configurazione'], el_v[el_v['id'] == i].iloc[0]['tensione'],
                                                    el_v[el_v['id'] == i].iloc[0]['sale'])) for i in sel_v])
        st.caption(f"{len(sel_v)} simulazioni; circa {len(sel_v) * n_m * S_NODO_PASSO * 270 / dt_cal:.0f} s.")
    if st.button("Esegui la validazione", type="primary", key="ce_valida", disabled=not sel_v):
        righe, dettagli = [], []
        barra = st.progress(0.0, text="Validazione in corso...")
        usate = set(S['calib'].get(chiave_sale(sale_v), {}).get('prove_usate', [])) if lbl_v.startswith("calibrate") else set()
        for k, idp in enumerate(sel_v):
            r = el_v[el_v['id'] == idp].iloc[0]
            d = ce.estrai_prova(df, r['configurazione'], r['tensione'], r['sale'])
            ris = ce.simula_prova(d, imp, prop_v, h_mesh, dt_cal)
            m = ce.confronta(ris, d)
            t = d['tempo'].values.astype(float)
            dettagli.append(dict(fasi=[('polpetta' if x else 'purea') for x in ris['in_polpetta']],
                                 exp=[d[f'T{i + 1}'].values for i in range(3)],
                                 sim=[np.interp(t, ris['tempo'], ris['T_sonde'][f'P{i + 1}']) for i in range(3)]))
            pu, po = m[m['fase'] == 'purea'], m[m['fase'] == 'polpetta']
            righe.append({'Prova': idp, 'configurazione': r['configurazione'], 'Tensione [V]': r['tensione'],
                          'Usata per calibrare': idp in usate,
                          'RMSE_tot': float(np.sqrt((m['RMSE'] ** 2).mean())),
                          'RMSE purea': float(np.sqrt((pu['RMSE'] ** 2).mean())) if len(pu) else np.nan,
                          'RMSE polpette': float(np.sqrt((po['RMSE'] ** 2).mean())) if len(po) else np.nan,
                          'Bias medio': float(m['bias'].mean())})
            barra.progress((k + 1) / len(sel_v), text=f"Validazione: {k + 1}/{len(sel_v)}")
        barra.empty()
        S['val'] = dict(tab=pd.DataFrame(righe), dettagli=dettagli, etichetta=lbl_v, sale=sale_v)
    if 'val' in S:
        vv = S['val']
        st.write(f"**Proprietà**: {vv['etichetta']}  (sale {vv['sale']:g}%)")
        tabv = vv['tab']
        mm1, mm2, mm3 = st.columns(3)
        mm1.metric("RMSE medio, tutte le prove", f"{tabv['RMSE_tot'].mean():.2f} °C")
        if tabv['Usata per calibrare'].any():
            mm2.metric("RMSE medio, prove usate", f"{tabv[tabv['Usata per calibrare']]['RMSE_tot'].mean():.2f} °C")
            if (~tabv['Usata per calibrare']).any():
                mm3.metric("RMSE medio, prove NON usate", f"{tabv[~tabv['Usata per calibrare']]['RMSE_tot'].mean():.2f} °C")
        st.dataframe(tabv.drop(columns=['configurazione']).round(2), hide_index=True)
        h1, h2 = st.columns(2)
        h1.pyplot(fig_parity(vv['dettagli']), width=LARG)
        h2.pyplot(fig_rmse_config(tabv), width=LARG)
        st.download_button("Scarica la tabella (CSV)", tabv.to_csv(index=False), "validazione.csv", "text/csv")

with st.expander("Ipotesi e limiti"):
    st.markdown("""
- **Sonde P1-P3** corrispondono a Temperatura 1-3 del file e ai centri delle posizioni P1-P3 della cella (assunto).
- **Ricetta**: scelta dalla Tabella 4.3; il sale viene dal file (fiocchi = resto) oppure è quello esatto della ricetta.
- **Temperature iniziali**: dalle sonde, distinte per fase; ambiente fisso (modificabile). Nel file variano molto tra le prove (7-26 °C).
- **Proprietà calibrate** = valori *efficaci*: assorbono errori residui del modello. Giudicare dalla validazione sulle prove non usate.
- **Mesh**: predefinita dall'analisi di sensitività (pagina dedicata); le temperature alle sonde sono stabili entro la tolleranza scelta.
- Il modello non include ebollizione, evaporazione interna né fluidodinamica.
""")
