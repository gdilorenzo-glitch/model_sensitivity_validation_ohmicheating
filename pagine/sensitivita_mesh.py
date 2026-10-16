"""
Pagina: Analisi di sensitività alla mesh.
Determina il passo di mesh più grossolano che dà risultati indipendenti dalla mesh entro una tolleranza,
e lo propone come predefinito per la pagina "Calibrazione e validazione".
"""

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

st.title("Analisi di sensitività alla mesh")
st.caption("Simula lo stesso caso con passi di mesh via via più fini e determina il **passo più grossolano** oltre il quale "
           "le temperature alle sonde non cambiano più oltre la tolleranza scelta. Il valore trovato diventa il predefinito "
           "della pagina «Calibrazione e validazione», dove si eseguono molte simulazioni e conviene la mesh più economica "
           "ancora accurata.")

S_NODO_PASSO = 6.5e-5       # secondi per nodo e passo termico (misurato)
KB_NODO = 0.017             # MB per nodo (misurato)

if 'sens' not in st.session_state:
    st.session_state.sens = {'raw': {}, 'chiave': None}
S = st.session_state.sens

PATTERN = {
    'RP (solo purea, nessuna polpetta)': (False, False, False),
    'Tris (3 polpette: P1, P2, P3)': (True, True, True),
    'Symm (2 polpette: P1, P3)': (True, False, True),
    'Asymm (2 polpette: P1, P2)': (True, True, False),
    'Middle (1 polpetta: P2)': (False, True, False),
    'Side (1 polpetta: P1)': (True, False, False),
}
TIPO = {1: 'RPF', 2: 'RPF', 3: 'RPFNB', 4: 'RPFNB', 5: 'RPFNB', 6: 'RPFNB'}
ETICHETTE_COMP = {i: f"Comp. {i} - {TIPO[i]}, sale {me.COMPOSIZIONI[i]['sale']:g}%" for i in range(1, 7)}
ETICHETTE_COMP['custom'] = 'Personalizzata (es. ricetta sperimentale con burro)'

# =============================================================================
# Parametri del caso
# =============================================================================
with st.sidebar:
    st.header("Caso da analizzare")
    with st.expander("Cella e ricetta", expanded=True):
        cella = st.selectbox("Cella", list(me.CELLE.keys()), key="sm_cella")
        Lx_mm = st.number_input("Distanza tra gli elettrodi [mm]", value=me.CELLE[cella]['Lx'] * 1000, min_value=10.0, key=f"sm_Lx_{cella}")
        id_comp = st.selectbox("Ricetta della purea (Tabella 4.3)", list(ETICHETTE_COMP.keys()),
                               format_func=lambda i: ETICHETTE_COMP[i], index=len(ETICHETTE_COMP) - 1, key="sm_comp")
        if id_comp == 'custom':
            b_c = st.number_input("Burro [% w/w]", value=5.8, min_value=0.0, key="sm_cb")
            s_c = st.number_input("Sale [% w/w]", value=2.9, min_value=0.0, key="sm_cs")
            w_c = st.number_input("Acqua [% w/w]", value=72.5, min_value=1.0, key="sm_cw")
            if b_c + s_c + w_c >= 100:
                st.error("La somma burro + sale + acqua deve essere minore di 100%.")
                st.stop()
            st.caption(f"Fiocchi di patate = {100 - b_c - s_c - w_c:.2f} % (resto).")
            c_ric = dict(burro=b_c, sale=s_c, fiocchi=100 - b_c - s_c - w_c, acqua=w_c, sigma=None)
        else:
            c_ric = me.COMPOSIZIONI[id_comp]
        sig_lett = c_ric['sigma']
        modi_sig = (["Letteratura (ricetta scelta)"] if sig_lett else []) + ["Manuale"]
        modo_sig = st.selectbox("σ(T) della purea", modi_sig, key=f"sm_modosig_{id_comp}")
        if modo_sig.startswith("Letteratura"):
            a_m, b_m = sig_lett[:2]
        else:
            s20_def = 1.6 if not sig_lett else 20 * sig_lett[0] + sig_lett[1]
            s20 = st.number_input("σ a 20 °C [S/m]", value=float(s20_def), min_value=0.05, key=f"sm_s20_{id_comp}",
                                  help="Valore rappresentativo delle prove sperimentali (1.6 S/m per la ricetta con burro e sale 2.9%).")
            al = st.number_input("α [1/K]", value=0.03, min_value=0.0, max_value=0.1, format="%.4f", key="sm_alpha")
            a_m, b_m = s20 * al, s20 * (1 - 20 * al)
        if not sig_lett:
            st.caption("Per questa ricetta non esiste regressione di letteratura: σ manuale.")
    with st.expander("Condizioni di prova", expanded=True):
        E = st.number_input("Campo elettrico efficace [V/cm]", value=4.0, min_value=0.1, key="sm_E")
        durata = st.number_input("Durata [s]", value=270.0, min_value=10.0, key="sm_dur")
        dt = st.selectbox("Passo temporale [s]", [10.0, 5.0, 15.0], key="sm_dt")
        T0 = st.number_input("Temperatura iniziale [°C]", value=20.0, key="sm_T0")
        T_amb = st.number_input("Temperatura ambiente [°C]", value=21.0, key="sm_Ta")
    with st.expander("Configurazioni e mesh", expanded=True):
        confs = st.multiselect("Configurazioni da analizzare", list(PATTERN.keys()),
                               default=['Tris (3 polpette: P1, P2, P3)', 'Side (1 polpetta: P1)'], key="sm_confs",
                               help="Tris è la più esigente (gap di 2.5 mm tra polpette). Il passo raccomandato è il più restrittivo tra le configurazioni scelte.")
        opz_h = [10.0, 8.0, 6.0, 5.0, 4.0, 3.5, 3.0, 2.5, 2.0, 1.75, 1.5]
        h_list = st.multiselect("Passi di mesh da confrontare [mm]", opz_h, default=[8.0, 6.0, 5.0, 4.0, 3.0, 2.5, 2.0],
                                key="sm_hlist", help="Il più fine è la soluzione di riferimento.")
        tol = st.number_input("Tolleranza sulle temperature alle sonde [°C]", value=0.2, min_value=0.01, step=0.05, key="sm_tol",
                              help="Scarto massimo ammesso rispetto alla mesh di riferimento, sia sulla temperatura finale sia "
                                   "come RMSE sull'intera curva.")
        limite_nodi = st.number_input("Limite massimo di nodi", value=40000, min_value=2000, step=2000, key="sm_lim",
                                      help="Sul cloud gratuito (~1 GB) restare sotto ~35.000.")

# =============================================================================
# Costruzione dei casi e stima del costo
# =============================================================================
if len(h_list) < 3 or not confs:
    st.info("Scegli almeno una configurazione e almeno tre passi di mesh (il più fine è il riferimento).")
    st.stop()
h_list = sorted(h_list, reverse=True)            # dal più grossolano al più fine


def geo_caso(nome):
    return me.geometria(cella, 'RP (solo purea)', dict(occupazione=list(PATTERN[nome]), Lx=Lx_mm / 1000.0))


def parametri_caso():
    c = c_ric
    p = dict(me.PARAMETRI_DEFAULT)
    p.update(comp_matrice=me.composizione_purea(c['burro'], c['sale'], c['fiocchi'], c['acqua']),
             matrice_a=float(a_m), matrice_b=float(b_m), V_rms=E * Lx_mm / 10.0,
             T_iniziale=T0, T_iniziale_polpetta=T0, T_amb=T_amb, durata=durata, dt=dt,
             prop_modo='choi_okos_T', h_modo='naturale_irraggiamento')
    return p


chiave_caso = (cella, round(Lx_mm, 3), id_comp, round(c_ric['burro'], 3), round(c_ric['sale'], 3), round(c_ric['acqua'], 3), round(a_m, 5), round(b_m, 5), E, durata, dt, T0, T_amb)
if S['chiave'] != chiave_caso:
    S['raw'], S['chiave'] = {}, chiave_caso            # il caso e' cambiato: i risultati precedenti non valgono piu'

passi = max(1, int(round(durata / dt)))
tabella_costo, scartate = [], []
tot_s = 0.0
for nome in confs:
    g = geo_caso(nome)
    for h in h_list:
        n, _ = me.stima_nodi(g, h / 1000)
        if n > limite_nodi:
            scartate.append((nome, h, n)); continue
        if (nome, h) not in S['raw']:
            tot_s += n * S_NODO_PASSO * passi
        tabella_costo.append((nome, h, n))
h_ok = sorted({h for _, h, _ in tabella_costo}, reverse=True)
if scartate:
    st.warning("Escluse perché oltre il limite di nodi: " + ", ".join(sorted({f"{h:g} mm ({n:,} nodi)" for _, h, n in scartate})) +
               ". Aumenta il limite solo se hai memoria sufficiente (uso in locale).")
if len(h_ok) < 3:
    st.error("Restano meno di tre passi di mesh utilizzabili: alza il limite di nodi o scegli passi più grossolani.")
    st.stop()
n_max = max(n for _, _, n in tabella_costo)
c1, c2, c3 = st.columns(3)
c1.metric("Simulazioni da eseguire", f"{sum(1 for nome, h, _ in tabella_costo if (nome, h) not in S['raw'])}")
c2.metric("Mesh più fine", f"{min(h_ok):g} mm, {n_max:,} nodi")
c3.metric("Tempo indicativo", f"{tot_s / 60:.1f} min" if tot_s >= 90 else f"{tot_s:.0f} s",
          help="Dipende dal server; sul cloud gratuito è tipicamente 1.5-2 volte più lento.")

# =============================================================================
# Esecuzione
# =============================================================================
if st.button("Esegui l'analisi di sensitività", type="primary"):
    p0 = parametri_caso()
    da_fare = [(nome, h) for nome, h, _ in tabella_costo if (nome, h) not in S['raw']]
    barra = st.progress(0.0, text="Analisi in corso...")
    for k, (nome, h) in enumerate(da_fare):
        g = geo_caso(nome)
        mesh, basis, h_eff = me.costruisci_mesh(g, h / 1000.0)
        t0 = time.time()
        ris = me.simula(g, mesh, basis, h_eff, p0)
        S['raw'][(nome, h)] = dict(tempo=ris['tempo'], T={n: ris['T_sonde'][n] for n in ('P1', 'P2', 'P3')},
                                   nodi=basis.N, h_eff=h_eff, s=time.time() - t0)
        barra.progress((k + 1) / len(da_fare), text=f"Analisi in corso: {nome}, passo {h:g} mm ({k + 1}/{len(da_fare)})")
    barra.empty()
    st.rerun()


# =============================================================================
# Analisi dei risultati
# =============================================================================
def analizza(nome):
    """Errori rispetto alla mesh piu' fine, per una configurazione. Restituisce (DataFrame, h raccomandato, h riferimento)."""
    hs = [h for h in h_ok if (nome, h) in S['raw']]
    if len(hs) < 3:
        return None, None, None, None
    h_rif = min(hs)
    ref = S['raw'][(nome, h_rif)]
    salto_ref = max(ref['T'][n][-1] - ref['T'][n][0] for n in ('P1', 'P2', 'P3'))
    righe = []
    for h in hs:
        r = S['raw'][(nome, h)]
        e_fin = max(abs(r['T'][n][-1] - ref['T'][n][-1]) for n in ('P1', 'P2', 'P3'))
        e_rmse = max(float(np.sqrt(np.mean((r['T'][n] - ref['T'][n]) ** 2))) for n in ('P1', 'P2', 'P3'))
        righe.append(dict(h=h, h_eff=r['h_eff'] * 1000, nodi=r['nodi'], tempo_s=r['s'],
                          T_P1=r['T']['P1'][-1], T_P2=r['T']['P2'][-1], T_P3=r['T']['P3'][-1],
                          err_finale=e_fin, err_rmse=e_rmse, rif=(h == h_rif)))
    df = pd.DataFrame(righe)
    df['ok'] = (df[['err_finale', 'err_rmse']].max(axis=1) <= tol)
    # raccomandato: il piu' grossolano tale che esso e TUTTI quelli piu' fini (esclusa la mesh di riferimento) rispettano la tolleranza
    h_rac = h_rif
    for h in sorted([x for x in hs if x != h_rif], reverse=True):
        sotto = df[(df['h'] <= h) & (~df['rif'])]
        if sotto['ok'].all():
            h_rac = h
            break
    return df, h_rac, h_rif, salto_ref


risultati = {nome: analizza(nome) for nome in confs}
pronti = {nome: r for nome, r in risultati.items() if r[0] is not None}
if not pronti:
    st.info("Premi «Esegui l'analisi di sensitività» per avviare le simulazioni.")
    st.stop()

# raccomandazione globale: la piu' restrittiva tra le configurazioni
h_globale = min(r[1] for r in pronti.values())
non_conv = [nome for nome, r in pronti.items() if r[1] == r[2]]
st.success(f"Passo di mesh minimo raccomandato: **{h_globale:g} mm** (tolleranza {tol:g} °C). "
           "Sarà il predefinito in «Calibrazione e validazione».")
if non_conv:
    st.warning("Per " + ", ".join(non_conv) + " nessuna mesh più grossolana del riferimento rispetta la tolleranza: "
               "la convergenza non è dimostrata. Aggiungi passi più fini oppure aumenta la tolleranza.")
if len(pronti) < len(confs):
    st.info("Alcune configurazioni non hanno ancora tutte le simulazioni: premi di nuovo il pulsante.")

geo_rif = geo_caso(confs[0])
n_rac, _ = me.stima_nodi(geo_rif, h_globale / 1000)
n_fine, _ = me.stima_nodi(geo_rif, min(h_ok) / 1000)
m1, m2, m3 = st.columns(3)
m1.metric("Nodi a mesh raccomandata", f"{n_rac:,}")
m2.metric("Nodi a mesh di riferimento", f"{n_fine:,}")
m3.metric("Risparmio di calcolo (stima)", f"{n_fine / n_rac:.1f}×")

st.session_state['mesh_minima'] = dict(h_mm=float(h_globale), tol=float(tol), configs=list(confs),
                                       riferimento_mm=float(min(h_ok)), quando=time.strftime('%H:%M:%S'))

# ---- grafici ----
fig, assi = plt.subplots(1, 3, figsize=(16, 4.3))
colori = plt.cm.tab10.colors
for k, (nome, (df, h_rac, h_rif, salto)) in enumerate(pronti.items()):
    breve = nome.split(' ')[0]
    d = df[~df['rif']]
    assi[0].plot(df['h_eff'], df['T_P1'], 'o-', color=colori[k], label=f'{breve} P1')
    assi[0].plot(df['h_eff'], df['T_P2'], 's--', color=colori[k], alpha=0.6, label=f'{breve} P2')
    assi[0].plot(df['h_eff'], df['T_P3'], '^:', color=colori[k], alpha=0.6, label=f'{breve} P3')
    assi[1].loglog(d['h_eff'], d['err_finale'].clip(lower=1e-4), 'o-', color=colori[k], label=breve)
    assi[2].loglog(d['h_eff'], d['err_rmse'].clip(lower=1e-4), 'o-', color=colori[k], label=breve)
assi[0].set_xlabel('passo di mesh effettivo [mm]'); assi[0].set_ylabel('T finale [°C]')
assi[0].set_title('Temperatura finale alle sonde'); assi[0].legend(fontsize=6, ncol=2); assi[0].invert_xaxis()
for ax, tit in ((assi[1], 'Errore sulla T finale (max sulle sonde)'), (assi[2], 'RMSE sull\'intera curva (max sulle sonde)')):
    ax.axhline(tol, color='r', ls='--', lw=1.5, label=f'tolleranza {tol:g} °C')
    ax.axvline(h_globale, color='g', ls=':', lw=1.5, label=f'raccomandato {h_globale:g} mm')
    ax.set_xlabel('passo di mesh effettivo [mm]'); ax.set_ylabel('errore vs riferimento [°C]'); ax.set_title(tit)
    ax.grid(alpha=.3, which='both'); ax.legend(fontsize=7)
for ax in assi:
    ax.grid(alpha=.3, which='both')
plt.tight_layout()
st.pyplot(fig, width=1500)

# ---- tabelle di dettaglio ----
for nome, (df, h_rac, h_rif, salto) in pronti.items():
    with st.expander(f"Dettaglio: {nome}  -  raccomandato {h_rac:g} mm, salto di temperatura di riferimento {salto:.1f} °C"):
        t = df.rename(columns={'h': 'Passo richiesto [mm]', 'h_eff': 'Passo effettivo [mm]', 'nodi': 'Nodi', 'tempo_s': 'Tempo [s]',
                               'T_P1': 'T finale P1 [°C]', 'T_P2': 'T finale P2 [°C]', 'T_P3': 'T finale P3 [°C]',
                               'err_finale': 'Errore T finale [°C]', 'err_rmse': 'RMSE curva [°C]', 'rif': 'Riferimento',
                               'ok': 'Entro tolleranza'})
        st.dataframe(t.round(3), hide_index=True)
        d = df[~df['rif']]
        d = d[d['err_finale'] > 1e-4]
        if len(d) >= 3:
            ordine = np.polyfit(np.log(d['h_eff']), np.log(d['err_finale']), 1)[0]
            st.caption(f"Ordine di convergenza osservato (pendenza log-log dell'errore): ≈ {ordine:.1f}.")

with st.expander("Come leggere questa analisi"):
    st.markdown("""
- L'errore è calcolato **rispetto alla mesh più fine tra quelle scelte**, non rispetto alla soluzione esatta: se la mesh di riferimento fosse ancora lontana dalla convergenza, l'errore sarebbe sottostimato. Un buon segno è che l'errore diminuisca regolarmente con il passo.
- Una mesh è «raccomandata» solo se **lei e tutte quelle più fini** (escluso il riferimento) rispettano la tolleranza: così si evita di scegliere una mesh che passa per coincidenza.
- Le grandezze controllate sono le **temperature alle sonde** (quelle confrontate con gli esperimenti). Campi locali come il picco di temperatura nei gap sottili tra polpette ed elettrodi possono richiedere mesh più fini.
- La struttura è strutturata e non conforme alle sfere: l'interfaccia purea/polpetta è a gradini e converge lentamente (ordine <2).
- Il passo raccomandato è il più restrittivo tra le configurazioni scelte, quindi vale per tutte.
""")
