"""
Pagina: riscaldamento ohmico MULTIFASE (purea di patate + polpette di pollo)  -  v2
Usa il motore di calcolo multifase_engine.py (cartella superiore).
"""

import io
import sys
import uuid
import tempfile
from pathlib import Path
# rende importabile multifase_engine.py (cartella superiore) comunque venga lanciata l'app
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.animation import FuncAnimation, PillowWriter
import streamlit as st

import multifase_engine as me

st.title("Riscaldamento ohmico del sistema multifase bicomponente")
st.caption("Modello elettro-termico 3D accoppiato con proprietà variabili per fase: σ(T) dalle Tabelle 5.1 e 6.1, "
           "proprietà termofisiche da composizione (Choi–Okos), scambio superficiale h(T), tensione sinusoidale a 50 Hz.")

LARG_2D = 480
LARG_SEZ = 900
TMP = tempfile.gettempdir() + '/'
KB_NODO = 0.017       # MB di RAM per nodo (misurato)
S_NODO_PASSO = 6.5e-5  # secondi per nodo e per passo termico (misurato)

if 'mf' not in st.session_state:
    st.session_state.mf = {}
S = st.session_state.mf


# =============================================================================
# Utilità
# =============================================================================
def griglia(coords):
    x_u, y_u, z_u = np.unique(coords[0]), np.unique(coords[1]), np.unique(coords[2])
    ix = np.searchsorted(x_u, coords[0])
    iy = np.searchsorted(y_u, coords[1])
    iz = np.searchsorted(z_u, coords[2])

    def a_griglia(campo):
        G = np.empty((len(x_u), len(y_u), len(z_u)))
        G[ix, iy, iz] = campo
        return G
    return x_u, y_u, z_u, a_griglia


def cerchi_sezione_xz(ax, geo, y, colore='white'):
    for (xc, yc, zc) in geo['centri']:
        dy = y - yc
        if abs(dy) < geo['R']:
            ax.add_patch(Circle((xc * 1000, zc * 1000), np.sqrt(geo['R'] ** 2 - dy ** 2) * 1000,
                                fill=False, ec=colore, lw=1.4, ls='--'))


def indice_piu_vicino(arr, v):
    return int(np.argmin(np.abs(arr - v)))


# =============================================================================
# Figure
# =============================================================================
def fig_geometria(geo):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.4))
    Lx, Ly, Lz = geo['Lx'] * 1000, geo['Ly'] * 1000, geo['Lz'] * 1000
    for ax, Lh, etichetta, idx_h in ((a1, Ly, 'y [mm]', 1), (a2, Lz, 'z [mm]', 2)):
        ax.add_patch(Rectangle((0, 0), Lx, Lh, fc='#d9d9d9', ec='k'))
        for c in geo['centri']:
            ax.add_patch(Circle((c[0] * 1000, c[idx_h] * 1000), geo['R'] * 1000, fc='#f2e3c6', ec='k'))
        for nome, pos in geo['sonde'].items():
            ax.plot(pos[0] * 1000, pos[idx_h] * 1000, 'ko', ms=4)
            ax.annotate(nome, (pos[0] * 1000, pos[idx_h] * 1000), textcoords='offset points',
                        xytext=(0, 7), ha='center', fontsize=8)
        ax.plot([0, 0], [0, Lh], color='tab:red', lw=4, solid_capstyle='butt')
        ax.plot([Lx, Lx], [0, Lh], color='tab:red', lw=4, solid_capstyle='butt')
        ax.set_xlim(-6, Lx + 6); ax.set_ylim(-6, Lh + 10)
        ax.set_aspect('equal'); ax.set_xlabel('x [mm]'); ax.set_ylabel(etichetta)
    a1.set_title("Vista dall'alto (xy)"); a2.set_title('Vista frontale (xz) - superficie libera in alto')
    a2.axhline(Lz, color='tab:blue', lw=1.5, ls=':')
    fig.suptitle('Elettrodi (rosso) in x = 0 e x = Lx; polpette in beige; punti = sonde', fontsize=9)
    plt.tight_layout()
    return fig


def fig_mesh_sezioni(mesh, geo, phi_n, coords):
    """La mesh COSTRUITA: spigoli reali dei tetraedri nei tre piani centrali, sopra la mappa di fase."""
    x_u, y_u, z_u, ag = griglia(coords)
    G = ag(phi_n)
    j = indice_piu_vicino(y_u, geo['Ly'] / 2)
    k = indice_piu_vicino(z_u, geo['Lz'] / 2)
    x_ref = geo['centri'][0][0] if geo['centri'] else geo['Lx'] / 2
    i = indice_piu_vicino(x_u, x_ref)
    # (asse fisso, quota del piano, ascisse, ordinate, mappa di fase, titolo, etichetta x, etichetta y)
    piani = [
        (1, y_u[j], x_u, z_u, G[:, j, :].T, f'Piano xz (y = {y_u[j] * 1000:.1f} mm)', 'x [mm]', 'z [mm]'),
        (2, z_u[k], x_u, y_u, G[:, :, k].T, f'Piano xy (z = {z_u[k] * 1000:.1f} mm)', 'x [mm]', 'y [mm]'),
        (0, x_u[i], y_u, z_u, G[i, :, :].T, f'Piano yz (x = {x_u[i] * 1000:.1f} mm)', 'y [mm]', 'z [mm]'),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(17, 3.6),
                             gridspec_kw={'width_ratios': [geo['Lx'], geo['Lx'], geo['Ly']]})
    for ax, (asse, quota, u, v, campo, titolo, lx, ly) in zip(axes, piani):
        c = ax.pcolormesh(u * 1000, v * 1000, campo, cmap='YlOrBr', shading='auto', vmin=0, vmax=1, alpha=0.85)
        seg, _ = me.spigoli_piano(mesh, asse, quota)
        ax.add_collection(LineCollection(seg * 1000, colors='k', linewidths=0.35, alpha=0.7))
        for (xc, yc, zc) in geo['centri']:
            # centro della sfera proiettato nel piano e distanza dal piano
            centro_2d, d = {1: ((xc, zc), quota - yc), 2: ((xc, yc), quota - zc), 0: ((yc, zc), quota - xc)}[asse]
            if abs(d) < geo['R']:
                ax.add_patch(Circle((centro_2d[0] * 1000, centro_2d[1] * 1000), np.sqrt(geo['R'] ** 2 - d ** 2) * 1000,
                                    fill=False, ec='tab:red', lw=1.3))
        ax.set_aspect('equal'); ax.set_xlabel(lx); ax.set_ylabel(ly); ax.set_title(titolo, fontsize=9)
    fig.colorbar(c, ax=axes, label='frazione di polpetta', shrink=0.8)
    fig.suptitle('Mesh costruita: spigoli dei tetraedri (nero), fase (colore), contorno esatto della sfera (rosso)', fontsize=10)
    return fig


def fig_mesh_3d(mesh, geo):
    seg = me.spigoli_superficie(mesh) * 1000
    cap = 30000
    sotto = False
    if len(seg) > cap:
        seg = seg[np.random.RandomState(0).choice(len(seg), cap, replace=False)]
        sotto = True
    fig = plt.figure(figsize=(7, 5.5))
    ax = fig.add_subplot(111, projection='3d')
    ax.add_collection3d(Line3DCollection(seg, colors='tab:blue', linewidths=0.3, alpha=0.6))
    ax.set_xlim(0, geo['Lx'] * 1000); ax.set_ylim(0, geo['Ly'] * 1000); ax.set_zlim(0, geo['Lz'] * 1000)
    ax.set_box_aspect((geo['Lx'], geo['Ly'], geo['Lz']))
    ax.set_xlabel('x [mm]'); ax.set_ylabel('y [mm]'); ax.set_zlabel('z [mm]')
    ax.set_title('Superficie esterna della mesh' + (f' (mostrati {cap} spigoli su {len(seg)}+)' if sotto else ''), fontsize=10)
    plt.tight_layout()
    return fig


def fig_sigma(p):
    T = np.linspace(0, 100, 101)
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    ax.plot(T, p['matrice_a'] * T + p['matrice_b'], color='tab:orange', label='purea')
    ax.plot(T, p['polpetta_a'] * T + p['polpetta_b'], color='tab:brown', label='polpette')
    ax.set_xlabel('T [°C]'); ax.set_ylabel('σ [S/m]'); ax.grid(alpha=.3); ax.legend(fontsize=8)
    ax.set_title('Conducibilità elettrica delle due fasi', fontsize=10)
    plt.tight_layout()
    return fig


def fig_proprieta(p):
    T = np.linspace(0, 100, 101)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
    for nome, comp, colore in (('purea', p['comp_matrice'], 'tab:orange'), ('polpetta', p['comp_polpetta'], 'tab:brown')):
        if p['prop_modo'] == 'manuali':
            k = np.full_like(T, p['k_matrice' if nome == 'purea' else 'k_polpetta'])
            r = np.full_like(T, p['rho_matrice' if nome == 'purea' else 'rho_polpetta'])
            c = np.full_like(T, p['cp_matrice' if nome == 'purea' else 'cp_polpetta'])
        elif p['prop_modo'] == 'choi_okos_Tref':
            k0, r0, c0 = me.proprieta_choi_okos(comp, p['T_rif_prop'])
            k, r, c = np.full_like(T, float(k0)), np.full_like(T, float(r0)), np.full_like(T, float(c0))
        else:
            k, r, c = me.proprieta_choi_okos(comp, T)
        axes[0].plot(T, k, color=colore, label=nome, lw=2)
        axes[1].plot(T, r, color=colore, label=nome, lw=2)
        axes[2].plot(T, c, color=colore, label=nome, lw=2)
    for ax, y in zip(axes, ('k [W/m/K]', 'ρ [kg/m³]', 'cp [J/kg/K]')):
        ax.set_xlabel('T [°C]'); ax.set_ylabel(y); ax.grid(alpha=.3); ax.legend(fontsize=8)
    plt.tight_layout()
    return fig


def fig_h(p, geo):
    L = geo['Lx'] * geo['Ly'] / (2 * (geo['Lx'] + geo['Ly']))
    Ts = np.linspace(p['T_amb'] + 1, 100, 100)
    fig, ax = plt.subplots(figsize=(6, 4))
    if p['h_modo'] == 'costante':
        ax.plot(Ts, np.full_like(Ts, p['h_conv']), lw=2, label='h costante')
    else:
        hn, hr, he = me.componenti_h(Ts, p, L)
        f = p.get('fattore_h', 1.0)
        ax.plot(Ts, f * hn, label='convezione naturale (McAdams)')
        ax.plot(Ts, f * hr, label='irraggiamento')
        if p['h_modo'] == 'naturale_irraggiamento_evaporazione':
            ax.plot(Ts, f * he, label='evaporazione (equivalente)')
        ax.plot(Ts, me.h_equivalente(Ts, p, L), 'k', lw=2, label='h totale equivalente')
    ax.set_xlabel('T superficiale [°C]'); ax.set_ylabel('h [W/m²/K]'); ax.grid(alpha=.3); ax.legend(fontsize=8)
    ax.set_title('Scambio sulla superficie libera superiore')
    plt.tight_layout()
    return fig


def fig_sonde(ris):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for n, v in ris['T_sonde'].items():
        fase = 'polpetta' if ris['fase_sonda'][n] > 0.5 else 'purea'
        ax.plot(ris['tempo'], v, label=f'{n} ({fase})', lw=2)
    ax.set_xlabel('tempo [s]'); ax.set_ylabel('T [°C]')
    ax.set_title('Temperatura nelle sonde P1-P3'); ax.grid(alpha=.3); ax.legend(fontsize=8)
    plt.tight_layout()
    return fig


def fig_sezioni(ris, k, j):
    geo = ris['geo']
    x_u, y_u, z_u, ag = griglia(ris['coords'])
    T = ris['T_hist'][k]
    sig = me.sigma_campo(T, ris['phi_n'], ris['p'])
    q = ris['q_hist'][k] * 1e-6
    campi = [('T [°C]', ag(T), 'inferno'), ('σ [S/m]', ag(sig), 'viridis'), ('q Joule [MW/m³]', ag(q), 'magma')]
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.6))
    for ax, (tit, G, cmap) in zip(axes, campi):
        c = ax.pcolormesh(x_u * 1000, z_u * 1000, G[:, j, :].T, cmap=cmap, shading='auto')
        fig.colorbar(c, ax=ax, label=tit)
        cerchi_sezione_xz(ax, geo, y_u[j])
        ax.set_aspect('equal'); ax.set_xlabel('x [mm]'); ax.set_ylabel('z [mm]')
        ax.set_title(f'{tit}  -  t = {ris["tempo"][k]:.0f} s  (y = {y_u[j] * 1000:.1f} mm)', fontsize=9)
    plt.tight_layout()
    return fig


def fig_cold_spot(ris, met):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(ris['tempo'], met['T_matrice_media'], label='purea (media)', lw=2)
    ax.plot(ris['tempo'], met['T_polpette_media'], label='polpette (media)', lw=2)
    ax.plot(ris['tempo'], met['T_polpette_min'], label='polpette (minimo)', lw=1.5, ls='--')
    ax.set_xlabel('tempo [s]'); ax.set_ylabel('T [°C]'); ax.grid(alpha=.3); ax.legend(fontsize=8)
    ax.set_title('Purea vs polpette: temperatura media nel volume')
    plt.tight_layout()
    return fig


def fig_scarto(ris, met):
    d = met['T_matrice_media'] - met['T_polpette_media']
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(ris['tempo'], d, color='tab:red', lw=2)
    ax.set_xlabel('tempo [s]'); ax.set_ylabel('T purea - T polpette [°C]')
    ax.set_title('Ritardo di riscaldamento delle polpette (cold spot)')
    ax.grid(alpha=.3)
    plt.tight_layout()
    return fig


def fig_profilo_x(ris, indici):
    geo = ris['geo']
    x_u, y_u, z_u, ag = griglia(ris['coords'])
    j = indice_piu_vicino(y_u, geo['Ly'] / 2)
    k = indice_piu_vicino(z_u, geo['Lz'] / 2)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i in indici:
        ax.plot(x_u * 1000, ag(ris['T_hist'][i])[:, j, k], label=f't = {ris["tempo"][i]:.0f} s')
    for (xc, _, _) in geo['centri']:
        ax.axvspan((xc - geo['R']) * 1000, (xc + geo['R']) * 1000, color='#f2e3c6', alpha=.5)
    ax.set_xlabel('x [mm]  (0 = elettrodo; zone beige = polpette)'); ax.set_ylabel('T [°C]')
    ax.set_title(f'Profilo lungo x sulla mezzeria (y = {y_u[j] * 1000:.1f}, z = {z_u[k] * 1000:.1f} mm)')
    ax.grid(alpha=.3); ax.legend(fontsize=8)
    plt.tight_layout()
    return fig


def fig_potenza(ris):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(ris['tempo'], ris['P'], lw=2, color='tab:purple')
    axes[0].set_ylabel('Potenza elettrica P [W]'); axes[0].set_title('Potenza dissipata (integrale di q su V)')
    I = ris['P'] / ris['V_rms']
    axes[1].plot(ris['tempo'], I, lw=2, color='tab:green')
    axes[1].set_ylabel('Corrente efficace I [A]'); axes[1].set_title(f'Corrente  (V_rms = {ris["V_rms"]:.1f} V)')
    for ax in axes:
        ax.set_xlabel('tempo [s]'); ax.grid(alpha=.3)
    plt.tight_layout()
    return fig


def fig_scambio(ris):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ris['tempo'], ris['h_hist'], lw=2, color='tab:blue')
    ax.set_xlabel('tempo [s]'); ax.set_ylabel('h equivalente medio [W/m²/K]')
    ax.set_title('Coefficiente di scambio sulla superficie libera nel tempo'); ax.grid(alpha=.3)
    plt.tight_layout()
    return fig


def genera_gif(ris, j):
    x_u, y_u, z_u, ag = griglia(ris['coords'])
    frames = [ag(T)[:, j, :].T for T in ris['T_hist']]
    vmin, vmax = ris['T_hist'][0].min(), ris['T_hist'][-1].max()
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    quad = ax.pcolormesh(x_u * 1000, z_u * 1000, frames[0], cmap='inferno', shading='auto', vmin=vmin, vmax=vmax)
    fig.colorbar(quad, ax=ax, label='T [°C]')
    cerchi_sezione_xz(ax, ris['geo'], y_u[j])
    tit = ax.set_title('')
    ax.set_aspect('equal'); ax.set_xlabel('x [mm]'); ax.set_ylabel('z [mm]')

    def upd(i):
        quad.set_array(frames[i].ravel())
        tit.set_text(f'Sezione xz (y = {y_u[j] * 1000:.1f} mm), t = {ris["tempo"][i]:.0f} s')
        return quad, tit
    anim = FuncAnimation(fig, upd, frames=len(frames), interval=90)
    path = TMP + f'multifase_{uuid.uuid4().hex[:8]}.gif'
    anim.save(path, writer=PillowWriter(fps=10))
    plt.close(fig)
    return path


# =============================================================================
# Barra laterale: parametri
# =============================================================================
ETICHETTE_COMP = {
    1: 'Comp. 1 - RPF, 5.7% sale (UCD)', 2: 'Comp. 2 - RPF, 10.9% sale (UCD)',
    3: 'Comp. 3 - RPFNB, 2.75% sale (UNISA)', 4: 'Comp. 4 - RPFNB, 1.39% sale (UNISA)',
    5: 'Comp. 5 - RPFNB, 0.74% sale (UNISA)', 6: 'Comp. 6 - RPFNB, 0.37% sale (UNISA)',
}

with st.sidebar:
    st.header("Parametri")

    # ---- 1) cella, configurazione, geometria editabile ----
    with st.expander("1. Cella e configurazione", expanded=True):
        cella = st.selectbox("Cella sperimentale", list(me.CELLE.keys()), key="mf_cella")
        conf = st.selectbox("Configurazione", list(me.CONFIGURAZIONI.keys()), key="mf_conf")
        modifica = st.checkbox("Modifica la geometria della cella", value=False, key="mf_edit")
        override = None
        if modifica:
            c0 = me.CELLE[cella]
            suf = f"{cella}_{conf}"
            Lx_m = st.number_input("Lx: distanza tra gli elettrodi [mm]", value=c0['Lx'] * 1000, min_value=10.0, step=1.0, key=f"mf_Lx_{suf}")
            Ly_m = st.number_input("Ly: larghezza [mm]", value=c0['Ly'] * 1000, min_value=10.0, step=1.0, key=f"mf_Ly_{suf}")
            Lz_m = st.number_input("Lz: altezza di riempimento [mm]", value=c0['Lz'] * 1000, min_value=5.0, step=1.0, key=f"mf_Lz_{suf}")
            D_m = st.number_input("Diametro polpetta [mm]", value=me.D_POLPETTA * 1000, min_value=2.0, step=1.0, key=f"mf_D_{suf}")
            mar_m = st.number_input("Distanza parete-polpetta, lungo x [mm]", value=c0['margine'] * 1000, min_value=0.0, step=0.25, key=f"mf_mar_{suf}")
            spz_m = st.number_input("Spaziatura sonde in RP [mm]", value=c0['spaz_rp'] * 1000, min_value=1.0, step=1.0, key=f"mf_spz_{suf}")
            occ_def = [i in me.CONFIGURAZIONI[conf] for i in range(3)]
            st.caption("Posizioni occupate da una polpetta:")
            o1, o2, o3 = st.columns(3)
            occ = [o1.checkbox("P1", value=occ_def[0], key=f"mf_o1_{suf}"),
                   o2.checkbox("P2", value=occ_def[1], key=f"mf_o2_{suf}"),
                   o3.checkbox("P3", value=occ_def[2], key=f"mf_o3_{suf}")]
            override = dict(Lx=Lx_m / 1000, Ly=Ly_m / 1000, Lz=Lz_m / 1000, D=D_m / 1000,
                            margine=mar_m / 1000, spaz_rp=spz_m / 1000, occupazione=occ)
        h_mm = st.slider("Passo di mesh [mm]", 0.5, 10.0, 2.5, 0.1, key="mf_h",
                         help="Valori sotto 0.5 mm non sono praticabili (decine di milioni di nodi). "
                              "Più piccolo = più accurato ma molto più lento e con più memoria.")
        limite_nodi = st.number_input("Limite massimo di nodi", value=70000, min_value=1000, step=5000, key="mf_lim",
                                      help="Protezione contro mesh troppo grandi per la memoria disponibile. "
                                           "Sul cloud gratuito (~1 GB) restare sotto ~35.000.")

    # ---- 2) composizione e conducibilita' ----
    with st.expander("2. Composizione della purea e σ(T)", expanded=True):
        id_def = 1 if cella.startswith('UCD') else 3
        opz = [ETICHETTE_COMP[i] for i in range(1, 7)] + ['Personalizzata']
        sel_c = st.selectbox("Composizione (Tabella 4.3)", opz, index=id_def - 1, key=f"mf_comp_{cella}")
        if sel_c == 'Personalizzata':
            b_c = st.number_input("Burro [% w/w]", value=0.0, min_value=0.0, key="mf_cb")
            s_c = st.number_input("Sale [% w/w]", value=2.75, min_value=0.0, key="mf_cs")
            f_c = st.number_input("Fiocchi di patate [% w/w]", value=12.68, min_value=0.0, key="mf_cf")
            w_c = st.number_input("Acqua [% w/w]", value=84.57, min_value=0.0, key="mf_cw")
            sigma_info = None
        else:
            id_comp = opz.index(sel_c) + 1
            cc = me.COMPOSIZIONI[id_comp]
            b_c, s_c, f_c, w_c = cc['burro'], cc['sale'], cc['fiocchi'], cc['acqua']
            sigma_info = cc['sigma']
            st.caption(f"Burro {b_c:.2f} %, sale {s_c:.2f} %, fiocchi {f_c:.2f} %, acqua {w_c:.2f} %")
        if sigma_info is None:
            st.warning("Regressione σ(T) non disponibile per questa composizione: sono mostrati, come segnaposto, "
                       "i coefficienti della composizione 3. Sostituiscili con quelli misurati.")
            sig_ref = me.COMPOSIZIONI[3]['sigma']
            fonte_sig = "segnaposto (comp. 3)"
        else:
            sig_ref = sigma_info
            fonte_sig = f"{sig_ref[3]}, R² = {sig_ref[2]:.4f}"
        st.caption(f"Purea: σ = a·T + b  ({fonte_sig})")
        suf_sig = sel_c
        m_a = st.number_input("Purea: a [S/m/°C]", value=float(sig_ref[0]), format="%.4f", key=f"mf_ma_{suf_sig}")
        m_b = st.number_input("Purea: b [S/m]", value=float(sig_ref[1]), format="%.4f", key=f"mf_mb_{suf_sig}")
        st.caption(f"Polpette: σ = a·T + b  ({me.SIGMA_POLPETTE[3]}, R² = {me.SIGMA_POLPETTE[2]:.4f})")
        p_a = st.number_input("Polpette: a [S/m/°C]", value=float(me.SIGMA_POLPETTE[0]), format="%.4f", key="mf_pa")
        p_b = st.number_input("Polpette: b [S/m]", value=float(me.SIGMA_POLPETTE[1]), format="%.4f", key="mf_pb")

    # ---- 3) proprieta' termofisiche ----
    with st.expander("3. Proprietà termofisiche (Choi–Okos)"):
        modo_lbl = {'Dipendenti da T, da composizione (consigliato)': 'choi_okos_T',
                    'Costanti, da composizione a T di riferimento': 'choi_okos_Tref',
                    'Valori manuali': 'manuali'}
        modo_sel = st.selectbox("Modello", list(modo_lbl.keys()), key="mf_modo")
        prop_modo = modo_lbl[modo_sel]
        T_rif = 50.0
        if prop_modo == 'choi_okos_Tref':
            T_rif = st.number_input("T di riferimento [°C]", value=50.0, key="mf_Trif")
        man = {}
        if prop_modo == 'manuali':
            man['k_matrice'] = st.number_input("Purea: k [W/m/K]", value=0.52, key="mf_km")
            man['rho_matrice'] = st.number_input("Purea: ρ [kg/m³]", value=1060.0, key="mf_rm")
            man['cp_matrice'] = st.number_input("Purea: cp [J/kg/K]", value=3500.0, key="mf_cm")
            man['k_polpetta'] = st.number_input("Polpette: k [W/m/K]", value=0.48, key="mf_kp")
            man['rho_polpetta'] = st.number_input("Polpette: ρ [kg/m³]", value=1050.0, key="mf_rp")
            man['cp_polpetta'] = st.number_input("Polpette: cp [J/kg/K]", value=3400.0, key="mf_cp")

    with st.expander("3b. Composizione dei componenti (USDA)"):
        st.caption("g/100 g, USDA FoodData Central (SR Legacy). Modificabile; la somma viene normalizzata a 100.")
        comp_us = {}
        for nome_c, tit in (('fiocchi', 'Fiocchi di patate disidratati'), ('burro', 'Burro non salato'),
                            ('pollo', 'Pollo (polpette)')):
            st.markdown(f"**{tit}**")
            comp_us[nome_c] = {}
            colonne = st.columns(2)
            for n_c, cmp_ in enumerate(me.NOMI_COMPONENTI):
                comp_us[nome_c][cmp_] = colonne[n_c % 2].number_input(
                    cmp_.capitalize(), value=float(me.COMPONENTI_USDA[nome_c][cmp_]),
                    min_value=0.0, format="%.2f", key=f"mf_us_{nome_c}_{cmp_}")

    # ---- 4) elettrico ----
    with st.expander("4. Tensione applicata", expanded=True):
        modo_el = st.selectbox("Imposta tramite", ["Campo elettrico efficace [V/cm]", "Tensione efficace [V]",
                                                    "Tensione di picco [V]"], key="mf_modoel")
        Lx_cm = (override['Lx'] if override else me.CELLE[cella]['Lx']) * 100
        if modo_el.startswith("Campo"):
            E_in = st.number_input("Campo elettrico efficace [V/cm]", value=4.0, min_value=0.01, key="mf_E")
            V_rms = E_in * Lx_cm
        elif modo_el.startswith("Tensione efficace"):
            V_in = st.number_input("Tensione efficace [V]", value=40.0, min_value=0.1, key="mf_Vr")
            V_rms = V_in
        else:
            V_in = st.number_input("Tensione di picco [V]", value=56.6, min_value=0.1, key="mf_Vp")
            V_rms = V_in / np.sqrt(2)
        st.caption(f"= {V_rms:.1f} V rms = {V_rms * np.sqrt(2):.1f} V di picco = {V_rms / Lx_cm:.2f} V/cm (su {Lx_cm:.1f} cm)")

    # ---- 5) termico ----
    with st.expander("5. Termico e scambio superficiale", expanded=True):
        T0 = st.number_input("T iniziale [°C]", value=20.0, key="mf_T0")
        T_a = st.number_input("T ambiente [°C]", value=20.0, key="mf_Ta")
        h_lbl = {'h(T): convezione naturale + irraggiamento (consigliato)': 'naturale_irraggiamento',
                 'h(T): + evaporazione (superficie libera umida)': 'naturale_irraggiamento_evaporazione',
                 'h costante': 'costante'}
        h_sel = st.selectbox("Scambio sulla superficie libera", list(h_lbl.keys()), key="mf_hmodo")
        h_modo = h_lbl[h_sel]
        h_c, emis, UR, aw, f_h = 10.0, 0.95, 0.5, 0.98, 1.0
        if h_modo == 'costante':
            h_c = st.number_input("h [W/m²/K]", value=10.0, key="mf_hc")
        else:
            emis = st.number_input("Emissività della superficie", value=0.95, min_value=0.0, max_value=1.0, key="mf_em")
            f_h = st.number_input("Fattore di copertura (1 = libera, 0 = adiabatica)", value=1.0, min_value=0.0, max_value=1.0, key="mf_fh",
                                  help="Riduce lo scambio se la cella ha un coperchio o la superficie è protetta.")
            if h_modo == 'naturale_irraggiamento_evaporazione':
                UR = st.number_input("Umidità relativa ambiente", value=0.5, min_value=0.0, max_value=1.0, key="mf_ur")
                aw = st.number_input("Attività dell'acqua della superficie", value=0.98, min_value=0.5, max_value=1.0, key="mf_aw")
        durata = st.number_input("Durata [s]", value=180.0, min_value=3.0, key="mf_dur")
        dt = st.number_input("Passo temporale DT [s]", value=3.0, min_value=0.1, key="mf_dt")

# --- geometria e parametri ---
try:
    geo = me.geometria(cella, conf, override)
except ValueError as e:
    st.error(f"Geometria non valida: {e}")
    st.stop()

fiocchi_u, burro_u, pollo_u = comp_us['fiocchi'], comp_us['burro'], comp_us['pollo']
comp_matrice = me.composizione_purea(b_c, s_c, f_c, w_c, fiocchi=fiocchi_u, burro=burro_u)
comp_polpetta = me.normalizza(pollo_u)

p = dict(
    matrice_a=m_a, matrice_b=m_b, polpetta_a=p_a, polpetta_b=p_b,
    prop_modo=prop_modo, T_rif_prop=T_rif, comp_matrice=comp_matrice, comp_polpetta=comp_polpetta,
    k_matrice=man.get('k_matrice', 0.52), rho_matrice=man.get('rho_matrice', 1060.0), cp_matrice=man.get('cp_matrice', 3500.0),
    k_polpetta=man.get('k_polpetta', 0.48), rho_polpetta=man.get('rho_polpetta', 1050.0), cp_polpetta=man.get('cp_polpetta', 3400.0),
    V_rms=V_rms, T_iniziale=T0, T_amb=T_a, h_modo=h_modo, h_conv=h_c, emissivita=emis, UR=UR, aw=aw, fattore_h=f_h,
    dt=dt, durata=durata, h_mesh_mm=h_mm,
)

for a in geo['avvisi']:
    st.warning(a)

# =============================================================================
# Composizione e proprieta' usate (sempre visibili, prima del calcolo)
# =============================================================================
with st.expander("Proprietà elettriche usate nel modello: σ(T)"):
    st.markdown("La conducibilità elettrica di ciascuna fase è una funzione lineare della temperatura; nei punti di transizione "
                "tra le fasi si usa la media pesata con la frazione di fase.")
    for r_ in me.LATEX_SIGMA:
        st.latex(r_)
    st.caption(me.LEGENDA_SIGMA)
    fonte_m = (f"{sigma_info[3]}, $R^2$ = {sigma_info[2]:.4f}" if sigma_info is not None else "valori inseriti (segnaposto)")
    st.markdown(me.md_sigma([("Purea", m_a, m_b, fonte_m),
                             ("Polpette", p_a, p_b, f"{me.SIGMA_POLPETTE[3]}, $R^2$ = {me.SIGMA_POLPETTE[2]:.4f}")]))
    st.pyplot(fig_sigma(p), width=LARG_2D + 100)

with st.expander("Proprietà termofisiche usate nel modello (Choi–Okos)"):
    righe = []
    for nome, comp in (('Purea', comp_matrice), ('Polpette', comp_polpetta)):
        righe.append({'Fase': nome, **{c.capitalize(): round(comp[c] * 100, 2) for c in me.NOMI_COMPONENTI}})
    st.write("Frazioni di massa $x_i$ [% w/w] delle due fasi (purea: dalla ricetta della Tabella 4.3 e dai componenti USDA; il sale è contato tra le ceneri):")
    st.dataframe(pd.DataFrame(righe), hide_index=True)
    if prop_modo == 'manuali':
        st.info("Modalità **valori manuali**: $k$, $\\rho$ e $c_p$ di ciascuna fase sono quelli inseriti nella barra laterale, costanti nel tempo.")
    else:
        if prop_modo == 'choi_okos_Tref':
            st.info(f"Le relazioni sono valutate una sola volta alla temperatura di riferimento ({p['T_rif_prop']:g} °C) e poi mantenute costanti.")
        else:
            st.write("Le proprietà dipendono dalla temperatura attraverso queste relazioni, valutate a ogni passo:")
        for r_ in me.LATEX_CHOI_OKOS:
            st.latex(r_)
        st.caption(me.LEGENDA_CHOI_OKOS)
        st.markdown(me.md_choi_okos())
        st.caption(f"$k$ in W/(m K), $\\rho$ in kg/m³, $c_p$ in kJ/(kg K). Fonti: " + me.FONTI['choi_okos'] + "; " + me.FONTI['miscelazione'] + "; " + me.FONTI['usda'] +
                   ". Gli additivi dei fiocchi (~1%) sono trascurati; l'effetto della dissoluzione del sale sulla densità è approssimato.")
        st.pyplot(fig_proprieta(p), width=1100)

with st.expander("Scambio sulla superficie libera: relazioni e parametri"):
    if p['h_modo'] == 'costante':
        st.write("Modalità **$h$ costante**: il flusso sulla superficie libera è")
        st.latex(r"q_s = f_c\,h\,(T_s-T_{amb})")
        st.caption("$f_c$ fattore di copertura (1 = superficie libera, 0 = adiabatica); $h$ coefficiente di scambio costante; "
                   "$T_s$ temperatura superficiale; $T_{amb}$ temperatura dell'aria.")
    else:
        st.write("Il flusso sulla superficie libera, moltiplicato per il fattore di copertura $f_c$, è la somma di questi contributi:")
        for r_ in (me.LATEX_SCAMBIO if p['h_modo'] == 'naturale_irraggiamento_evaporazione' else me.LATEX_SCAMBIO[:4]):
            st.latex(r_)
        st.caption(me.LEGENDA_SCAMBIO)
    par = {"Modalità": {'costante': 'h costante', 'naturale_irraggiamento': 'convezione naturale + irraggiamento',
                        'naturale_irraggiamento_evaporazione': 'convezione naturale + irraggiamento + evaporazione'}.get(p['h_modo'], p['h_modo']),
           "Temperatura dell'aria [°C]": p['T_amb'], "Fattore di copertura": p.get('fattore_h', 1.0)}
    if p['h_modo'] == 'costante':
        par["h costante [W/(m² K)]"] = p['h_conv']
    else:
        par["Emissività"] = p['emissivita']
        if p['h_modo'] == 'naturale_irraggiamento_evaporazione':
            par["Umidità relativa dell'aria"] = p['UR']; par["Attività dell'acqua"] = p['aw']
    st.write("Parametri usati:")
    st.dataframe(pd.DataFrame({'Parametro': list(par.keys()), 'Valore': [str(v) for v in par.values()]}), hide_index=True)
    st.pyplot(fig_h(p, geo), width=LARG_2D + 100)
    st.caption("Il flusso non lineare è linearizzato con il metodo di Newton a ogni passo temporale.")

# =============================================================================
# Passo 1 - geometria e mesh
# =============================================================================
n_est, (nx_, ny_, nz_) = me.stima_nodi(geo, h_mm / 1000.0)
passi_est = max(1, int(round(durata / dt)))
ram_est = n_est * KB_NODO
t_est = n_est * S_NODO_PASSO * passi_est
gap = me.gap_minimo(geo)

st.subheader("Stima del costo di calcolo")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Nodi stimati", f"{n_est:,}")
m2.metric("Memoria stimata", f"{ram_est:,.0f} MB")
m3.metric("Tempo stimato", f"{t_est / 60:.1f} min" if t_est >= 90 else f"{t_est:.0f} s")
if gap is not None:
    m4.metric("Elementi nello strato di purea più sottile", f"{gap / (h_mm / 1000):.1f}",
              help=f"Strato più sottile: {gap * 1000:.1f} mm. Sotto ~2 elementi il passaggio di corrente "
                   "nei gap è poco risolto.")
too_big = n_est > limite_nodi
if too_big:
    st.error(f"La mesh richiesta ha ~{n_est:,} nodi, oltre il limite impostato ({limite_nodi:,}). "
             "Aumenta il passo di mesh oppure il limite (solo se hai memoria sufficiente).")
elif n_est > 35000:
    st.warning("Oltre ~35.000 nodi la memoria richiesta supera quella del piano gratuito di Streamlit Cloud (~1 GB): "
               "usa l'app in locale.")
if gap is not None and gap / (h_mm / 1000) < 2:
    st.warning(f"Lo strato di purea più sottile ({gap * 1000:.1f} mm) è risolto da meno di 2 elementi: "
               "i risultati nei gap sono qualitativi. Riduci il passo di mesh.")

c1, c2 = st.columns(2)
with c1:
    click_mesh = st.button("1) Costruisci geometria e mesh", type="primary", width='stretch', disabled=too_big)
with c2:
    click_sim = st.button("2) Avvia simulazione", type="primary", width='stretch',
                          disabled=('mesh' not in S))

if click_mesh:
    mesh, basis, h_eff = me.costruisci_mesh(geo, h_mm / 1000.0)
    S.clear()
    S.update(geo=geo, mesh=mesh, basis=basis, h_eff=h_eff, chiave=(cella, conf, h_mm, repr(override)))
    S['phi_n'] = me.frazione_fase(basis.doflocs, geo['centri'], geo['R'], 0.5 * h_eff)
    st.rerun()      # aggiorna subito lo stato del bottone 2 (evita il doppio click)

if 'mesh' in S:
    st.subheader("Geometria e mesh")
    if S['chiave'] != (cella, conf, h_mm, repr(override)):
        st.warning("Hai cambiato cella, configurazione, geometria o passo di mesh dopo averla costruita: "
                   "premi di nuovo '1) Costruisci geometria e mesh'.")
    g1, g2 = st.columns([3, 2])
    with g1:
        st.pyplot(fig_geometria(S['geo']), width=LARG_SEZ)
    with g2:
        st.metric("Nodi", f"{S['basis'].N:,}")
        st.metric("Tetraedri", f"{S['mesh'].t.shape[1]:,}")
        st.metric("Passo effettivo di mesh", f"{S['h_eff'] * 1000:.2f} mm")
        if S['geo']['centri']:
            st.metric("Elementi lungo il diametro della polpetta", f"{S['geo']['D'] / S['h_eff']:.1f}")
        st.caption(f"Griglia: {nx_} x {ny_} x {nz_} nodi; ogni cella è divisa in 6 tetraedri.")
    st.pyplot(fig_mesh_sezioni(S['mesh'], S['geo'], S['phi_n'], S['basis'].doflocs), width=1500)
    with st.expander("Superficie esterna della mesh (3D)"):
        st.pyplot(fig_mesh_3d(S['mesh'], S['geo']), width=LARG_SEZ)

# =============================================================================
# Passo 2 - simulazione
# =============================================================================
if click_sim:
    barra = st.progress(0.0, text="Simulazione in corso...")
    try:
        ris = me.simula(S['geo'], S['mesh'], S['basis'], S['h_eff'], p,
                        progress_callback=lambda f: barra.progress(f, text=f"Simulazione in corso... {f * 100:.0f}%"))
    except RuntimeError as e:
        barra.empty()
        st.error(f"Errore di convergenza: {e}")
        st.stop()
    barra.empty()
    S['ris'] = ris
    S.pop('gif', None)

# =============================================================================
# Risultati
# =============================================================================
if 'ris' in S:
    ris = S['ris']
    Tmax = ris['T_hist'].max()
    st.success(f"Simulazione completata: T_max = {Tmax:.1f} °C, potenza finale = {ris['P'][-1]:.0f} W.")
    if Tmax > 100:
        st.warning("T > 100 °C: fuori dal range di validità del modello (nessuna ebollizione, "
                   "regressione σ(T) estrapolata oltre i dati sperimentali).")

    met = me.metriche_cold_spot(ris)
    ha_polpette = met['T_polpette_media'] is not None
    tab_s, tab_z, tab_c, tab_p, tab_h, tab_a = st.tabs(
        ["Sonde P1-P3", "Sezioni T / σ / q", "Cold spot", "Potenza e corrente", "Scambio superficiale", "Animazione"])

    with tab_s:
        st.pyplot(fig_sonde(ris), width=LARG_2D)
        csv = io.StringIO()
        csv.write("tempo_s," + ",".join(ris['T_sonde'].keys()) + ",P_W\n")
        for i, t in enumerate(ris['tempo']):
            csv.write(f"{t:.1f}," + ",".join(f"{ris['T_sonde'][n][i]:.3f}" for n in ris['T_sonde']) + f",{ris['P'][i]:.2f}\n")
        st.download_button("Scarica sonde e potenza (CSV)", csv.getvalue(), "multifase_sonde.csv", "text/csv")

    with tab_z:
        x_u, y_u, z_u, _ = griglia(ris['coords'])
        k_t = st.slider("Istante", 0, len(ris['tempo']) - 1, len(ris['tempo']) - 1, key="mf_kt")
        j_y = st.slider("Sezione xz: indice y", 0, len(y_u) - 1, indice_piu_vicino(y_u, ris['geo']['Ly'] / 2), key="mf_jy")
        st.caption(f"t = {ris['tempo'][k_t]:.0f} s   |   y = {y_u[j_y] * 1000:.1f} mm")
        st.pyplot(fig_sezioni(ris, k_t, j_y), width=1500)

    with tab_c:
        if ha_polpette:
            cc1, cc2 = st.columns(2)
            with cc1:
                st.pyplot(fig_cold_spot(ris, met), width=LARG_2D)
            with cc2:
                st.pyplot(fig_scarto(ris, met), width=LARG_2D)
            d = met['T_matrice_media'] - met['T_polpette_media']
            st.write(f"Scarto massimo purea-polpette: **{d.max():.1f} °C** a t = {ris['tempo'][int(np.argmax(d))]:.0f} s. "
                     f"Scarto finale: **{d[-1]:.1f} °C**.")
            n_t = len(ris['tempo'])
            idx_def = sorted(set([n_t // 4, n_t // 2, 3 * n_t // 4, n_t - 1]))
            idx = st.multiselect("Istanti per il profilo lungo x", list(range(n_t)), default=idx_def,
                                 format_func=lambda i: f"{ris['tempo'][i]:.0f} s", key="mf_idx")
            if idx:
                st.pyplot(fig_profilo_x(ris, idx), width=LARG_SEZ)
        else:
            st.info("La configurazione scelta non ha polpette: nessun cold spot da analizzare.")

    with tab_p:
        st.pyplot(fig_potenza(ris), width=1000)
        st.caption("P(t) e I(t) sono direttamente confrontabili con la potenza/corrente misurate dal sistema di acquisizione.")

    with tab_h:
        st.pyplot(fig_scambio(ris), width=LARG_2D)

    with tab_a:
        x_u, y_u, z_u, _ = griglia(ris['coords'])
        j_a = indice_piu_vicino(y_u, ris['geo']['Ly'] / 2)
        st.caption("Sezione xz sulla mezzeria (y centrale), che taglia le polpette nel loro piano equatoriale.")
        if st.button("Genera animazione", key="mf_gif"):
            with st.spinner("Generazione animazione..."):
                S['gif'] = genera_gif(ris, j_a)
        if S.get('gif'):
            # width='content': un width numerico farebbe ridimensionare la GIF da Streamlit
            # perdendo tutti i frame tranne il primo.
            st.image(S['gif'], width='content')

with st.expander("Ipotesi e limiti del modello (da leggere)"):
    st.markdown("""
- **Elettrodi** a faccia piena alle due estremità lungo x (confermato), direzione delle sonde P1-P3.
- **σ(T)**: regressioni lineari delle Tabelle 5.1 (composizioni 1-2 e polpette) e 6.1 (composizioni 3-6). Per le composizioni 7-9 non esiste regressione: va inserita.
- **Proprietà termofisiche**: Choi–Okos da composizione e temperatura; la composizione dei fiocchi, del burro e del pollo è quella USDA (SR Legacy), modificabile. La ricetta reale delle polpette non è nota.
- **Interfaccia purea/polpetta**: fase diffusa su un passo di mesh dimezzato (mesh strutturata, non conforme alla sfera). Validazione contro la soluzione analitica sfera/mezzo: errore sul campo interno ≈ +3% a R/h = 4, decrescente con la mesh.
- **Sinusoide a 50 Hz**: integrata in modo esatto sfruttando la linearità del problema elettrico (una risoluzione per passo termico).
- **Scambio**: solo sulla superficie libera superiore, h(T) con linearizzazione di Newton; pareti laterali e fondo adiabatici. Nessuna ebollizione, evaporazione interna o fluidodinamica.
- **Solutori**: elettrico con gradiente coniugato e preconditionatore algebrico multigrid (pyamg) oltre 3000 nodi; termico con gradiente coniugato. Validati contro il solutore diretto (differenze < 10⁻⁸ °C) e col bilancio energetico (chiude a 0.02%).
""")
