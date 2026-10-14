"""
Pagina: Modello base per il riscaldamento ohmico (cella omogenea, modello elettro-termico 3D accoppiato).
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import streamlit as st
from skfem import *
from skfem.helpers import dot, grad
import os
import tempfile

CARTELLA_OUTPUT = tempfile.gettempdir() + '/'

# Larghezza fissa (in pixel) per tutti i grafici, cosi' non dipendono dalla
# larghezza del contenitore (Streamlit di default fa width='stretch', che
# ingrandisce i grafici a tutta la larghezza pagina -- specialmente evidente
# sui grafici fuori da una colonna, come la mesh).
LARGHEZZA_2D = 480   # grafici 2D (linee, sezioni, GIF)
LARGHEZZA_3D = 560   # grafici 3D (mesh, scatter completo)


# =============================================================================
# FUNZIONI DEL MODELLO (identiche nella logica alla versione Colab; le
# funzioni di plotting restituiscono la Figure invece di chiamare plt.show(),
# perche' in Streamlit e' st.pyplot(fig) a disegnarla)
# =============================================================================

def costruisci_mesh(p):
    if p['infittisci_elettrodi']:
        spess_el = p['spessore_elettrodi']
        if 2 * spess_el >= p['Lx']:
            raise ValueError("spessore_elettrodi troppo grande rispetto a Lx: le due zone infittite si sovrappongono.")
        x_nodi = np.concatenate([
            np.linspace(0, spess_el, p['n_nodi_elettrodi']),
            np.linspace(spess_el, p['Lx'] - spess_el, p['n_nodi_x_corpo'])[1:],
            np.linspace(p['Lx'] - spess_el, p['Lx'], p['n_nodi_elettrodi'])[1:],
        ])
    else:
        x_nodi = np.linspace(0, p['Lx'], p['n_nodi_x_corpo'])

    if p['infittisci_superficie']:
        if p['spessore_strato_limite'] >= p['Lz']:
            raise ValueError("spessore_strato_limite troppo grande rispetto a Lz.")
        z_nodi = np.concatenate([
            np.linspace(0, p['Lz'] - p['spessore_strato_limite'], p['n_nodi_z_corpo']),
            np.linspace(p['Lz'] - p['spessore_strato_limite'], p['Lz'], p['n_nodi_z_strato'])[1:],
        ])
    else:
        z_nodi = np.linspace(0, p['Lz'], p['n_nodi_z_corpo'])

    y_nodi = np.linspace(0, p['Ly'], p['n_nodi_y'])

    if p['tipo_mesh'] == 'tetraedrica':
        mesh = MeshTet.init_tensor(x_nodi, y_nodi, z_nodi)
        basis = Basis(mesh, ElementTetP1())
    else:
        mesh = MeshHex.init_tensor(x_nodi, y_nodi, z_nodi)
        basis = Basis(mesh, ElementHex1())
    return mesh, basis


def visualizza_mesh(mesh, basis, p):
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    coords_mm = mesh.p.T * 1000
    edges = mesh.edges.T
    soglia = 6000
    if len(edges) > soglia:
        idx = np.random.RandomState(0).choice(len(edges), soglia, replace=False)
        edges_da_disegnare = edges[idx]
        sottocampionato = True
    else:
        edges_da_disegnare = edges
        sottocampionato = False
    segmenti = coords_mm[edges_da_disegnare]

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.add_collection3d(Line3DCollection(segmenti, colors='tab:blue', linewidths=0.4, alpha=0.6))
    ax.scatter(coords_mm[:,0], coords_mm[:,1], coords_mm[:,2], s=6, color='tab:red', depthshade=False)
    ax.set_xlim(coords_mm[:,0].min(), coords_mm[:,0].max())
    ax.set_ylim(coords_mm[:,1].min(), coords_mm[:,1].max())
    ax.set_zlim(coords_mm[:,2].min(), coords_mm[:,2].max())
    ax.set_xlabel('x [mm]'); ax.set_ylabel('y [mm]'); ax.set_zlabel('z [mm]')
    nome_mesh = "tetraedrica" if p['tipo_mesh'] == 'tetraedrica' else "esaedrica"
    nome_elemento = "tetraedri" if p['tipo_mesh'] == 'tetraedrica' else "esaedri"
    titolo = f"Mesh {nome_mesh}: {basis.N} nodi, {mesh.t.shape[1]} {nome_elemento}"
    if sottocampionato:
        titolo += f" (disegnati {soglia} spigoli su {len(edges)})"
    ax.set_title(titolo)
    plt.tight_layout()

    info = [f"Nodi totali: {basis.N}", f"Elementi totali ({nome_elemento}): {mesh.t.shape[1]}"]
    if p['infittisci_elettrodi']:
        info.append(f"Infittimento elettrodi: {p['spessore_elettrodi']*1000:.1f}mm, {p['n_nodi_elettrodi']} nodi/lato")
    else:
        info.append("Infittimento elettrodi: disattivato")
    if p['infittisci_superficie']:
        info.append(f"Infittimento superficie: {p['spessore_strato_limite']*1000:.1f}mm, {p['n_nodi_z_strato']} nodi")
    else:
        info.append("Infittimento superficie: disattivato")
    return fig, info


def esegui_simulazione(mesh, basis, p, progress_callback=None):
    coords = basis.doflocs

    def sigma_di_T(T_nodale):
        return p['sigma0'] * (1.0 + p['alpha_sigma'] * (T_nodale - p['T0_rif']))

    facce_convezione = mesh.facets_satisfying(lambda x: np.isclose(x[2], p['Lz']))
    fbasis = FacetBasis(mesh, basis.elem, facets=facce_convezione)

    @BilinearForm
    def robin_bilineare(T, w_test, w):
        return p['h_conv'] * T * w_test

    @LinearForm
    def robin_lineare(w_test, w):
        return p['h_conv'] * p['T_amb'] * w_test

    K_robin = robin_bilineare.assemble(fbasis)
    b_robin = robin_lineare.assemble(fbasis)

    elettrodo_sx = basis.get_dofs(lambda x: np.isclose(x[0], 0.0))
    elettrodo_dx = basis.get_dofs(lambda x: np.isclose(x[0], p['Lx']))
    D_elettrico = elettrodo_sx + elettrodo_dx

    @BilinearForm
    def laplace_elettrico(V, w_test, w):
        return w['sigma_qp'] * dot(grad(V), grad(w_test))

    @LinearForm
    def sorgente_joule(w_test, w):
        return w['sigma_qp'] * dot(w['gradV'], w['gradV']) * w_test

    @BilinearForm
    def massa(T, w_test, w):
        return p['rho'] * p['cp'] * T * w_test

    @BilinearForm
    def rigidezza_termica(T, w_test, w):
        return p['k_termica'] * dot(grad(T), grad(w_test))

    M = massa.assemble(basis)
    K = rigidezza_termica.assemble(basis)
    A_tempo = M / p['dt'] + K + K_robin

    T = np.full(basis.N, p['T_iniziale'])

    def nodo_piu_vicino(x_t, y_t, z_t):
        d = (coords[0]-x_t)**2 + (coords[1]-y_t)**2 + (coords[2]-z_t)**2
        return np.argmin(d)

    sonde = {nome: nodo_piu_vicino(*pos) for nome, pos in p['posizioni_sonde'].items()}
    storico_T = {nome: [] for nome in sonde}
    storico_tempo, T_storia_completa = [], []
    snapshot_T, snapshot_sigma, snapshot_V = {}, {}, {}

    istanti_periodo = np.linspace(0, 1/p['frequenza'], p['n_sub_cicli'], endpoint=False)

    for passo in range(p['n_passi']):
        sigma_nodale = sigma_di_T(T)
        sigma_interp = basis.interpolate(sigma_nodale)
        A_el = laplace_elettrico.assemble(basis, sigma_qp=sigma_interp.value)
        b_el = basis.zeros()

        b_joule_acc = basis.zeros()
        for t_sub in istanti_periodo:
            V_ist = p['V0_picco'] * np.sin(2*np.pi*p['frequenza']*t_sub)
            V = basis.zeros()
            V[elettrodo_sx.all()] = V_ist
            V[elettrodo_dx.all()] = 0.0
            V = solve(*condense(A_el, b_el, x=V, D=D_elettrico))
            V_interp = basis.interpolate(V)
            b_joule_acc += sorgente_joule.assemble(basis, gradV=V_interp.grad, sigma_qp=sigma_interp.value)
        b_joule = b_joule_acc / p['n_sub_cicli']

        rhs = M @ T / p['dt'] + b_joule + b_robin
        T = solve(A_tempo, rhs)

        storico_tempo.append((passo+1)*p['dt'])
        T_storia_completa.append(T.copy())
        for nome, idx in sonde.items():
            storico_T[nome].append(T[idx])

        t_corr = (passo+1)*p['dt']
        for t_target in p['istanti_snapshot']:
            if abs(t_corr - t_target) < p['dt']/2 and t_target not in snapshot_T:
                snapshot_T[t_target] = T.copy()
                snapshot_sigma[t_target] = sigma_nodale.copy()
                V_picco = basis.zeros()
                V_picco[elettrodo_sx.all()] = p['V0_picco']
                V_picco[elettrodo_dx.all()] = 0.0
                snapshot_V[t_target] = solve(*condense(A_el, b_el, x=V_picco, D=D_elettrico))

        if progress_callback is not None:
            progress_callback((passo+1) / p['n_passi'])

    posizioni_sonde_mm = {nome: tuple(v*1000 for v in xyz) for nome, xyz in p['posizioni_sonde'].items()}

    return dict(
        T=T, coords=coords, sonde=sonde, storico_T=storico_T, storico_tempo=storico_tempo,
        T_storia_completa=T_storia_completa, snapshot_T=snapshot_T,
        snapshot_sigma=snapshot_sigma, snapshot_V=snapshot_V,
        posizioni_sonde_mm=posizioni_sonde_mm,
    )


def a_griglia_builder(coords):
    x_u, y_u, z_u = np.unique(coords[0]), np.unique(coords[1]), np.unique(coords[2])
    nx, ny, nz = len(x_u), len(y_u), len(z_u)
    ix = np.searchsorted(x_u, coords[0]); iy = np.searchsorted(y_u, coords[1]); iz = np.searchsorted(z_u, coords[2])
    def a_griglia(campo):
        G = np.empty((nx, ny, nz)); G[ix, iy, iz] = campo; return G
    return a_griglia, x_u, y_u, z_u, nx, ny, nz


def _nearest_idx(arr, val):
    return int(np.argmin(np.abs(arr - val)))


# --- Figure (ognuna restituisce una Figure, non chiama plt.show()) ---

def fig_profilo_termico_z(risultati, viz):
    _, x_u, y_u, z_u, nx, ny, nz = a_griglia_builder(risultati['coords'])
    tempi = viz['istanti'] if viz['istanti'] else sorted(risultati['snapshot_T'].keys())
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for t in tempi:
        if t not in risultati['snapshot_T']:
            continue
        T_snap = risultati['snapshot_T'][t]
        T_media = [T_snap[np.isclose(risultati['coords'][2], zv)].mean() for zv in z_u]
        ax.plot(z_u*1000, T_media, 'o-', markersize=3, label=f't={t:.0f}s')
    ax.set_xlabel('z [mm]'); ax.set_ylabel('T [°C]')
    ax.set_title('Profilo termico lungo z')
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    plt.tight_layout()
    return fig


def fig_sonde(risultati):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for nome, valori in risultati['storico_T'].items():
        ax.plot(risultati['storico_tempo'], valori, label=nome)
    ax.set_xlabel('tempo [s]'); ax.set_ylabel('T [°C]')
    ax.set_title('Temperatura nei punti sonda')
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    plt.tight_layout()
    return fig


def fig_sezione_xz(risultati, viz):
    a_griglia, x_u, y_u, z_u, nx, ny, nz = a_griglia_builder(risultati['coords'])
    T_grid = a_griglia(risultati['T'])
    j = viz['idx_y']
    vmin, vmax = risultati['T'].min(), risultati['T'].max()
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    c = ax.pcolormesh(x_u*1000, z_u*1000, T_grid[:, j, :].T, cmap='inferno', shading='auto', vmin=vmin, vmax=vmax)
    fig.colorbar(c, ax=ax, label='T [°C]')
    ax.set_xlabel('x [mm]'); ax.set_ylabel('z [mm]')
    ax.set_title(f'Sezione xz (y = {y_u[j]*1000:.1f} mm)')
    plt.tight_layout()
    return fig


def fig_sezione_xy(risultati, viz):
    a_griglia, x_u, y_u, z_u, nx, ny, nz = a_griglia_builder(risultati['coords'])
    T_grid = a_griglia(risultati['T'])
    k = viz['idx_z']
    vmin, vmax = risultati['T'].min(), risultati['T'].max()
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    c = ax.pcolormesh(x_u*1000, y_u*1000, T_grid[:, :, k].T, cmap='inferno', shading='auto', vmin=vmin, vmax=vmax)
    fig.colorbar(c, ax=ax, label='T [°C]')
    ax.set_xlabel('x [mm]'); ax.set_ylabel('y [mm]')
    ax.set_title(f'Sezione xy (z = {z_u[k]*1000:.1f} mm)')
    ax.set_aspect('equal')
    plt.tight_layout()
    return fig


def fig_scatter3d(risultati):
    coords = risultati['coords']
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection='3d')
    pnt = ax.scatter(coords[0]*1000, coords[1]*1000, coords[2]*1000, c=risultati['T'], cmap='inferno', s=25)
    fig.colorbar(pnt, ax=ax, shrink=.6, label='T [°C]')
    ax.set_xlabel('x [mm]'); ax.set_ylabel('y [mm]'); ax.set_zlabel('z [mm]')
    ax.set_title('Campo 3D completo')
    plt.tight_layout()
    return fig


def fig_profilo_elettrodi(risultati, viz):
    a_griglia, x_u, y_u, z_u, nx, ny, nz = a_griglia_builder(risultati['coords'])
    j = _nearest_idx(y_u, viz['linea_y']); k = _nearest_idx(z_u, viz['linea_z'])
    tempi = viz['istanti'] if viz['istanti'] else sorted(risultati['snapshot_T'].keys())
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for t in tempi:
        if t not in risultati['snapshot_T']:
            continue
        T_grid = a_griglia(risultati['snapshot_T'][t])
        ax.plot(x_u*1000, T_grid[:, j, k], 'o-', markersize=3, label=f't={t:.0f}s')
    ax.set_xlabel('x [mm]  (0=elettrodo, Lx=elettrodo opposto)'); ax.set_ylabel('T [°C]')
    ax.set_title(f'Profilo T tra i due elettrodi (y={y_u[j]*1000:.1f}mm, z={z_u[k]*1000:.1f}mm)')
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    plt.tight_layout()
    return fig


def fig_conducibilita(risultati, viz):
    a_griglia, x_u, y_u, z_u, nx, ny, nz = a_griglia_builder(risultati['coords'])
    i = _nearest_idx(x_u, viz['linea_x']); j = _nearest_idx(y_u, viz['linea_y2'])
    tempi = viz['istanti'] if viz['istanti'] else sorted(risultati['snapshot_sigma'].keys())
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for t in tempi:
        if t not in risultati['snapshot_sigma']:
            continue
        sigma_grid = a_griglia(risultati['snapshot_sigma'][t])
        ax.plot(z_u*1000, sigma_grid[i, j, :], 'o-', markersize=3, label=f't={t:.0f}s')
    ax.set_xlabel('z [mm]  (0=fondo, Lz=superficie)'); ax.set_ylabel('sigma [S/m]')
    ax.set_title(f'Conducibilita\' (x={x_u[i]*1000:.1f}mm, y={y_u[j]*1000:.1f}mm)')
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    plt.tight_layout()
    return fig


def fig_potenziale(risultati, viz, p):
    a_griglia, x_u, y_u, z_u, nx, ny, nz = a_griglia_builder(risultati['coords'])
    j = _nearest_idx(y_u, viz['linea_y']); k = _nearest_idx(z_u, viz['linea_z'])
    tempi = viz['istanti'] if viz['istanti'] else sorted(risultati['snapshot_V'].keys())
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for t in tempi:
        if t not in risultati['snapshot_V']:
            continue
        V_grid = a_griglia(risultati['snapshot_V'][t])
        ax.plot(x_u*1000, V_grid[:, j, k], 'o-', markersize=3, label=f't={t:.0f}s')
    ax.plot(x_u*1000, p['V0_picco']*(1-x_u/p['Lx']), '--', color='gray', label='retta teorica')
    ax.set_xlabel('x [mm]'); ax.set_ylabel('V [V] (istante di picco)')
    ax.set_title(f'Potenziale, istante di picco (y={y_u[j]*1000:.1f}mm, z={z_u[k]*1000:.1f}mm)')
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    plt.tight_layout()
    return fig


def fig_sorgente_joule(risultati, viz, p):
    a_griglia, x_u, y_u, z_u, nx, ny, nz = a_griglia_builder(risultati['coords'])
    i = _nearest_idx(x_u, viz['linea_x']); j = _nearest_idx(y_u, viz['linea_y2'])
    tempi = viz['istanti'] if viz['istanti'] else sorted(risultati['snapshot_sigma'].keys())
    E_campo = (p['V0_picco']/np.sqrt(2)) / p['Lx']
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for t in tempi:
        if t not in risultati['snapshot_sigma']:
            continue
        sigma_grid = a_griglia(risultati['snapshot_sigma'][t])
        ax.plot(z_u*1000, sigma_grid[i, j, :]*E_campo**2, 'o-', markersize=3, label=f't={t:.0f}s')
    ax.set_xlabel('z [mm]'); ax.set_ylabel('q [W/m3]')
    ax.set_title(f'Sorgente Joule q(z) (x={x_u[i]*1000:.1f}mm, y={y_u[j]*1000:.1f}mm)')
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    plt.tight_layout()
    return fig


def genera_animazioni(risultati, viz):
    import uuid
    id_univoco = uuid.uuid4().hex[:8]  # nomi unici: evita conflitti tra sessioni/utenti concorrenti

    a_griglia, x_u, y_u, z_u, nx, ny, nz = a_griglia_builder(risultati['coords'])
    T_grid_storia = [a_griglia(Tf) for Tf in risultati['T_storia_completa']]
    vmin_a, vmax_a = risultati['T_storia_completa'][0].min(), risultati['T_storia_completa'][-1].max()
    j = viz['idx_y']
    coords = risultati['coords']

    fig_a, ax_a = plt.subplots(figsize=(5, 4.5))
    quad = ax_a.pcolormesh(x_u*1000, z_u*1000, T_grid_storia[0][:, j, :].T, cmap='inferno', shading='auto', vmin=vmin_a, vmax=vmax_a)
    fig_a.colorbar(quad, ax=ax_a, label='T [°C]')
    tit_a = ax_a.set_title(f"Sezione xz (y={y_u[j]*1000:.1f}mm), t={risultati['storico_tempo'][0]:.0f}s")
    def upd2d(i):
        quad.set_array(T_grid_storia[i][:, j, :].T.ravel())
        tit_a.set_text(f"Sezione xz (y={y_u[j]*1000:.1f}mm), t={risultati['storico_tempo'][i]:.0f}s")
        return quad, tit_a
    anim2d = FuncAnimation(fig_a, upd2d, frames=len(T_grid_storia), interval=80)
    path2d = CARTELLA_OUTPUT + f'animazione_2d_{id_univoco}.gif'
    anim2d.save(path2d, writer=PillowWriter(fps=12)); plt.close(fig_a)

    fig_b = plt.figure(figsize=(5.5, 5)); ax_b = fig_b.add_subplot(111, projection='3d')
    sc = ax_b.scatter(coords[0]*1000, coords[1]*1000, coords[2]*1000, c=risultati['T_storia_completa'][0], cmap='inferno', s=25, vmin=vmin_a, vmax=vmax_a)
    fig_b.colorbar(sc, ax=ax_b, shrink=.6, label='T [°C]')
    tit_b = ax_b.set_title(f"t={risultati['storico_tempo'][0]:.0f}s")
    def upd3d(i):
        sc.set_array(risultati['T_storia_completa'][i]); tit_b.set_text(f"t={risultati['storico_tempo'][i]:.0f}s"); return sc, tit_b
    anim3d = FuncAnimation(fig_b, upd3d, frames=len(T_grid_storia), interval=80)
    path3d = CARTELLA_OUTPUT + f'animazione_3d_{id_univoco}.gif'
    anim3d.save(path3d, writer=PillowWriter(fps=12)); plt.close(fig_b)

    return path2d, path3d


def parse_sonde(testo):
    sonde = {}
    for riga in testo.strip().split('\n'):
        riga = riga.strip()
        if not riga:
            continue
        nome, coords_str = riga.split(':')
        x, y, z = [float(v) for v in coords_str.split(',')]
        sonde[nome.strip()] = (x/1000, y/1000, z/1000)
    return sonde


# =============================================================================
# INTERFACCIA STREAMLIT
# =============================================================================

st.title("Modello base per il riscaldamento ohmico")

if 'stato' not in st.session_state:
    st.session_state.stato = {}
stato = st.session_state.stato

with st.sidebar:
    st.header("Parametri")

    with st.expander("Geometria e mesh", expanded=True):
        Lx_mm = st.number_input("Lx [mm]", value=100.0, key="Lx")
        Ly_mm = st.number_input("Ly [mm]", value=50.0, key="Ly")
        Lz_mm = st.number_input("Lz [mm]", value=50.0, key="Lz")
        tipo_mesh = st.radio("Tipo di mesh", ["tetraedrica", "esaedrica"],
                              format_func=str.capitalize, key="tipo_mesh")
        st.caption("Tetraedrica: verso geometrie reali/gmsh. Esaedrica: 6x meno elementi a parita' di nodi.")

        infittisci_elettrodi = st.checkbox("Infittisci mesh vicino agli elettrodi", value=True, key="infel")
        nx_corpo = st.slider("Nodi x (corpo)", 4, 30, 10, key="nxc")
        nx_elettrodi = st.slider("Nodi x (elettrodi)", 3, 20, 6, disabled=not infittisci_elettrodi, key="nxe")
        spessore_elettrodi = st.number_input("Spessore infittimento elettrodi [mm]", value=10.0,
                                              disabled=not infittisci_elettrodi, key="spel")
        ny = st.slider("Nodi lungo y", 4, 20, 7, key="ny")

        infittisci_superficie = st.checkbox("Infittisci mesh vicino alla superficie convettiva", value=True, key="infsup")
        nzc = st.slider("Nodi z (corpo)", 3, 15, 5, key="nzc")
        nzs = st.slider("Nodi z (strato limite)", 3, 20, 8, disabled=not infittisci_superficie, key="nzs")
        spessore_sl = st.number_input("Spessore strato limite [mm]", value=10.0,
                                       disabled=not infittisci_superficie, key="spsl")

    with st.expander("Proprieta' del prodotto"):
        k_termica = st.number_input("k termica [W/mK]", value=0.55, key="k")
        rho = st.number_input("rho [kg/m3]", value=1050.0, key="rho")
        cp = st.number_input("cp [J/kgK]", value=3900.0, key="cp")
        sigma0 = st.number_input("sigma0 [S/m]", value=0.8, key="s0")
        alpha_sigma = st.number_input("alpha sigma [1/K]", value=0.02, format="%.4f", key="al")
        T0_rif = st.number_input("T0 riferimento [°C]", value=20.0, key="T0")

    with st.expander("Convezione"):
        h_conv = st.number_input("h convezione [W/m2K]", value=15.0, key="hc")
        T_amb = st.number_input("T ambiente [°C]", value=20.0, key="Tamb")

    with st.expander("Elettrodi (AC)"):
        V0_picco = st.number_input("V0 picco [V]", value=220.0, key="V0")
        frequenza = st.number_input("Frequenza [Hz]", value=50.0, key="freq")
        n_sub = st.slider("Sub-cicli/periodo", 5, 50, 20, key="nsub")

    with st.expander("Tempo"):
        T_iniziale = st.number_input("T iniziale [°C]", value=20.0, key="Tini")
        durata = st.number_input("Durata simulazione [s]", value=120.0, key="dur")
        dt = st.number_input("DT [s]", value=2.0, key="dt")
        n_passi = max(1, round(durata / dt))
        st.caption(f"→ {n_passi} passi temporali ({n_passi*n_sub} risoluzioni elettriche)")

    with st.expander("Punti sonda", expanded=True):
        sonde_testo = st.text_area(
            "Punti sonda [mm], formato 'nome: x, y, z'",
            value="fondo: 50, 25, 5\ncentro: 50, 25, 25\nsuperficie: 50, 25, 45",
            key="sonde",
        )

p = dict(
    Lx=Lx_mm/1000, Ly=Ly_mm/1000, Lz=Lz_mm/1000,
    infittisci_elettrodi=infittisci_elettrodi, tipo_mesh=tipo_mesh,
    n_nodi_x_corpo=nx_corpo, n_nodi_elettrodi=nx_elettrodi, spessore_elettrodi=spessore_elettrodi/1000,
    infittisci_superficie=infittisci_superficie, n_nodi_y=ny,
    n_nodi_z_corpo=nzc, n_nodi_z_strato=nzs, spessore_strato_limite=spessore_sl/1000,
    k_termica=k_termica, rho=rho, cp=cp, sigma0=sigma0, alpha_sigma=alpha_sigma, T0_rif=T0_rif,
    h_conv=h_conv, T_amb=T_amb, V0_picco=V0_picco, frequenza=frequenza, n_sub_cicli=n_sub,
    T_iniziale=T_iniziale, dt=dt, n_passi=n_passi,
    istanti_snapshot=sorted(set(max(dt, round(dt*n_passi*f/dt)*dt) for f in (0.08,0.17,0.33,0.5,0.75,1.0))),
)

col1, col2 = st.columns(2)
with col1:
    click_mesh = st.button("1) Costruisci mesh", type="primary", use_container_width=True)
with col2:
    click_sim = st.button("2) Avvia simulazione", type="primary", use_container_width=True,
                           disabled=('mesh' not in stato))

if click_mesh:
    try:
        sonde_parsate = parse_sonde(sonde_testo)
        if not sonde_parsate:
            raise ValueError("Inserisci almeno un punto sonda.")
        mesh, basis = costruisci_mesh(p)
        stato['mesh'], stato['basis'], stato['p_mesh'] = mesh, basis, p
        stato.pop('risultati', None)
        fig, info = visualizza_mesh(mesh, basis, p)
        stato['mesh_fig'], stato['mesh_info'] = fig, info
        # Forza un rerun immediato: senza questo, il bottone "2) Avvia
        # simulazione" (disegnato PRIMA di questo blocco, nello stesso giro)
        # non "vedrebbe" stato['mesh'] appena scritto e resterebbe
        # disabilitato fino al click successivo.
        st.rerun()
    except ValueError as e:
        st.error(f"Errore: {e}")

if 'mesh_fig' in stato:
    st.subheader("Mesh")
    st.pyplot(stato['mesh_fig'], width=LARGHEZZA_3D)
    st.text("\n".join(stato['mesh_info']))

if click_sim:
    try:
        sonde_parsate = parse_sonde(sonde_testo)
        if not sonde_parsate:
            raise ValueError("Inserisci almeno un punto sonda.")
        p_sim = dict(p)
        p_sim['Lx'], p_sim['Ly'], p_sim['Lz'] = stato['p_mesh']['Lx'], stato['p_mesh']['Ly'], stato['p_mesh']['Lz']
        p_sim['posizioni_sonde'] = sonde_parsate
        barra = st.progress(0.0, text="Simulazione in corso...")
        risultati = esegui_simulazione(stato['mesh'], stato['basis'], p_sim,
                                        progress_callback=lambda f: barra.progress(f, text=f"Simulazione in corso... {f*100:.0f}%"))
        barra.empty()
        stato['risultati'], stato['p_sim'] = risultati, p_sim
        stato.pop('viz_output', None)
    except ValueError as e:
        st.error(f"Errore: {e}")

if 'risultati' in stato:
    r = stato['risultati']
    st.success(f"Simulazione completata: T_min={r['T'].min():.2f}°C, T_max={r['T'].max():.2f}°C")

    st.subheader("Opzioni di visualizzazione")
    _, x_u, y_u, z_u, nx, ny_, nz = a_griglia_builder(r['coords'])
    tempi_disp = sorted(r['snapshot_T'].keys())

    istanti_sel = st.multiselect("Istanti da mostrare", tempi_disp, default=tempi_disp,
                                  format_func=lambda t: f"{t:.0f}s", key="istsel")

    c1, c2 = st.columns(2)
    with c1:
        idx_y = st.slider("Slice sezione xz (indice y)", 0, ny_-1, min(ny_//2, ny_-1), key="slicey")
        st.caption(f"y = {y_u[idx_y]*1000:.1f} mm")
    with c2:
        idx_z = st.slider("Slice sezione xy (indice z)", 0, nz-1, min(max(0, nz-3), nz-1), key="slicez")
        st.caption(f"z = {z_u[idx_z]*1000:.1f} mm")

    c3, c4 = st.columns(2)
    with c3:
        st.caption("Linea profili lungo x (T tra elettrodi, Potenziale) — default: mezzeria")
        linea_el_y = st.number_input("y [mm]", value=round(stato['p_sim']['Ly']*1000/2, 1), key="liny1")
        linea_el_z = st.number_input("z [mm]", value=round(stato['p_sim']['Lz']*1000/2, 1), key="linz1")
    with c4:
        st.caption("Linea profili lungo z (Conducibilita', Sorgente Joule) — default: centro")
        linea_v_x = st.number_input("x [mm]", value=round(stato['p_sim']['Lx']*1000/2, 1), key="linx2")
        linea_v_y = st.number_input("y [mm]", value=round(stato['p_sim']['Ly']*1000/2, 1), key="liny2")

    viz = dict(istanti=istanti_sel, idx_y=idx_y, idx_z=idx_z,
               linea_y=linea_el_y/1000, linea_z=linea_el_z/1000,
               linea_x=linea_v_x/1000, linea_y2=linea_v_y/1000)

    if st.button("3) Aggiorna visualizzazioni", type="primary"):
        # chiude le figure del giro precedente (se presenti), altrimenti si
        # accumulano ad ogni click senza mai essere rilasciate
        vecchio = stato.get('viz_output')
        if vecchio:
            for k, v in vecchio.items():
                if k.startswith('gif'):
                    continue
                plt.close(v)

        with st.spinner("Generazione grafici e animazioni..."):
            try:
                vo = {}
                vo['profilo_z'] = fig_profilo_termico_z(r, viz)
                vo['sonde'] = fig_sonde(r)
                vo['sezione_xz'] = fig_sezione_xz(r, viz)
                vo['sezione_xy'] = fig_sezione_xy(r, viz)
                vo['scatter3d'] = fig_scatter3d(r)
                vo['profilo_elettrodi'] = fig_profilo_elettrodi(r, viz)
                vo['conducibilita'] = fig_conducibilita(r, viz)
                vo['potenziale'] = fig_potenziale(r, viz, stato['p_sim'])
                vo['sorgente_joule'] = fig_sorgente_joule(r, viz, stato['p_sim'])
                try:
                    vo['gif2d'], vo['gif3d'] = genera_animazioni(r, viz)
                except Exception as e:
                    st.warning(f"Le animazioni non sono state generate: {e}")
                    vo['gif2d'], vo['gif3d'] = None, None
                stato['viz_output'] = vo
            except Exception as e:
                st.exception(e)

    if 'viz_output' in stato:
        vo = stato['viz_output']
        st.subheader("Andamenti nel tempo")
        c1, c2 = st.columns(2)
        with c1:
            st.pyplot(vo['profilo_z'], width=LARGHEZZA_2D)
        with c2:
            st.pyplot(vo['sonde'], width=LARGHEZZA_2D)
        with_coords = "\n".join(f"{n}: x={x:.1f}mm, y={y:.1f}mm, z={z:.1f}mm" for n,(x,y,z) in r['posizioni_sonde_mm'].items())
        st.text(with_coords)

        st.subheader("Struttura spaziale")
        c3, c4 = st.columns(2)
        with c3: st.pyplot(vo['sezione_xz'], width=LARGHEZZA_2D)
        with c4: st.pyplot(vo['sezione_xy'], width=LARGHEZZA_2D)
        st.pyplot(vo['scatter3d'], width=LARGHEZZA_3D)

        st.subheader("Profili lungo gli assi")
        st.pyplot(vo['profilo_elettrodi'], width=LARGHEZZA_2D)
        c5, c6 = st.columns(2)
        with c5: st.pyplot(vo['conducibilita'], width=LARGHEZZA_2D)
        with c6: st.pyplot(vo['potenziale'], width=LARGHEZZA_2D)
        st.pyplot(vo['sorgente_joule'], width=LARGHEZZA_2D)

        st.subheader("Animazioni")
        # NOTA IMPORTANTE: qui NON si puo' usare width=<numero> come per gli
        # altri grafici. Streamlit, quando riceve un width numerico piu'
        # piccolo dell'immagine originale, la ridimensiona internamente
        # riaprendola con PIL e risalvandola SENZA save_all=True -- questo
        # elimina tutti i frame della GIF tranne il primo (il bug "si vede
        # come un'immagine statica"). width="content" mostra la GIF alla sua
        # dimensione nativa (gia' contenuta, vedi figsize in genera_animazioni)
        # ed evita del tutto quel percorso di ridimensionamento.
        c7, c8 = st.columns(2)
        with c7:
            if vo.get('gif2d'):
                st.image(vo['gif2d'], width='content')
            else:
                st.info("Animazione 2D non disponibile (vedi avviso sopra).")
        with c8:
            if vo.get('gif3d'):
                st.image(vo['gif3d'], width='content')
            else:
                st.info("Animazione 3D non disponibile (vedi avviso sopra).")
