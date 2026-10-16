"""
Motore di calcolo: riscaldamento ohmico MULTIFASE (purea di patate + polpette di pollo)  -  v2

Modello elettro-termico 3D accoppiato:
  - problema elettrico quasi-statico   div( sigma(x,T) grad V ) = 0
  - sorgente Joule                     q = sigma |grad V|^2
  - bilancio termico transitorio       rho(T) cp(T) dT/dt = div( k(T) grad T ) + q
  - sigma(T) lineare per fase:         sigma = a*T + b   [S/m, T in gradi C]   (Tab. 5.1 e 6.1)
  - proprieta' termofisiche:           Choi & Okos (1986), da composizione e temperatura
  - scambio sulla superficie libera:   h(T) = convezione naturale (McAdams) + irraggiamento
                                       (+ evaporazione, opzionale)

Sinusoide a 50 Hz: il problema elettrico e' lineare in V, quindi la potenza media su un
periodo e' V_rms^2 * q_unit(x), con q_unit calcolata dal problema a potenziale unitario
(una sola risoluzione elettrica per passo termico, equivalente al sub-ciclo esplicito).

Fonti dei dati (vedi anche FONTI):
  - Equazioni di Choi & Okos (1986): ASHRAE Handbook - Refrigeration, cap. "Thermal Properties
    of Foods"; regole di miscelazione (cp per frazioni di massa, rho = 1/sum(x/rho_i),
    k = modello parallelo con frazioni di volume): Becker & Fricke, HVAC&R Research 2001.
  - Composizioni dei componenti: USDA FoodData Central (SR Legacy).
  - Convezione naturale, piastra orizzontale con superficie calda rivolta verso l'alto:
    McAdams (1954), L = A/P.
"""

import numpy as np
from skfem import *
from skfem.helpers import dot, grad
from scipy.sparse.linalg import cg
from scipy.sparse import diags

try:
    import pyamg
    HA_PYAMG = True
except Exception:       # pyamg opzionale: senza, si ricade sul solutore diretto
    HA_PYAMG = False

FONTI = {
    'choi_okos': "Choi & Okos (1986), come riportato in ASHRAE Handbook - Refrigeration, cap. Thermal Properties of Foods",
    'miscelazione': "Becker & Fricke, HVAC&R Research 7(4), 2001 (regole di miscelazione)",
    'usda': "USDA FoodData Central, SR Legacy (fiocchi di patate disidratati, burro non salato, pollo macinato crudo)",
    'convezione': "McAdams, Heat Transmission, 3a ed., 1954 (piastra orizzontale, superficie calda verso l'alto)",
}

# --------------------------------------------------------------------------
# Celle sperimentali (Figure 4.2 - 4.4). Quote in metri.
# --------------------------------------------------------------------------
CELLE = {
    'UNISA (10 x 5.1 x 3.7 cm)': dict(Lx=0.100, Ly=0.051, Lz=0.037, margine=0.0025, spaz_rp=0.030),
    'UCD (13 x 10 x 5 cm)':      dict(Lx=0.130, Ly=0.100, Lz=0.050, margine=0.010,  spaz_rp=0.040),
}
D_POLPETTA = 0.030

CONFIGURAZIONI = {
    'RP (solo purea)': [],
    '2PSiRP (polpette in P1 e P3)': [0, 2],
    '2PAiRP (polpette in P1 e P2)': [0, 1],
    '3PiRP (polpette in P1, P2, P3)': [0, 1, 2],
}

# --------------------------------------------------------------------------
# Composizioni (Tabella 4.3, % w/w) e regressioni sigma(T) = a*T + b
#   Tab. 5.1 (RPF, composizioni 1-2) e Tab. 6.1 (RPFNB, composizioni 3-6).
#   Le composizioni 7-9 (RPFB, con burro) non hanno una regressione utilizzabile e sono state rimosse: la ricetta
#   sperimentale con burro si inserisce come 'Personalizzata'.
# --------------------------------------------------------------------------
COMPOSIZIONI = {
    1: dict(burro=5.00, sale=5.70,  fiocchi=23.70, acqua=65.60, cella='UCD',   sigma=(0.0920, 1.9226, 0.9897, 'Tab. 5.1')),
    2: dict(burro=4.90, sale=10.90, fiocchi=19.80, acqua=64.40, cella='UCD',   sigma=(0.1715, 2.8874, 0.9979, 'Tab. 5.1')),
    3: dict(burro=0.00, sale=2.75,  fiocchi=12.68, acqua=84.57, cella='UNISA', sigma=(0.1244, 1.6252, 0.9966, 'Tab. 6.1')),
    4: dict(burro=0.00, sale=1.39,  fiocchi=12.86, acqua=85.74, cella='UNISA', sigma=(0.0508, 1.3689, 0.9933, 'Tab. 6.1')),
    5: dict(burro=0.00, sale=0.74,  fiocchi=12.95, acqua=86.31, cella='UNISA', sigma=(0.0317, 0.8095, 0.9977, 'Tab. 6.1')),
    6: dict(burro=0.00, sale=0.37,  fiocchi=13.00, acqua=86.63, cella='UNISA', sigma=(0.0208, 0.4320, 0.9961, 'Tab. 6.1')),
}
SIGMA_POLPETTE = (0.0340, 0.6087, 0.9829, 'Tab. 5.1')

# Composizione prossimale dei componenti, g/100 g (USDA FoodData Central, SR Legacy).
#   Fiocchi: 'Potatoes, mashed, dehydrated, flakes without milk, dry form'. L'etichetta del prodotto
#   riporta fiocchi di patate al 99% piu' additivi (mono/digliceridi, difosfati, metabisolfito): ~1%, trascurati.
#   Burro: 'Butter, without salt'; l'acqua e' ricavata per differenza a 100 (SR: 17.94).
#   Pollo: 'Chicken, ground, raw' (ricetta delle polpette non nota: valore da sostituire se disponibile).
COMPONENTI_USDA = {
    'fiocchi': dict(acqua=6.58, proteine=8.34, grassi=0.41, carboidrati=74.57, fibre=6.60, ceneri=3.50),
    'burro':   dict(acqua=17.89, proteine=0.85, grassi=81.11, carboidrati=0.06, fibre=0.0, ceneri=0.09),
    'pollo':   dict(acqua=73.30, proteine=17.44, grassi=8.10, carboidrati=0.04, fibre=0.0, ceneri=1.12),
}
NOMI_COMPONENTI = ['acqua', 'proteine', 'grassi', 'carboidrati', 'fibre', 'ceneri']


def normalizza(comp):
    s = sum(comp.get(c, 0.0) for c in NOMI_COMPONENTI)
    return {c: comp.get(c, 0.0) / s for c in NOMI_COMPONENTI}


def composizione_purea(w_burro, w_sale, w_fiocchi, w_acqua, fiocchi=None, burro=None):
    """Frazioni di massa (acqua, proteine, grassi, carboidrati, fibre, ceneri) della purea a
    partire dalla ricetta (% w/w della Tabella 4.3). Il sale (NaCl) e' contato tra le ceneri."""
    fiocchi = fiocchi or COMPONENTI_USDA['fiocchi']
    burro = burro or COMPONENTI_USDA['burro']
    nf, nb = normalizza(fiocchi), normalizza(burro)
    tot = w_burro + w_sale + w_fiocchi + w_acqua
    wb, ws, wf, ww = w_burro / tot, w_sale / tot, w_fiocchi / tot, w_acqua / tot
    comp = {c: wf * nf[c] + wb * nb[c] for c in NOMI_COMPONENTI}
    comp['acqua'] += ww
    comp['ceneri'] += ws
    return comp


# --------------------------------------------------------------------------
# Choi & Okos (1986). T in gradi C; k [W/m/K], rho [kg/m3], cp [kJ/kg/K]
# --------------------------------------------------------------------------
_CO = {
    'proteine':    dict(k=(0.17881, 1.1958e-3, -2.7178e-6), rho=(1329.9, -0.51840, 0.0),  cp=(2.0082, 1.2089e-3, -1.3129e-6)),
    'grassi':      dict(k=(0.18071, -2.7604e-4, -1.7749e-7), rho=(925.59, -0.41757, 0.0), cp=(1.9842, 1.4733e-3, -4.8008e-6)),
    'carboidrati': dict(k=(0.20141, 1.3874e-3, -4.3312e-6), rho=(1599.1, -0.31046, 0.0), cp=(1.5488, 1.9625e-3, -5.9399e-6)),
    'fibre':       dict(k=(0.18331, 1.2497e-3, -3.1683e-6), rho=(1311.5, -0.36589, 0.0), cp=(1.8459, 1.8306e-3, -4.6509e-6)),
    'ceneri':      dict(k=(0.32962, 1.4011e-3, -2.9069e-6), rho=(2423.8, -0.28063, 0.0), cp=(1.0926, 1.8896e-3, -3.6817e-6)),
    'acqua':       dict(k=(0.57109, 1.7625e-3, -6.7036e-6), rho=(997.18, 3.1439e-3, -3.7574e-3), cp=(4.1762, -9.0864e-5, 5.4731e-6)),
}


def _poly(c, T):
    return c[0] + c[1] * T + c[2] * T ** 2


def proprieta_choi_okos(comp, T):
    """(k [W/m/K], rho [kg/m3], cp [J/kg/K]) della miscela alla temperatura T [C] (scalare o array)."""
    T = np.asarray(T, dtype=float)
    X = normalizza(comp)
    rho_i = {c: _poly(_CO[c]['rho'], T) for c in NOMI_COMPONENTI}
    rho = 1.0 / sum(X[c] / rho_i[c] for c in NOMI_COMPONENTI)
    cp = 1000.0 * sum(X[c] * _poly(_CO[c]['cp'], T) for c in NOMI_COMPONENTI)
    k = rho * sum(_poly(_CO[c]['k'], T) * X[c] / rho_i[c] for c in NOMI_COMPONENTI)   # modello parallelo
    return k, rho, cp


# --------------------------------------------------------------------------
# Scambio sulla superficie libera superiore
# --------------------------------------------------------------------------
SIGMA_SB = 5.670374e-8


def _aria(Tf):
    """Proprieta' dell'aria alla temperatura di film Tf [C] (fit lineari validi 0-150 C)."""
    nu = 1.33e-5 + 9.0e-8 * Tf
    alpha = 1.9e-5 + 1.3e-7 * Tf
    k = 0.0243 + 7.5e-5 * Tf
    return nu, alpha, k


def _psat(Tc):
    """Pressione di saturazione dell'acqua [Pa] (formula di Magnus), T in C."""
    return 610.94 * np.exp(17.625 * Tc / (Tc + 243.04))


def componenti_h(Ts, p, L_car):
    """Restituisce (h_conv_naturale, h_irraggiamento, h_evaporazione) equivalenti [W/m2/K],
    definiti rispetto a (Ts - T_amb), per temperatura superficiale Ts [C] (array)."""
    Ts = np.asarray(Ts, dtype=float)
    Ta = p['T_amb']
    dT = np.maximum(Ts - Ta, 0.5)                    # evita la singolarita' per Ts -> Ta
    Tf = 0.5 * (Ts + Ta)
    nu, alpha, kair = _aria(Tf)
    beta = 1.0 / (Tf + 273.15)
    Ra = 9.81 * beta * dT * L_car ** 3 / (nu * alpha)
    Nu = np.where(Ra < 1e7, 0.54 * Ra ** 0.25, 0.15 * Ra ** (1.0 / 3.0))     # McAdams (1954)
    h_nat = Nu * kair / L_car
    Tsk, Tak = Ts + 273.15, Ta + 273.15
    h_rad = p.get('emissivita', 0.95) * SIGMA_SB * (Tsk ** 2 + Tak ** 2) * (Tsk + Tak)
    h_ev = np.zeros_like(Ts)
    if p.get('h_modo') == 'naturale_irraggiamento_evaporazione':
        Rv = 461.5
        rho_v_s = p.get('aw', 0.98) * _psat(Ts) / (Rv * Tsk)
        rho_v_inf = p.get('UR', 0.5) * _psat(Ta) / (Rv * Tak)
        rho_aria, cp_aria, Le = 1.1, 1007.0, 0.85
        h_m = h_nat / (rho_aria * cp_aria) * Le ** (-2.0 / 3.0)          # analogia calore-massa (Chilton-Colburn)
        h_fg = (2501.0 - 2.37 * Ts) * 1000.0
        q_ev = np.maximum(h_m * (rho_v_s - rho_v_inf), 0.0) * h_fg
        h_ev = q_ev / dT
    return h_nat, h_rad, h_ev


def flusso_superficie(Ts, p, L_car):
    """Flusso di calore uscente dalla superficie libera q_s(Ts) [W/m2]."""
    Ts = np.asarray(Ts, dtype=float)
    if p.get('h_modo', 'costante') == 'costante':
        return p.get('fattore_h', 1.0) * p['h_conv'] * (Ts - p['T_amb'])
    h_nat, h_rad, h_ev = componenti_h(Ts, p, L_car)
    dT = np.maximum(Ts - p['T_amb'], 0.5)
    # (h_nat + h_rad) agiscono su (Ts - Ta) con segno; l'evaporazione e' un flusso diretto h_ev*dT
    return p.get('fattore_h', 1.0) * ((h_nat + h_rad) * (Ts - p['T_amb']) + h_ev * dT)


def flusso_e_derivata(Ts, p, L_car):
    """(q_s, dq_s/dT): linearizzazione di Newton del flusso, usata come condizione di Robin:
    q ~ q0 + h_lin (T - Ts). Valida anche vicino alla temperatura ambiente."""
    d = 0.05
    q0 = flusso_superficie(Ts, p, L_car)
    h_lin = (flusso_superficie(Ts + d, p, L_car) - flusso_superficie(Ts - d, p, L_car)) / (2 * d)
    return q0, np.maximum(h_lin, 0.0)


def h_equivalente(Ts, p, L_car):
    """h equivalente totale [W/m2/K] (somma dei contributi), solo per visualizzazione."""
    Ts = np.asarray(Ts, dtype=float)
    if p.get('h_modo', 'costante') == 'costante':
        return np.full_like(Ts, p['h_conv'] * p.get('fattore_h', 1.0))
    h_nat, h_rad, h_ev = componenti_h(Ts, p, L_car)
    return p.get('fattore_h', 1.0) * (h_nat + h_rad + h_ev)


# --------------------------------------------------------------------------
# Geometria
# --------------------------------------------------------------------------
def geometria(cella, configurazione, override=None):
    """Dimensioni, centri delle polpette e sonde P1-P3. 'override' (dict, opzionale) sostituisce:
    Lx, Ly, Lz, margine, spaz_rp (m), D (diametro polpetta, m), occupazione (3 bool: P1, P2, P3)."""
    c = dict(CELLE[cella])
    D = D_POLPETTA
    occ = list(CONFIGURAZIONI[configurazione])
    if override:
        for k in ('Lx', 'Ly', 'Lz', 'margine', 'spaz_rp'):
            if override.get(k) is not None:
                c[k] = override[k]
        D = override.get('D', D)
        if override.get('occupazione') is not None:
            occ = [i for i, b in enumerate(override['occupazione']) if b]
    r = D / 2
    y0, z0 = c['Ly'] / 2, c['Lz'] / 2
    avvisi = []
    if not occ:
        xs = [c['Lx'] / 2 - c['spaz_rp'], c['Lx'] / 2, c['Lx'] / 2 + c['spaz_rp']]
    else:
        xs = [c['margine'] + r, c['Lx'] / 2, c['Lx'] - c['margine'] - r]
        if 2 * r > c['Ly'] + 1e-12 or 2 * r > c['Lz'] + 1e-12:
            raise ValueError("La polpetta e' piu' grande della sezione della cella (diametro > Ly o Lz).")
        if c['margine'] < 0 or xs[0] - r < -1e-12 or xs[2] + r > c['Lx'] + 1e-12:
            raise ValueError("Le polpette escono dalla cella lungo x: controlla Lx, diametro e margine.")
        sel = [xs[i] for i in occ]
        for a, b in zip(sel[:-1], sel[1:]):
            if b - a < 2 * r - 1e-12:
                avvisi.append("Le polpette si sovrappongono lungo x: l'unione delle sfere e' trattata come una sola fase.")
                break
    centri = [(xs[i], y0, z0) for i in occ]
    sonde = {f'P{i + 1}': (xs[i], y0, z0) for i in range(3)}
    return dict(Lx=c['Lx'], Ly=c['Ly'], Lz=c['Lz'], R=r, D=D, centri=centri, sonde=sonde, avvisi=avvisi)


def gap_minimo(geo):
    """Spessore minimo (m) degli strati di purea tra polpette e pareti/elettrodi/altre polpette."""
    if not geo['centri']:
        return None
    R = geo['R']
    gaps = [(geo['Ly'] - 2 * R) / 2, (geo['Lz'] - 2 * R) / 2]
    xs = sorted(c[0] for c in geo['centri'])
    gaps.append(xs[0] - R)
    gaps.append(geo['Lx'] - xs[-1] - R)
    gaps += [b - a - 2 * R for a, b in zip(xs[:-1], xs[1:])]
    gaps = [g for g in gaps if g > 1e-9]
    return min(gaps) if gaps else None


def stima_nodi(geo, h):
    nx = max(3, int(round(geo['Lx'] / h)) + 1)
    ny = max(3, int(round(geo['Ly'] / h)) + 1)
    nz = max(3, int(round(geo['Lz'] / h)) + 1)
    return nx * ny * nz, (nx, ny, nz)


def costruisci_mesh(geo, h):
    """Mesh tetraedrica strutturata con passo ~h [m] (ogni cella e' divisa in 6 tetraedri)."""
    _, (nx, ny, nz) = stima_nodi(geo, h)
    mesh = MeshTet.init_tensor(np.linspace(0, geo['Lx'], nx),
                               np.linspace(0, geo['Ly'], ny),
                               np.linspace(0, geo['Lz'], nz))
    basis = Basis(mesh, ElementTetP1())
    h_eff = max(geo['Lx'] / (nx - 1), geo['Ly'] / (ny - 1), geo['Lz'] / (nz - 1))
    return mesh, basis, h_eff


def spigoli_piano(mesh, asse, valore):
    """Spigoli della mesh giacenti nel piano coordinata[asse] = valore (asse 0,1,2).
    Restituisce (segmenti (n,2,2), assi_del_piano)."""
    P = mesh.p
    E = mesh.edges
    tol = 1e-9
    in_piano = (np.abs(P[asse, E[0]] - valore) < tol) & (np.abs(P[asse, E[1]] - valore) < tol)
    ass = [i for i in range(3) if i != asse]
    seg = np.stack([P[ass][:, E[0][in_piano]].T, P[ass][:, E[1][in_piano]].T], axis=1)
    return seg, ass


def spigoli_superficie(mesh):
    """Spigoli dei triangoli di bordo (la 'pelle' della mesh), shape (n,2,3)."""
    ids = mesh.boundary_facets()
    F = mesh.facets[:, ids]
    coppie = np.concatenate([F[[0, 1]], F[[1, 2]], F[[2, 0]]], axis=1)
    coppie = np.sort(coppie, axis=0)
    coppie = np.unique(coppie, axis=1)
    return np.stack([mesh.p[:, coppie[0]].T, mesh.p[:, coppie[1]].T], axis=1)


def frazione_fase(X, centri, R, h_eff):
    """Frazione di polpetta phi(x) in [0,1] nei punti X (shape (3, ...)). Rampa lineare di
    larghezza h_eff attorno alla superficie sferica (il chiamante passa meta' del passo di mesh)."""
    phi = np.zeros(X.shape[1:])
    for (cx, cy, cz) in centri:
        r = np.sqrt((X[0] - cx) ** 2 + (X[1] - cy) ** 2 + (X[2] - cz) ** 2)
        phi = np.maximum(phi, np.clip(0.5 - (r - R) / h_eff, 0.0, 1.0))
    return phi


def sigma_campo(T, phi, p):
    """sigma(T, phi): miscela aritmetica delle due leggi lineari per fase."""
    s_m = p['matrice_a'] * T + p['matrice_b']
    s_p = p['polpetta_a'] * T + p['polpetta_b']
    return np.maximum((1.0 - phi) * s_m + phi * s_p, 1e-3)


# --------------------------------------------------------------------------
# Parametri di default
# --------------------------------------------------------------------------
PARAMETRI_DEFAULT = dict(
    matrice_a=0.1244, matrice_b=1.6252,             # composizione 3 (UNISA), Tab. 6.1
    polpetta_a=SIGMA_POLPETTE[0], polpetta_b=SIGMA_POLPETTE[1],
    prop_modo='choi_okos_T',                        # 'choi_okos_T' | 'choi_okos_Tref' | 'manuali'
    T_rif_prop=50.0,
    comp_matrice=composizione_purea(0.0, 2.75, 12.68, 84.57),
    comp_polpetta=normalizza(COMPONENTI_USDA['pollo']),
    k_matrice=0.52, rho_matrice=1060.0, cp_matrice=3500.0,        # usati solo in modo 'manuali'
    k_polpetta=0.48, rho_polpetta=1050.0, cp_polpetta=3400.0,
    V_rms=4.0 * 100.0 * 0.100,                      # 4 V/cm su 10 cm
    T_iniziale=20.0,
    h_modo='naturale_irraggiamento', h_conv=10.0, T_amb=20.0,
    emissivita=0.95, UR=0.5, aw=0.98, fattore_h=1.0,
    dt=3.0, durata=180.0,
    h_mesh_mm=2.5,
    fk_matrice=1.0, fk_polpetta=1.0, fc_matrice=1.0, fc_polpetta=1.0,   # fattori di calibrazione
)


def _tabelle_proprieta(p, T_grid):
    """Tabelle (k, rho*cp) per fase su una griglia di temperature, per interpolazione veloce."""
    out = {}
    for fase, ck, kk, rk, ck_ in (('m', 'comp_matrice', 'k_matrice', 'rho_matrice', 'cp_matrice'),
                                  ('p', 'comp_polpetta', 'k_polpetta', 'rho_polpetta', 'cp_polpetta')):
        if p['prop_modo'] == 'manuali':
            k = np.full_like(T_grid, p[kk]); rc = np.full_like(T_grid, p[rk] * p[ck_])
        elif p['prop_modo'] == 'choi_okos_Tref':
            k0, r0, c0 = proprieta_choi_okos(p[ck], p['T_rif_prop'])
            k = np.full_like(T_grid, float(k0)); rc = np.full_like(T_grid, float(r0 * c0))
        else:
            k, r, c = proprieta_choi_okos(p[ck], T_grid)
            rc = r * c
        sfx = 'matrice' if fase == 'm' else 'polpetta'
        out[fase] = (k * p.get('fk_' + sfx, 1.0), rc * p.get('fc_' + sfx, 1.0))
    return out


# --------------------------------------------------------------------------
# Solutore elettrico
# --------------------------------------------------------------------------
def _risolvi_elettrico(A, b0, V_init, D):
    Ai, bi, xx, I = condense(A, b0, x=V_init, D=D)
    if HA_PYAMG and A.shape[0] > 3000:
        ml = pyamg.smoothed_aggregation_solver(Ai.tocsr())
        xi, info = cg(Ai, bi, x0=xx[I], M=ml.aspreconditioner(cycle='V'), rtol=1e-10, maxiter=400)
        if info != 0:
            raise RuntimeError("Il solutore elettrico iterativo non e' converso.")
        xx[I] = xi
        return xx
    return solve(Ai, bi, xx, I)


def simula(geo, mesh, basis, h_eff, p, progress_callback=None):
    """Ciclo temporale accoppiato. Restituisce un dizionario con tutti i risultati."""
    dt = p['dt']
    n_passi = max(1, int(round(p['durata'] / dt)))
    V_rms = p['V_rms'] if 'V_rms' in p else p['E_rms_V_cm'] * 100.0 * geo['Lx']

    Xq = basis.global_coordinates().value
    phi_q = frazione_fase(Xq, geo['centri'], geo['R'], 0.5 * h_eff)
    phi_n = frazione_fase(basis.doflocs, geo['centri'], geo['R'], 0.5 * h_eff)

    T_grid = np.arange(-10.0, 160.1, 1.0)
    tab = _tabelle_proprieta(p, T_grid)
    prop_dip_T = p['prop_modo'] == 'choi_okos_T'

    def campi_termici(T_q):
        k_m = np.interp(T_q, T_grid, tab['m'][0]); k_p = np.interp(T_q, T_grid, tab['p'][0])
        r_m = np.interp(T_q, T_grid, tab['m'][1]); r_p = np.interp(T_q, T_grid, tab['p'][1])
        return (1 - phi_q) * k_m + phi_q * k_p, (1 - phi_q) * r_m + phi_q * r_p

    @BilinearForm
    def laplace(u, v, w):
        return w['sigma'] * dot(grad(u), grad(v))

    @LinearForm
    def sorgente(v, w):
        return w['q'] * v

    @BilinearForm
    def massa(u, v, w):
        return w['rhocp'] * u * v

    @BilinearForm
    def conduzione(u, v, w):
        return w['k'] * dot(grad(u), grad(v))

    @BilinearForm
    def massa_unitaria(u, v, w):
        return u * v

    ml = np.asarray(massa_unitaria.assemble(basis).sum(axis=1)).ravel()

    # scambio sulla superficie libera z = Lz (coefficiente h variabile con T superficiale)
    facce = mesh.facets_satisfying(lambda x: np.isclose(x[2], geo['Lz']))
    fb = FacetBasis(mesh, basis.elem, facets=facce)
    L_car = geo['Lx'] * geo['Ly'] / (2.0 * (geo['Lx'] + geo['Ly']))     # A/P della faccia superiore

    @BilinearForm
    def robin_a(u, v, w):
        return w['h'] * u * v

    @LinearForm
    def robin_l(v, w):
        return w['g'] * v

    dofs_sx = basis.get_dofs(lambda x: np.isclose(x[0], 0.0))
    dofs_dx = basis.get_dofs(lambda x: np.isclose(x[0], geo['Lx']))
    D = np.concatenate([dofs_sx.all(), dofs_dx.all()])
    V_init = 1.0 - basis.doflocs[0] / geo['Lx']        # potenziale lineare: punto di partenza per l'iterativo
    b_zero = basis.zeros()

    nomi = list(geo['sonde'].keys())
    punti = np.array([geo['sonde'][n] for n in nomi]).T
    Pr = basis.probes(punti).tocsr()
    fase_sonda = {n: float(frazione_fase(np.array(geo['sonde'][n]).reshape(3, 1), geo['centri'], geo['R'], 0.5 * h_eff)[0])
                  for n in nomi}

    # temperatura iniziale per fase (le polpette possono partire piu' fredde della purea)
    T0_m = p['T_iniziale']
    T0_p = p.get('T_iniziale_polpetta', T0_m)
    T = (1.0 - phi_n) * T0_m + phi_n * T0_p
    T_q0 = basis.interpolate(T).value
    k_q, rhocp_q = campi_termici(T_q0)
    M = massa.assemble(basis, rhocp=rhocp_q)
    A0 = (M / dt + conduzione.assemble(basis, k=k_q)).tocsr()

    T_hist, q_hist, P_hist, tempo = [T.copy()], [], [0.0], [0.0]
    h_hist = []
    T_sonde = {n: [float((Pr @ T)[i])] for i, n in enumerate(nomi)}

    for passo in range(n_passi):
        T_q = basis.interpolate(T).value
        # (a) problema elettrico a potenziale unitario con sigma(T^n)
        sig_q = sigma_campo(T_q, phi_q, p)
        Vu = _risolvi_elettrico(laplace.assemble(basis, sigma=sig_q), b_zero, V_init, D)
        gV = basis.interpolate(Vu).grad
        q_q = V_rms ** 2 * sig_q * (gV[0] ** 2 + gV[1] ** 2 + gV[2] ** 2)     # media su un periodo [W/m3]
        b_q = sorgente.assemble(basis, q=q_q)
        # (b) proprieta' termiche (aggiornate con T^n se dipendenti da T)
        if prop_dip_T and passo > 0:
            k_q, rhocp_q = campi_termici(T_q)
            M = massa.assemble(basis, rhocp=rhocp_q)
            A0 = (M / dt + conduzione.assemble(basis, k=k_q)).tocsr()
        # (c) scambio con l'ambiente: h(T superficiale)
        Tf = fb.interpolate(T).value
        q0_f, h_lin = flusso_e_derivata(Tf, p, L_car)
        K_r = robin_a.assemble(fb, h=h_lin)
        b_r = robin_l.assemble(fb, g=h_lin * Tf - q0_f)
        h_hist.append(float(h_equivalente(Tf, p, L_car).mean()))
        # (d) passo termico implicito (sistema SPD dominato dalla massa: CG con Jacobi)
        A = (A0 + K_r).tocsr()
        Dinv = diags(1.0 / A.diagonal())
        T, info = cg(A, M @ T / dt + b_q + b_r, x0=T, M=Dinv, rtol=1e-11, maxiter=500)
        if info != 0:
            raise RuntimeError("Il solutore termico non e' converso.")

        tempo.append((passo + 1) * dt)
        T_hist.append(T.copy())
        q_hist.append(b_q / ml)
        P_hist.append(float(b_q.sum()))
        Tp = Pr @ T
        for i, n in enumerate(nomi):
            T_sonde[n].append(float(Tp[i]))
        if progress_callback is not None:
            progress_callback((passo + 1) / n_passi)

    q_hist.insert(0, q_hist[0])
    P_hist[0] = P_hist[1]
    h_hist.insert(0, h_hist[0])

    return dict(
        tempo=np.array(tempo), T_hist=np.array(T_hist), q_hist=np.array(q_hist),
        P=np.array(P_hist), V_rms=V_rms, phi_n=phi_n, ml=ml, h_hist=np.array(h_hist),
        T_sonde={n: np.array(v) for n, v in T_sonde.items()}, fase_sonda=fase_sonda,
        coords=basis.doflocs, geo=geo, p=p, h_eff=h_eff,
    )


def metriche_cold_spot(ris):
    """Temperatura media (pesata sul volume) di purea e polpette nel tempo."""
    ml, phi = ris['ml'], ris['phi_n']
    w_m, w_p = ml * (1 - phi), ml * phi
    T_m = (ris['T_hist'] @ w_m) / w_m.sum()
    if w_p.sum() > 0:
        T_p = (ris['T_hist'] @ w_p) / w_p.sum()
        T_p_min = np.array([T[phi > 0.5].min() for T in ris['T_hist']])
    else:
        T_p, T_p_min = None, None
    return dict(T_matrice_media=T_m, T_polpette_media=T_p, T_polpette_min=T_p_min)


# --------------------------------------------------------------------------
# Relazioni in LaTeX (per la visualizzazione) e tabelle di letteratura
# --------------------------------------------------------------------------
LATEX_SIGMA = [r"\sigma(T)=a\,T+b=\sigma_{20}\,\big[1+\alpha\,(T-20)\big],\qquad \sigma_{20}=20\,a+b,\qquad \alpha=\frac{a}{\sigma_{20}}",
               r"\sigma=(1-\varphi)\,\sigma_m(T)+\varphi\,\sigma_p(T)"]
LATEX_CHOI_OKOS = [r"P_i(T)=c_{0,i}+c_{1,i}\,T+c_{2,i}\,T^{2}\qquad (T\ \text{in}\ ^\circ\mathrm{C};\ P=k,\ \rho,\ c_p)",
                   r"c_p=\sum_i x_i\,c_{p,i}",
                   r"\rho=\Big(\sum_i \frac{x_i}{\rho_i}\Big)^{-1}",
                   r"k=\sum_i k_i\,v_i,\qquad v_i=\frac{x_i/\rho_i}{\sum_j x_j/\rho_j}"]
LATEX_SCAMBIO = [r"q_s=\big(h_{nat}+h_{rad}\big)\,(T_s-T_{amb})+q_{ev}",
                 r"\mathrm{Nu}=0.54\,\mathrm{Ra}^{1/4}\ \ (10^{4}<\mathrm{Ra}<10^{7}),\qquad \mathrm{Nu}=0.15\,\mathrm{Ra}^{1/3}\ \ (\mathrm{Ra}>10^{7}),\qquad h_{nat}=\frac{\mathrm{Nu}\,k_a}{L_c},\qquad L_c=\frac{A}{P}",
                 r"\mathrm{Ra}=\frac{g\,\beta\,(T_s-T_{amb})\,L_c^{3}}{\nu\,\alpha_a},\qquad \beta=\frac{1}{T_f}",
                 r"h_{rad}=\varepsilon\,\sigma_{SB}\,\big(T_s^{2}+T_{amb}^{2}\big)\big(T_s+T_{amb}\big)",
                 r"q_{ev}=h_m\,(\rho_{v,s}-\rho_{v,\infty})\,h_{fg},\qquad h_m=\frac{h_c}{\rho_a\,c_{p,a}\,\mathrm{Le}^{2/3}},\qquad \mathrm{Le}=0.85",
                 r"p_{sat}=610.94\,\exp\!\Big(\frac{17.625\,T}{T+243.04}\Big)\ [\mathrm{Pa}]",
                 r"\nu=1.33\cdot10^{-5}+9.0\cdot10^{-8}\,T,\qquad \alpha_a=1.9\cdot10^{-5}+1.3\cdot10^{-7}\,T,\qquad k_a=0.0243+7.5\cdot10^{-5}\,T\qquad (T\ \text{in}\ ^\circ\mathrm{C})"]
LEGENDA_SCAMBIO = ("$T_s$ temperatura superficiale; $T_{amb}$ temperatura dell'aria; $h_{nat}$, $h_{rad}$ coefficienti di convezione naturale e di "
                   "irraggiamento; $q_{ev}$ flusso evaporativo; $\\mathrm{Nu}$, $\\mathrm{Ra}$ numeri di Nusselt e di Rayleigh; $k_a$, $\\nu$, $\\alpha_a$ "
                   "conducibilità, viscosità cinematica e diffusività termica dell'aria; $L_c$ lunghezza caratteristica; $g$ gravità; $T_f$ temperatura di "
                   "film (K); $\\varepsilon$ emissività; $\\sigma_{SB}$ costante di Stefan-Boltzmann; $h_m$ coefficiente di scambio di massa; "
                   "$\\rho_{v,s}$, $\\rho_{v,\\infty}$ densità del vapore alla superficie e nell'aria; $h_{fg}$ calore latente; $\\mathrm{Le}$ numero di Lewis.")
LEGENDA_CHOI_OKOS = ("$x_i$ frazione di massa del componente $i$ (acqua, proteine, grassi, carboidrati, fibre, ceneri); $v_i$ frazione di volume; "
                     "$c_{0,i},c_{1,i},c_{2,i}$ coefficienti del polinomio di ciascun componente (tabella seguente); $k_i,\\rho_i,c_{p,i}$ proprietà del "
                     "componente alla temperatura $T$.")
LEGENDA_SIGMA = ("$\\sigma$ conducibilità elettrica [S/m]; $a$ pendenza [S/(m °C)]; $b$ intercetta [S/m]; $\\sigma_{20}$ conducibilità a 20 °C; "
                 "$\\alpha$ coefficiente di temperatura [1/K]; $\\varphi$ frazione di fase (0 nella purea, 1 nella polpetta); pedici $m$ e $p$: purea e polpetta.")


def _sci(v):
    """numero con notazione scientifica in LaTeX se molto piccolo/grande"""
    if v == 0:
        return "0"
    if abs(v) < 1e-3 or abs(v) >= 1e5:
        m, e = f"{v:.4e}".split('e')
        return f"${float(m):g}\\cdot10^{{{int(e)}}}$"
    return f"{v:.5g}"


def md_tabella(intestazioni, righe):
    """Tabella Markdown (le celle possono contenere LaTeX tra $...$)."""
    t = "| " + " | ".join(intestazioni) + " |\n|" + "|".join(["---"] * len(intestazioni)) + "|\n"
    for r in righe:
        t += "| " + " | ".join(str(x) for x in r) + " |\n"
    return t


def md_choi_okos():
    """Coefficienti di Choi-Okos dei sei componenti, con intestazioni in LaTeX."""
    head = ["Componente"] + [f"${g}$: $c_{{{j}}}$" for g in ("k", r"\rho", "c_p") for j in (0, 1, 2)]
    righe = []
    for c in NOMI_COMPONENTI:
        d = _CO[c]
        righe.append([c.capitalize()] + [_sci(v) for g in ('k', 'rho', 'cp') for v in d[g]])
    return md_tabella(head, righe)


def md_sigma(par):
    """par: lista di (fase, a, b, fonte) -> tabella Markdown dei coefficienti di sigma(T)."""
    righe = []
    for nome, a, b, fonte in par:
        s20 = 20 * a + b
        righe.append([nome, f"{a:.4f}", f"{b:.4f}", f"{s20:.3f}", f"{a / s20:.4f}" if s20 else "-", fonte])
    return md_tabella(["Fase", r"$a$ [S/(m °C)]", r"$b$ [S/m]", r"$\sigma_{20}$ [S/m]", r"$\alpha$ [1/K]", "Fonte"], righe)


def tabelle_letteratura():
    """Lista di (titolo, contenuto, nota). 'contenuto' e': DataFrame (st.dataframe), ('md', str) (st.markdown, con LaTeX)
    oppure ('latex', [str, ...]) (una st.latex per elemento)."""
    import pandas as pd
    out = []
    tipo = {1: 'RPF', 2: 'RPF', 3: 'RPFNB', 4: 'RPFNB', 5: 'RPFNB', 6: 'RPFNB'}
    out.append(("Tabella 4.3 - Composizione dei campioni di purea (% w/w)",
                pd.DataFrame([{'Composizione': i, 'Tipo': tipo[i], 'Burro non salato': c['burro'], 'Sale': c['sale'],
                               'Fiocchi di patate': c['fiocchi'], 'Acqua distillata': c['acqua'], 'Usata presso': c['cella']}
                              for i, c in COMPOSIZIONI.items()]),
                "RPF = purea ricostituita (comp. 1-2); RPFNB = senza burro (comp. 3-6). Le composizioni con burro (7-9), prive di regressione di "
                "conducibilità utilizzabile, non sono incluse: la ricetta con burro si inserisce come «Personalizzata». Fonte: tesi (Tabella 4.3)."))

    righe = []
    for i, c in COMPOSIZIONI.items():
        a, b, r2, fonte = c['sigma']
        righe.append({'Prodotto': f"Comp. {i} ({tipo[i]}, sale {c['sale']:g}%)", 'a [S/m/°C]': a, 'b [S/m]': b,
                      'R²': r2, 'σ(20 °C) [S/m]': 20 * a + b, 'Fonte': fonte})
    a, b, r2, fonte = SIGMA_POLPETTE
    righe.append({'Prodotto': 'Polpette di pollo', 'a [S/m/°C]': a, 'b [S/m]': b, 'R²': r2, 'σ(20 °C) [S/m]': 20 * a + b, 'Fonte': fonte})
    out.append(("Tabelle 5.1 e 6.1 - Conducibilità elettrica: regressione lineare σ = a·T + b (T in °C)", pd.DataFrame(righe),
                "Valide nell'intervallo di temperatura delle misure originali; estrapolate fuori da esso."))
    out.append(("Relazioni della conducibilità elettrica", ('latex', LATEX_SIGMA), LEGENDA_SIGMA.replace('$', '$')))

    righe = []
    for nome, c in CELLE.items():
        righe.append({'Cella': nome, 'Lx - distanza elettrodi [mm]': c['Lx'] * 1000, 'Ly [mm]': c['Ly'] * 1000,
                      'Lz - altezza riempimento [mm]': c['Lz'] * 1000, 'Diametro polpette [mm]': D_POLPETTA * 1000,
                      'Distanza parete-polpetta [mm]': c['margine'] * 1000,
                      'Spaziatura sonde in RP [mm]': c['spaz_rp'] * 1000})
    out.append(("Celle sperimentali (Figure 4.2, 4.3 e 4.4)", pd.DataFrame(righe),
                "Elettrodi a faccia piena alle due estremità lungo x. Polpette (sfere) e sonde sulla mezzeria (y = Ly/2, z = Lz/2)."))

    righe = []
    for nome, titolo in (('fiocchi', 'Fiocchi di patate disidratati'), ('burro', 'Burro non salato'), ('pollo', 'Pollo macinato crudo')):
        d = {'Componente': titolo}
        d.update({c.capitalize(): COMPONENTI_USDA[nome][c] for c in NOMI_COMPONENTI})
        righe.append(d)
    out.append(("Composizione prossimale dei componenti, g/100 g (USDA FoodData Central, SR Legacy)", pd.DataFrame(righe),
                "Fiocchi: 'Potatoes, mashed, dehydrated, flakes without milk, dry form'. L'etichetta del prodotto riporta fiocchi di patate al 99% "
                "più additivi (mono/digliceridi, difosfati, metabisolfito) circa 1%, trascurati. Burro: acqua per differenza a 100 (SR: 17.94). "
                "Pollo: ricetta delle polpette non nota."))

    out.append(("Equazioni di Choi e Okos (1986): relazioni e regole di miscelazione", ('latex', LATEX_CHOI_OKOS), LEGENDA_CHOI_OKOS))
    out.append(("Coefficienti di Choi e Okos dei componenti", ('md', md_choi_okos()),
                f"$k$ in W/(m K), $\\rho$ in kg/m³, $c_p$ in kJ/(kg K). Fonte: {FONTI['choi_okos']}. Il coefficiente lineare di $k$ per i grassi è "
                r"$-2.7604\cdot10^{-4}$ (ASHRAE); alcune trascrizioni riportano $-2.7604\cdot10^{-3}$, che darebbe conducibilità negative sopra circa 65 °C."))
    out.append(("Scambio sulla superficie libera: correlazioni e relazioni", ('latex', LATEX_SCAMBIO),
                LEGENDA_SCAMBIO + f" Fonti: {FONTI['convezione']}; analogia calore-massa (Chilton-Colburn); formula di Magnus; fit lineari dell'aria validi tra 0 e 150 °C."))

    righe = []
    for i, c in COMPOSIZIONI.items():
        comp = composizione_purea(c['burro'], c['sale'], c['fiocchi'], c['acqua'])
        for T in (20, 50, 80):
            k, r, cp = proprieta_choi_okos(comp, T)
            righe.append({'Prodotto': f"Comp. {i}", 'T [°C]': T, 'k [W/m/K]': float(k), 'ρ [kg/m³]': float(r), 'cp [J/kg/K]': float(cp)})
    for T in (20, 50, 80):
        k, r, cp = proprieta_choi_okos(normalizza(COMPONENTI_USDA['pollo']), T)
        righe.append({'Prodotto': 'Polpette (pollo)', 'T [°C]': T, 'k [W/m/K]': float(k), 'ρ [kg/m³]': float(r), 'cp [J/kg/K]': float(cp)})
    out.append(("Esempio: proprietà termofisiche risultanti (Choi-Okos applicata alle composizioni sopra)", pd.DataFrame(righe),
                "Valori calcolati dalle relazioni precedenti, a titolo di controllo."))
    return out
