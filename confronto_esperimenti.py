"""
Calibrazione e validazione del modello fisico con dati sperimentali (riscaldamento ohmico multifase).

Legge un file Excel con colonne
    Configurazione | Tensione | Concentrazione salina | Ttempo | Temperatura 1 | Temperatura 2 | Temperatura 3
(una riga per istante, piu' prove nello stesso foglio), traduce gli attributi di ciascuna prova nei
parametri del modello (multifase_engine), simula e confronta con le sonde. Permette di calibrare
QUALSIASI sottoinsieme delle proprieta' del modello rispetto ai valori di letteratura.
"""

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

import multifase_engine as me

# ---------------------------------------------------------------------------
# Lettura e validazione del file
# ---------------------------------------------------------------------------
COLONNE = ['configurazione', 'tensione', 'sale', 'tempo', 'T1', 'T2', 'T3']


def _normalizza_nome(c):
    k = str(c).strip().lower().replace(' ', '').replace('_', '')
    if k.startswith('config'):
        return 'configurazione'
    if k.startswith('tension') or k in ('voltage', 'volt', 'v'):
        return 'tensione'
    if k.startswith('concentr') or k.startswith('sal') or 'salt' in k:
        return 'sale'
    if k in ('ttempo', 'tempo', 'time', 'tempos', 't[s]', 'ts'):
        return 'tempo'
    for i in (1, 2, 3):
        if k in (f'temperatura{i}', f'temp{i}', f't{i}', f'temperature{i}', f'sonda{i}', f'p{i}'):
            return f'T{i}'
    return None


def leggi_dati(sorgente):
    """Restituisce (DataFrame normalizzato, lista di avvisi). Solleva ValueError se mancano colonne."""
    df = pd.read_excel(sorgente)
    ren = {}
    for c in df.columns:
        n = _normalizza_nome(c)
        if n and n not in ren.values():
            ren[c] = n
    df = df.rename(columns=ren)
    mancanti = [c for c in COLONNE if c not in df.columns]
    if mancanti:
        raise ValueError(f"Colonne non riconosciute: {mancanti}. Colonne trovate nel file: {list(df.columns)}")
    df = df[COLONNE].copy()
    avvisi = []
    for c in COLONNE[1:]:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df['configurazione'] = df['configurazione'].astype(str).str.strip()
    nan_righe = df[df[['tensione', 'sale', 'tempo']].isna().any(axis=1)]
    if len(nan_righe):
        avvisi.append(f"{len(nan_righe)} righe senza tensione/sale/tempo sono state scartate.")
        df = df.drop(nan_righe.index)
    incomplete = df[df[['T1', 'T2', 'T3']].isna().any(axis=1)]
    if len(incomplete):
        for (c, v, s), n in incomplete.groupby(['configurazione', 'tensione', 'sale']).size().items():
            avvisi.append(f"Prova {c} / {v:g} V / sale {s:g}: {n} riga/righe con temperature mancanti, escluse dal confronto.")
        df = df.drop(incomplete.index)
    df = df.sort_values(['configurazione', 'tensione', 'sale', 'tempo']).reset_index(drop=True)
    return df, avvisi


def elenco_prove(df):
    g = df.groupby(['configurazione', 'tensione', 'sale'])
    r = g.agg(punti=('tempo', 'size'), t_max=('tempo', 'max'), dt=('tempo', lambda x: x.diff().median())).reset_index()
    r['T_ini'] = g[['T1', 'T2', 'T3']].first().mean(axis=1).values
    r['T_fin_media'] = g[['T1', 'T2', 'T3']].last().mean(axis=1).values
    r['id'] = r.apply(lambda x: f"{x['configurazione']} | {x['tensione']:g} V | sale {x['sale']:g}", axis=1)
    return r


def estrai_prova(df, conf, V, sale):
    d = df[(df['configurazione'] == conf) & (df['tensione'] == V) & (np.isclose(df['sale'], sale))]
    return d.sort_values('tempo').reset_index(drop=True)


# ---------------------------------------------------------------------------
# Configurazioni -> posizioni delle polpette
# ---------------------------------------------------------------------------
# 'Middle' e 'Side' (una sola polpetta, al centro o a un lato) sono dedotte dalle temperature
# iniziali (la sonda dentro la polpetta parte piu' fredda) e VANNO CONFERMATE.
OCCUPAZIONE_DEFAULT = {
    'tris': (True, True, True), '3pirp': (True, True, True),
    'symm': (True, False, True), '2psirp': (True, False, True),
    'asymm': (True, True, False), '2pairp': (True, True, False),
    'middle': (False, True, False),
    'side': (True, False, False),
    'rp': (False, False, False), 'omogeneo': (False, False, False), 'purea': (False, False, False),
    'solopurea': (False, False, False), 'solo purea': (False, False, False),
}


def occupazione_default(nome):
    return OCCUPAZIONE_DEFAULT.get(str(nome).strip().lower(), (False, False, False))


# ---------------------------------------------------------------------------
# Ricetta della purea e proprieta' di partenza (letteratura)
# ---------------------------------------------------------------------------
def composizione_da_sale(sale, burro=5.8, acqua=72.5):
    """Burro e acqua fissati; fiocchi = 100 - burro - acqua - sale."""
    fiocchi = 100.0 - burro - acqua - sale
    if fiocchi <= 0:
        raise ValueError("La somma burro + acqua + sale supera il 100%.")
    return dict(w_burro=burro, w_sale=sale, w_fiocchi=fiocchi, w_acqua=acqua)


def sigma_piu_vicina(sale, acqua):
    """Regressione sigma(T) di letteratura piu' vicina per rapporto sale/acqua (Tab. 5.1 e 6.1)."""
    r = sale / acqua
    best, dist = None, 1e9
    for i, c in me.COMPOSIZIONI.items():
        if c['sigma'] is None:
            continue
        d = abs(c['sale'] / c['acqua'] - r)
        if d < dist:
            best, dist = i, d
    return best


def alpha_da_regressione(a, b):
    s20 = 20.0 * a + b
    return a / s20, s20


def sigma_da_s20_alpha(s20, alpha):
    """sigma(T) = s20 * (1 + alpha (T - 20))  ->  (a, b) con sigma = a T + b."""
    a = s20 * alpha
    return a, s20 - 20.0 * a


class Impostazioni:
    """Impostazioni di modello comuni a tutte le prove."""
    def __init__(self, cella='UNISA (10 x 5.1 x 3.7 cm)', Lx_mm=None, tensione_rms=True,
                 ricetta_id=None, burro=5.8, acqua=72.5, sale_da_file=True, regressione_partenza=None,
                 h_modo='naturale_irraggiamento', T_amb=21.0, prop_modo='choi_okos_T'):
        self.cella = cella
        self.Lx_mm = Lx_mm
        self.tensione_rms = tensione_rms
        self.ricetta_id = ricetta_id            # 1-6 (Tab. 4.3) oppure None = personalizzata (burro, acqua)
        self.burro = burro                      # usati se ricetta_id e' None
        self.acqua = acqua
        self.sale_da_file = sale_da_file        # True: sale dal file; False: ricetta esatta della tabella
        self.regressione_partenza = regressione_partenza   # id composizione (1-6) oppure None = automatica
        self.h_modo = h_modo
        self.T_amb = T_amb
        self.prop_modo = prop_modo
        self.occupazione = {}                   # nome configurazione -> (b1, b2, b3)


def composizione_prova(imp, sale):
    """Ricetta della purea per una prova: dict(w_burro, w_sale, w_fiocchi, w_acqua)."""
    if imp.ricetta_id is not None:
        c = me.COMPOSIZIONI[imp.ricetta_id]
        if imp.sale_da_file:
            return composizione_da_sale(sale, c['burro'], c['acqua'])
        return dict(w_burro=c['burro'], w_sale=c['sale'], w_fiocchi=c['fiocchi'], w_acqua=c['acqua'])
    return composizione_da_sale(sale, imp.burro, imp.acqua)


def sigma_partenza(imp, sale):
    """(a, b, etichetta, id composizione) della regressione di partenza per la purea."""
    comp = composizione_prova(imp, sale)
    cid = imp.regressione_partenza
    if cid is None and imp.ricetta_id is not None and not imp.sale_da_file and me.COMPOSIZIONI[imp.ricetta_id]['sigma']:
        cid = imp.ricetta_id
    if cid is None:
        cid = sigma_piu_vicina(comp['w_sale'], comp['w_acqua'])
    a, b = me.COMPOSIZIONI[cid]['sigma'][:2]
    return a, b, f"letteratura, comp. {cid} (σ20 = {20 * a + b:.2f} S/m)", cid


def prop_letteratura(imp, sale):
    """Insieme completo di proprieta' di letteratura (punto di partenza)."""
    a, b, etichetta, cid = sigma_partenza(imp, sale)
    return dict(sigma_m=(a, b), sigma_p=tuple(me.SIGMA_POLPETTE[:2]),
                fk_m=1.0, fk_p=1.0, fc_m=1.0, fc_p=1.0, fh=1.0, etichetta=etichetta, comp_sigma=cid)


# Parametri calibrabili. 'log': parametro positivo stimato su scala logaritmica; 'lin': scala lineare.
PARAMETRI = [
    dict(id='sig_m_s20', nome='σ purea a 20 °C', unita='S/m', tipo='log', min=0.02, max=20.0),
    dict(id='sig_m_alpha', nome='α purea (coefficiente di temperatura di σ)', unita='1/K', tipo='lin', min=0.0, max=0.10),
    dict(id='sig_p_s20', nome='σ polpette a 20 °C', unita='S/m', tipo='log', min=0.02, max=20.0),
    dict(id='sig_p_alpha', nome='α polpette (coefficiente di temperatura di σ)', unita='1/K', tipo='lin', min=0.0, max=0.10),
    dict(id='fk_m', nome='k purea (fattore sul valore Choi-Okos)', unita='-', tipo='log', min=0.5, max=2.0),
    dict(id='fk_p', nome='k polpette (fattore sul valore Choi-Okos)', unita='-', tipo='log', min=0.5, max=2.0),
    dict(id='fc_m', nome='ρ·cp purea (fattore sul valore Choi-Okos)', unita='-', tipo='log', min=0.5, max=2.0),
    dict(id='fc_p', nome='ρ·cp polpette (fattore sul valore Choi-Okos)', unita='-', tipo='log', min=0.5, max=2.0),
    dict(id='fh', nome='h superficie libera (fattore)', unita='-', tipo='log', min=0.2, max=5.0),
]
PARAM_PER_ID = {p['id']: p for p in PARAMETRI}


def valori_da_prop(prop):
    """Valori numerici dei parametri calibrabili per un insieme di proprieta'."""
    al_m, s_m = alpha_da_regressione(*prop['sigma_m'])
    al_p, s_p = alpha_da_regressione(*prop['sigma_p'])
    return dict(sig_m_s20=s_m, sig_m_alpha=al_m, sig_p_s20=s_p, sig_p_alpha=al_p,
                fk_m=prop['fk_m'], fk_p=prop['fk_p'], fc_m=prop['fc_m'], fc_p=prop['fc_p'], fh=prop['fh'])


def applica_valori(prop, vals):
    """Copia di 'prop' con i parametri in 'vals' (dict id -> valore) sostituiti."""
    v = valori_da_prop(prop)
    v.update(vals)
    new = dict(prop)
    new['sigma_m'] = sigma_da_s20_alpha(v['sig_m_s20'], v['sig_m_alpha'])
    new['sigma_p'] = sigma_da_s20_alpha(v['sig_p_s20'], v['sig_p_alpha'])
    for k in ('fk_m', 'fk_p', 'fc_m', 'fc_p', 'fh'):
        new[k] = v[k]
    return new


def _normalizza_prop(prop):
    """Accetta anche una tupla (a, b) = sigma della sola purea (compatibilita')."""
    if isinstance(prop, dict):
        return prop
    a, b = prop
    return dict(sigma_m=(a, b), sigma_p=tuple(me.SIGMA_POLPETTE[:2]), fk_m=1.0, fk_p=1.0, fc_m=1.0, fc_p=1.0, fh=1.0)


# ---------------------------------------------------------------------------
# Mappatura attributi -> modello
# ---------------------------------------------------------------------------
def _geometria_prova(imp, conf):
    occ = imp.occupazione.get(conf, occupazione_default(conf))
    ov = dict(occupazione=list(occ))
    if imp.Lx_mm:
        ov['Lx'] = imp.Lx_mm / 1000.0
    return me.geometria(imp.cella, 'RP (solo purea)', ov), occ


def prepara_prova(d, imp, prop):
    """Costruisce (geo, p, in_polpetta) per una prova a partire dai suoi attributi."""
    prop = _normalizza_prop(prop)
    conf = d['configurazione'].iloc[0]
    V = float(d['tensione'].iloc[0])
    sale = float(d['sale'].iloc[0])
    geo, occ = _geometria_prova(imp, conf)
    in_polpetta = []
    for n in ('P1', 'P2', 'P3'):
        pos = np.array(geo['sonde'][n]).reshape(3, 1)
        in_polpetta.append(float(me.frazione_fase(pos, geo['centri'], geo['R'], 1e-3)[0]) > 0.5)
    T0 = np.array([d['T1'].iloc[0], d['T2'].iloc[0], d['T3'].iloc[0]], dtype=float)
    pur = [T0[i] for i in range(3) if not in_polpetta[i]]
    pol = [T0[i] for i in range(3) if in_polpetta[i]]
    T0_m = float(np.mean(pur)) if pur else float(np.mean(pol))
    T0_p = float(np.mean(pol)) if pol else T0_m
    comp = composizione_prova(imp, sale)
    p = dict(me.PARAMETRI_DEFAULT)
    p.update(
        matrice_a=float(prop['sigma_m'][0]), matrice_b=float(prop['sigma_m'][1]),
        polpetta_a=float(prop['sigma_p'][0]), polpetta_b=float(prop['sigma_p'][1]),
        fk_matrice=prop['fk_m'], fk_polpetta=prop['fk_p'], fc_matrice=prop['fc_m'], fc_polpetta=prop['fc_p'],
        fattore_h=prop['fh'],
        comp_matrice=me.composizione_purea(**comp),
        prop_modo=imp.prop_modo, h_modo=imp.h_modo,
        V_rms=(V if imp.tensione_rms else V / np.sqrt(2.0)),
        T_iniziale=T0_m, T_iniziale_polpetta=T0_p,
        T_amb=imp.T_amb, durata=float(d['tempo'].max()),
    )
    return geo, p, in_polpetta


# ---------------------------------------------------------------------------
# Simulazione e metriche
# ---------------------------------------------------------------------------
_CACHE_MESH = {}


def mesh_per(geo, h_mm):
    chiave = (round(geo['Lx'], 6), round(geo['Ly'], 6), round(geo['Lz'], 6), tuple(round(c[0], 6) for c in geo['centri']),
              round(geo['R'], 6), round(h_mm, 4))
    if chiave not in _CACHE_MESH:
        if len(_CACHE_MESH) > 12:
            _CACHE_MESH.clear()
        _CACHE_MESH[chiave] = me.costruisci_mesh(geo, h_mm / 1000.0)
    return _CACHE_MESH[chiave]


def simula_prova(d, imp, prop, h_mm, dt, progress_callback=None):
    geo, p, in_pol = prepara_prova(d, imp, prop)
    p['dt'] = float(dt)
    mesh, basis, h_eff = mesh_per(geo, h_mm)
    ris = me.simula(geo, mesh, basis, h_eff, p, progress_callback=progress_callback)
    ris['in_polpetta'] = in_pol
    return ris


def confronta(ris, d):
    """Metriche per sonda: simulato interpolato agli istanti sperimentali."""
    t = d['tempo'].values.astype(float)
    out = []
    for i, n in enumerate(('P1', 'P2', 'P3')):
        exp = d[f'T{i + 1}'].values.astype(float)
        sim = np.interp(t, ris['tempo'], ris['T_sonde'][n])
        e = sim - exp
        ss_tot = np.sum((exp - exp.mean()) ** 2)
        out.append(dict(sonda=n, fase='polpetta' if ris['in_polpetta'][i] else 'purea',
                        RMSE=float(np.sqrt(np.mean(e ** 2))), bias=float(np.mean(e)),
                        errore_finale=float(e[-1]), errore_max=float(np.max(np.abs(e))),
                        salto_exp=float(exp[-1] - exp[0]), salto_sim=float(sim[-1] - sim[0]),
                        R2=float(1 - np.sum(e ** 2) / ss_tot) if ss_tot > 0 else np.nan))
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# Calibrazione di un sottoinsieme qualsiasi di proprieta'
# ---------------------------------------------------------------------------
def calibra(prove, imp, prop0, selezionati, limiti=None, fasi=('purea',), h_mm=5.0, dt=10.0,
            max_nfev=14, progress_callback=None):
    """Stima i parametri 'selezionati' (lista di id di PARAMETRI) minimizzando lo scarto
    (simulato - misurato) sulle sonde delle fasi indicate, per le prove date (lista di DataFrame).
    'limiti' = dict id -> (min, max). Restituisce un dict con tabella dei parametri (valore di partenza,
    calibrato, incertezza ~1 sigma, vicinanza ai limiti), correlazioni, RMSE e proprieta' calibrate."""
    if not selezionati:
        raise ValueError("Seleziona almeno un parametro da calibrare.")
    limiti = limiti or {}
    prop0 = _normalizza_prop(prop0)
    v0 = valori_da_prop(prop0)
    pre = []
    for d in prove:
        geo, p, in_pol = prepara_prova(d, imp, prop0)
        sel = [i for i in range(3) if (('polpetta' if in_pol[i] else 'purea') in fasi)]
        if sel:
            pre.append((d, sel))
    if not pre:
        raise ValueError("Nessuna sonda delle fasi scelte nelle prove selezionate: cambia le prove o le fasi dell'obiettivo.")

    info = [PARAM_PER_ID[i] for i in selezionati]
    lb, ub, x0, xs = [], [], [], []
    for pr in info:
        lo, hi = limiti.get(pr['id'], (pr['min'], pr['max']))
        v = v0[pr['id']]
        lo, hi = min(lo, v), max(hi, v)          # il valore di partenza deve stare nei limiti
        if pr['tipo'] == 'log':
            lb.append(np.log(max(lo, 1e-9))); ub.append(np.log(hi)); x0.append(np.log(max(v, 1e-9))); xs.append(1.0)
        else:
            lb.append(lo); ub.append(hi); x0.append(v); xs.append(0.02)
    lb, ub, x0 = np.array(lb), np.array(ub), np.array(x0)
    contatore = {'n': 0}

    def a_valori(x):
        return {pr['id']: (float(np.exp(x[k])) if pr['tipo'] == 'log' else float(x[k])) for k, pr in enumerate(info)}

    def residui(x):
        prop = applica_valori(prop0, a_valori(x))
        res = []
        for d, sel in pre:
            ris = simula_prova(d, imp, prop, h_mm, dt)
            t = d['tempo'].values.astype(float)
            for i in sel:
                res.append(np.interp(t, ris['tempo'], ris['T_sonde'][f'P{i + 1}']) - d[f'T{i + 1}'].values)
        contatore['n'] += 1
        if progress_callback:
            progress_callback(contatore['n'])
        return np.concatenate(res)

    r0 = residui(x0)
    sol = least_squares(residui, x0, bounds=(lb, ub), diff_step=0.03, max_nfev=max_nfev, x_scale=xs)
    val = a_valori(sol.x)

    # incertezze e correlazioni dalla jacobiana al punto finale (approssimazione lineare)
    m, n = len(sol.fun), len(sol.x)
    corr, std_x = None, np.full(n, np.nan)
    if m > n:
        J = sol.jac
        s2 = float(np.sum(sol.fun ** 2) / (m - n))
        try:
            cov = s2 * np.linalg.pinv(J.T @ J)
            std_x = np.sqrt(np.clip(np.diag(cov), 0, None))
            with np.errstate(invalid='ignore', divide='ignore'):
                c = cov / np.outer(std_x, std_x)
            corr = pd.DataFrame(c, index=[p_['id'] for p_ in info], columns=[p_['id'] for p_ in info])
        except np.linalg.LinAlgError:
            pass
    tab = []
    for k, pr in enumerate(info):
        v = val[pr['id']]
        inc = std_x[k] * (v if pr['tipo'] == 'log' else 1.0)
        al_limite = bool(min(sol.x[k] - lb[k], ub[k] - sol.x[k]) < 1e-3 * max(1.0, abs(ub[k] - lb[k])))
        tab.append(dict(id=pr['id'], parametro=pr['nome'], unita=pr['unita'], partenza=v0[pr['id']], calibrato=v,
                        incertezza=float(inc) if np.isfinite(inc) else np.nan, al_limite=al_limite))
    prop = applica_valori(prop0, val)
    prop['etichetta'] = 'calibrata'
    return dict(selezionati=list(selezionati), tabella=tab, prop=prop, corr=corr,
                rmse=float(np.sqrt(np.mean(sol.fun ** 2))), rmse_prima=float(np.sqrt(np.mean(r0 ** 2))),
                n_residui=int(m), n_parametri=int(n), n_chiamate=contatore['n'], fasi=list(fasi))
