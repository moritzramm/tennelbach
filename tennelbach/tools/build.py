"""
3D-Modell (GLB) der Villa Callenberg, Sonnenberg - nach Entwurf Diez 1904 (Maßstab 1:100).
Koordinaten: x = links->rechts (Vorderansicht), y = vorne->hinten, z = oben. Einheiten: Meter.
Vorderfassade bei y=0, Hinterfassade bei y=14.5.
"""
import numpy as np, trimesh
from collections import defaultdict
from shapely.geometry import Polygon, Point, box as sbox, LineString
from shapely import affinity
from shapely.ops import unary_union
from scipy.spatial import HalfspaceIntersection

MITRE = dict(join_style='mitre', mitre_limit=4.0)
G = defaultdict(list)          # einfache Meshes je Material
WALL, TRIM_U, SOCKEL_U, CUT = [], [], [], []   # Boolean-Gruppen
Z = np.array([0, 0, 1.0])

def add(cat, m):
    if m is not None and len(m.faces): G[cat].append(m)

def ext(poly, z0, z1):
    if poly is None or poly.is_empty: return None
    ps = [poly] if poly.geom_type == 'Polygon' else [g for g in poly.geoms if g.geom_type == 'Polygon']
    ms = []
    for p in ps:
        if p.area < 1e-5: continue
        m = trimesh.creation.extrude_polygon(p, z1 - z0)
        m.apply_translation([0, 0, z0]); ms.append(m)
    return trimesh.util.concatenate(ms) if ms else None

def bx(x0, x1, y0, y1, z0, z1):
    return trimesh.creation.box(bounds=[[min(x0,x1), min(y0,y1), z0], [max(x0,x1), max(y0,y1), z1]])

class Face:
    """Fassadenebene: u entlang der Fassade, v = z, w = nach außen."""
    def __init__(s, O, U):
        s.O = np.array(O, float); s.U = np.array(U, float); s.U /= np.linalg.norm(s.U)
        s.W = np.cross(s.U, Z)
        s.M = np.eye(4); s.M[:3, 0] = s.U; s.M[:3, 1] = Z; s.M[:3, 2] = s.W; s.M[:3, 3] = s.O
    def shift(s, dw): return Face(s.O + dw * s.W, s.U)
    def pt(s, u, v, w): return s.O + u * s.U + v * Z + w * s.W
    def ext(s, poly, w0, w1):
        m = ext(poly, w0, w1)
        if m is not None: m.apply_transform(s.M)
        return m
    def box(s, u0, u1, v0, v1, w0, w1): return s.ext(sbox(min(u0,u1), v0, max(u0,u1), v1), w0, w1)

def seg_face(A, B):
    A = np.array([A[0], A[1], 0.0]); B = np.array([B[0], B[1], 0.0])
    return Face(A, B - A), float(np.linalg.norm(B - A))

# ---------------------------------------------------------------- Profile
def bez(p0, p1, p2, p3, n=16):
    t = np.linspace(0, 1, n)[:, None]; P = np.array([p0, p1, p2, p3], float)
    return [tuple(p) for p in ((1-t)**3*P[0] + 3*(1-t)**2*t*P[1] + 3*(1-t)*t*t*P[2] + t**3*P[3])]

def mirror(half):
    right = [tuple(map(float, p)) for p in half]
    left = [(-u, v) for (u, v) in reversed(right[:-1])]
    return Polygon(right + left).buffer(0)

def profile(W, H, style):
    hw = W / 2
    if style == 'front':     # geschweifter Barockgiebel mit Glockenhaube
        half = [(hw, 0), (hw, .10*H)] + bez((hw,.10*H),(hw,.36*H),(.70*hw,.34*H),(.70*hw,.56*H))[1:] \
             + [(.80*hw,.56*H),(.80*hw,.61*H),(.66*hw,.61*H)] \
             + bez((.66*hw,.61*H),(.70*hw,.86*H),(.10*hw,.86*H),(0,H))[1:]
    elif style == 'ogee':    # Kielbogengiebel (rechte Seite)
        half = [(hw, 0), (hw, .08*H)] + bez((hw,.08*H),(hw,.55*H),(.18*hw,.48*H),(0,H))[1:]
    elif style == 'mansard': # Mansardgiebel mit Haube (linke Seite)
        half = [(hw, 0)] + bez((hw,0),(.92*hw,.45*H),(.62*hw,.52*H),(.66*hw,.72*H))[1:] \
             + [(.74*hw,.72*H),(.74*hw,.77*H),(.64*hw,.77*H)] \
             + bez((.64*hw,.77*H),(.45*hw,.80*H),(.12*hw,.86*H),(0,H))[1:]
    elif style == 'bell':
        half = [(hw, 0)] + bez((hw,0),(hw,.55*H),(.12*hw,.6*H),(0,H))[1:]
    elif style == 'barrel':
        R = (hw**2 + H**2) / (2*H)
        a0 = np.arcsin(hw / R); a = np.linspace(a0, 0, 16)
        half = [(R*np.sin(t), H - R + R*np.cos(t)) for t in a]
    elif style == 'tri':
        half = [(hw, 0), (0, H)]
    return mirror(half)

def outline(w, h, top='flat', rise=None):
    if top == 'flat': return sbox(-w/2, 0, w/2, h)
    if top == 'round':
        r = w/2; base = sbox(-w/2, 0, w/2, h - r)
        return unary_union([base, Point(0, h - r).buffer(r, quad_segs=12).intersection(sbox(-w/2, h-r, w/2, h+1))])
    if top == 'seg':
        rise = rise or 0.22*w
        R = ((w/2)**2 + rise**2) / (2*rise)
        base = sbox(-w/2, 0, w/2, h - rise)
        return unary_union([base, Point(0, h - R).buffer(R, quad_segs=24).intersection(sbox(-w/2, h-rise, w/2, h+1))])
    if top == 'chamfer':
        c = min(0.18, w*0.15)
        return Polygon([(-w/2,0),(w/2,0),(w/2,h-c),(w/2-c,h),(-w/2+c,h),(-w/2,h-c)])

def springline(w, h, top, rise=None):
    if top == 'round': return h - w/2
    if top == 'seg': return h - (rise or 0.22*w)
    return h

# ---------------------------------------------------------------- Fenster
def window(F, uc, z0, w, h, top='flat', rise=None, cols=2, transom='auto', frame=0.13,
           hood=None, keystone=False, sill=True, door=False, cut_out=0.15, gd=0.13,
           ears=False, grille=False, muntins=True, cut=True, extra=(), embed=0.0):
    g0 = outline(w, h, top, rise); g = affinity.translate(g0, uc, z0)
    sp = z0 + springline(w, h, top, rise)
    ztop = z0 + h
    if cut: CUT.append(F.ext(g, -0.34, cut_out))
    # Füllung
    if door:
        zl = z0 + 0.48 * (sp - z0)
        low = g.intersection(sbox(uc-w, z0-1, uc+w, zl))
        add('wood', F.ext(low, -gd-0.03, -gd+0.01))
        nl = max(1, cols)
        for i in range(nl):
            a = uc - w/2 + i*w/nl
            for (b0, b1) in [(z0+0.12, z0+0.45*(zl-z0)), (z0+0.55*(zl-z0), zl-0.1)]:
                add('wood2', F.box(a+0.1, a+w/nl-0.1, b0, b1, -gd+0.01, -gd+0.04))
        add('glass', F.ext(g.difference(low), -gd-0.02, -gd))
        add('sash', F.box(uc-w/2, uc+w/2, zl-0.04, zl+0.04, -gd, -gd+0.05))
    else:
        add('glass', F.ext(g, -gd-0.02, -gd))
    inner = g.buffer(-0.045, **MITRE)
    add('sash', F.ext(g.difference(inner), -gd, -gd+0.05))
    bars = []
    for i in range(1, cols):
        u = uc - w/2 + i*w/cols; bars.append(sbox(u-0.03, z0, u+0.03, ztop+1))
    zt = None
    if transom == 'auto' and (sp - z0) > 1.3: zt = z0 + 0.74*(sp - z0)
    elif isinstance(transom, (int, float)): zt = transom
    if zt: bars.append(sbox(uc-w, zt-0.035, uc+w, zt+0.035))
    for e in extra: bars.append(sbox(uc-w, e-0.03, uc+w, e+0.03))
    if muntins and zt:   # Sprossen im Oberlicht
        n = cols*2
        for i in range(1, n):
            if i % 2 == 0: continue
            u = uc - w/2 + i*w/n; bars.append(sbox(u-0.015, zt, u+0.015, ztop+1))
        if top != 'flat' and (ztop - zt) > 0.6:
            bars.append(sbox(uc-w, sp-0.015, uc+w, sp+0.015))
    if bars:
        add('sash', F.ext(unary_union(bars).intersection(g), -gd, -gd+0.05))
    if grille:
        gl = [sbox(uc-w/2+i*0.12-0.01, z0, uc-w/2+i*0.12+0.01, ztop) for i in range(1, int(w/0.12)+1)]
        add('iron', F.ext(unary_union(gl).intersection(g), -0.07, -0.05))
    # Faschen / Rahmen
    if frame > 0:
        fo = g.buffer(frame, **MITRE)
        if ears:
            fo = unary_union([fo, sbox(uc-w/2-frame-0.09, ztop-0.28, uc+w/2+frame+0.09, ztop+frame)])
        add('trim', F.ext(fo.difference(g), -embed, 0.065))
    if sill and not door:
        add('trim', F.box(uc-w/2-frame-0.08, uc+w/2+frame+0.08, z0-frame-0.08, z0-frame+0.02, -0.02-embed, 0.14))
    ft = ztop + frame
    if hood == 'cornice':
        add('trim', F.box(uc-w/2-frame-0.12, uc+w/2+frame+0.12, ft, ft+0.09, 0, 0.15))
        add('trim', F.box(uc-w/2-frame-0.17, uc+w/2+frame+0.17, ft+0.09, ft+0.16, 0, 0.21))
        for s in (-1, 1):
            uu = uc + s*(w/2 + frame/2)
            add('trim', F.box(uu-0.06, uu+0.06, ft-0.34, ft, 0.05, 0.14))
    elif hood == 'arch':
        ring = g.buffer(frame+0.13).difference(g.buffer(frame-0.01)).intersection(sbox(uc-w, sp-0.02, uc+w, ztop+1))
        add('trim', F.ext(ring, -embed, 0.11))
    elif hood in ('pediment', 'ornate'):
        Wp = w + 2*frame + 0.34; rp = 0.34
        seg_o = affinity.translate(profile(Wp, rp, 'barrel'), uc, ft+0.1)
        seg_i = affinity.translate(profile(Wp-0.3, rp-0.12, 'barrel'), uc, ft+0.1)
        add('trim', F.ext(seg_o.difference(seg_i), 0, 0.16))
        add('trim', F.box(uc-Wp/2, uc+Wp/2, ft, ft+0.1, 0, 0.17))
        if hood == 'ornate':
            add('trim', F.ext(Point(uc, ft+0.24).buffer(0.13, quad_segs=6), 0, 0.15))
    if keystone:
        kb = sp + (ztop - sp) - 0.1 if top != 'flat' else ztop - 0.1
        ks = Polygon([(uc-0.09, kb), (uc+0.09, kb), (uc+0.14, ft+0.1), (uc-0.14, ft+0.1)])
        add('trim', F.ext(ks, -embed, 0.12))
    return g

def panel(F, uc, z0, w, h, top='flat', ring=0.07, rosette=False, raised=0.02, holes=()):
    g = affinity.translate(outline(w, h, top), uc, z0)
    for hh in holes: g = g.difference(hh)
    add('stuc', F.ext(g, 0, raised))
    if ring > 0: add('trim', F.ext(g.buffer(ring, **MITRE).difference(g), 0, raised + 0.025))
    if rosette:
        c = g.centroid
        add('trim', F.ext(Point(c.x, c.y).buffer(min(w, h)*0.22, quad_segs=6), 0, raised + 0.06))
        add('trim', F.ext(Point(c.x, c.y).buffer(min(w, h)*0.10, quad_segs=6), 0, raised + 0.1))

# ---------------------------------------------------------------- Geländer
def iron_rail(pts, z0, h, rings=True, posts=True, closed=False):
    pts = list(pts) + ([pts[0]] if closed else [])
    for A, B in zip(pts[:-1], pts[1:]):
        F, L = seg_face(A, B)
        if L < 0.05: continue
        add('iron', F.box(0, L, z0+h-0.05, z0+h, -0.03, 0.03))
        add('iron', F.box(0, L, z0+0.05, z0+0.09, -0.02, 0.02))
        add('iron', F.box(0, L, z0+h-0.22, z0+h-0.19, -0.015, 0.015))
        n = max(2, int(L / 0.12))
        bars = [sbox(i*L/n - 0.009, z0+0.07, i*L/n + 0.009, z0+h-0.03) for i in range(1, n)]
        if rings:
            k = max(1, int(L / 0.55))
            for j in range(k):
                c = (j + 0.5) * L / k
                bars.append(Point(c, z0+h*0.48).buffer(0.15, quad_segs=6).difference(Point(c, z0+h*0.48).buffer(0.115, quad_segs=6)))
        add('iron', F.ext(unary_union(bars), -0.011, 0.011))
        if posts: add('iron', F.box(-0.03, 0.03, z0, z0+h+0.04, -0.03, 0.03))
    if posts:
        F, L = seg_face(pts[-2], pts[-1]); add('iron', F.box(L-0.03, L+0.03, z0, z0+h+0.04, -0.03, 0.03))

def rod(p0, p1, r=0.03, cat='iron', sections=6):
    add(cat, trimesh.creation.cylinder(radius=r, segment=[p0, p1], sections=sections))

def finial(x, y, z, s=1.0, cat='metal'):
    add(cat, trimesh.creation.cylinder(radius=0.07*s, height=0.25*s, sections=8).apply_translation([x, y, z+0.125*s]))
    sp = trimesh.creation.icosphere(subdivisions=1, radius=0.14*s).apply_scale([1, 1, 1.35]).apply_translation([x, y, z+0.42*s])
    add(cat, sp)
    add(cat, trimesh.creation.cone(radius=0.05*s, height=0.7*s, sections=8).apply_translation([x, y, z+0.55*s]))
    add(cat, trimesh.creation.icosphere(subdivisions=1, radius=0.045*s).apply_translation([x, y, z+0.95*s]))

# ---------------------------------------------------------------- Dachkörper
class Frustum:
    def __init__(s, base, top, z0, z1):
        s.b = base; s.t = top; s.z0 = z0; s.z1 = z1
        bx0, by0, bx1, by1 = base; tx0, ty0, tx1, ty1 = top; d = z1 - z0
        s.sl = d / (tx0 - bx0); s.sr = d / (bx1 - tx1); s.sf = d / (ty0 - by0); s.sb = d / (by1 - ty1)
    def h(s, x, y):
        bx0, by0, bx1, by1 = s.b
        if not (bx0 <= x <= bx1 and by0 <= y <= by1): return -1e9
        return min(s.z0 + (x-bx0)*s.sl, s.z0 + (bx1-x)*s.sr, s.z0 + (y-by0)*s.sf, s.z0 + (by1-y)*s.sb, s.z1)
    def mesh(s, zb):
        bx0, by0, bx1, by1 = s.b
        hs = [[-1,0,0,bx0],[1,0,0,-bx1],[0,-1,0,by0],[0,1,0,-by1],[0,0,-1,zb],[0,0,1,-s.z1],
              [-s.sl,0,1,-(s.z0 - s.sl*bx0)], [s.sr,0,1,-(s.z0 + s.sr*bx1)],
              [0,-s.sf,1,-(s.z0 - s.sf*by0)], [0,s.sb,1,-(s.z0 + s.sb*by1)]]
        cx = (s.t[0]+s.t[2])/2; cy = (s.t[1]+s.t[3])/2
        hi = HalfspaceIntersection(np.array(hs, float), np.array([cx, cy, (zb + s.z0)/2]))
        return trimesh.convex.convex_hull(hi.intersections)
    def top_corners(s):
        tx0, ty0, tx1, ty1 = s.t; return [(tx0,ty0),(tx1,ty0),(tx1,ty1),(tx0,ty1)]
    def base_corners(s):
        bx0, by0, bx1, by1 = s.b; return [(bx0,by0),(bx1,by0),(bx1,by1),(bx0,by1)]

ZC = 10.62   # Traufe / Oberkante Hauptgesims
ROOFS = [
    Frustum((-0.5, -0.62, 10.45, 15.0), (4.9, 4.6, 5.6, 9.0), ZC, 17.0),     # Hauptdach mit Plattform (Grate laut DG-Plan ab den Ecken (0,0)/(9.95,0))
    Frustum((9.45, -0.45, 11.92, 5.6), (9.9, 0.4, 11.1, 3.5), ZC, 14.2),     # steiles Dach über Ecknische/Loggia (vorne rechts)
    Frustum((4.1, 5.1, 11.98, 15.0), (7.95, 10.6, 8.1, 12.0), ZC, 15.5),     # Walmdach Flügel hinten rechts
    Frustum((-0.5, 9.0, 3.6, 14.43), (0.6, 11.2, 1.9, 12.4), ZC, 13.4),      # niedriger Dachteil hinten links
]
ROOF_CUT = []   # Dachüberstand vor Giebeln/Zwerchhäusern entfernen
def z_roof(x, y): return max(r.h(x, y) for r in ROOFS)

def zt(y):  # Gelände (Hang steigt nach hinten)
    if y < 0: return max(-0.9, 0.1 + 0.05*y)
    if y <= 14.5: return 0.1 + 1.7*(y/14.5)**2
    return min(3.6, 1.8 + 0.12*(y-14.5))

# ---------------------------------------------------------------- Gauben & Giebel
def dormer(F, uc, sill, w, h, top='flat', hood='bell', cols=2, hoodH=0.75, fin=True, rise=None):
    wf = None
    for wq in np.arange(1.2, -9, -0.01):
        p = F.pt(uc, 0, wq)
        if z_roof(p[0], p[1]) >= sill - 0.14: wf = wq; break
    D = F.shift(wf)
    bw = w + 0.42
    WALL.append(D.box(uc-bw/2, uc+bw/2, sill-0.8, sill+h+0.12, -3.0, 0))
    window(D, uc, sill, w, h, top, rise=rise, cols=cols, frame=0.07, sill=True, transom=None, gd=0.1, cut_out=0.1)
    hb = sill + h + 0.08
    if hood == 'barrel':   # gerade Satteldachgaube statt Tonnenwölbung
        hood = 'tri'; hoodH = max(hoodH * 1.6, 0.45)
    prof = affinity.translate(profile(bw+0.28, hoodH, hood), uc, hb)
    if hood == 'tri':
        add('roof', D.ext(prof, -3.2, 0.16))
    else:                  # Zierhaube als kurzer Vorbau, dahinter Satteldach mit geraden Flächen
        add('roof', D.ext(prof, -0.45, 0.16))
        tri = Polygon([(uc - bw/2 - 0.14, hb), (uc + bw/2 + 0.14, hb), (uc, hb + hoodH * 0.75)])
        add('roof', D.ext(tri, -3.2, -0.44))
    add('trim', D.ext(prof.buffer(0.04, **MITRE).difference(prof).intersection(sbox(-99, hb+0.02, 99, 99)), 0.1, 0.2))
    add('trim', D.box(uc-bw/2-0.16, uc+bw/2+0.16, hb-0.06, hb+0.02, -0.02, 0.2))
    if fin and hood in ('bell', 'tri'):
        p = D.pt(uc, hb + hoodH, 0.0); finial(p[0], p[1], p[2], 0.55)

def gable(F, uc, W, H, style, roofH, base=10.6, w0=-0.35, w1=0.0, back=6.0, fin=1.0, balls=False):
    prof = affinity.translate(profile(W, H, style), uc, base)
    ROOF_CUT.append(F.box(uc-W/2, uc+W/2, base-0.25, base+H+2, w1-0.02, 1.5))
    WALL.append(F.ext(prof, w0, w1))
    cop = prof.buffer(0.08, **MITRE).difference(prof).intersection(sbox(uc-W, base+0.35, uc+W, base+H+1))
    add('trim', F.ext(cop, w0, w1+0.07))
    add('trim', F.box(uc-W/2-0.06, uc+W/2+0.06, base, base+0.22, w0, w1+0.05))
    tri = Polygon([(uc-W/2-0.15, base), (uc+W/2+0.15, base), (uc, base+roofH)])
    add('roof', F.ext(tri, -back, w0+0.15))
    p = F.pt(uc, base+H+0.06, (w0+w1)/2); finial(p[0], p[1], p[2], fin)
    if balls:
        for s in (-1, 1):
            q = F.pt(uc + s*(W/2-0.05), base+0.45, w1-0.12)
            add('trim', trimesh.creation.icosphere(subdivisions=1, radius=0.17).apply_translation(q))
            add('trim', F.box(uc+s*(W/2-0.05)-0.16, uc+s*(W/2-0.05)+0.16, base+0.05, base+0.3, w1-0.3, w1+0.02))
    for s_ in (-1, 1):   # verkröpfte Postamente auf dem Hauptgesims
        add('trim', F.box(uc+s_*W/2-0.2, uc+s_*W/2+0.2, base, base+0.32, w0+0.05, w1+0.12))
    return prof

def volute(F, u, v, w1, r=0.28):
    ring = Point(u, v).buffer(r, quad_segs=8).difference(Point(u, v).buffer(r*0.55, quad_segs=8))
    add('trim', F.ext(ring, w1-0.05, w1+0.08))
    add('trim', F.ext(Point(u, v).buffer(r*0.3, quad_segs=6), w1-0.05, w1+0.1))

# ================================================================= v3: Kalibrierung, Ornament-Übernahme, Detailbauteile
import cv2
from PIL import Image as _PILImage
from shapely.geometry import MultiPolygon
from shapely import ops as _ops

# Ansichten: Pixel = X0 + sx*u ; Zeile = Y0 - sz*z   (Maßstabsleiste 94,6 px/m; Breiten an Grundriss angepasst)
VM = {
    'front': ('D0_sw.jpg', 242.0, 92.9, 2178.7, 94.6),
    'right': ('D0_sw.jpg', 1989.0, 92.9, 2178.7, 94.6),
    'left':  ('E0_sw.jpg', 328.0, 94.5, 2238.7, 94.6),
    'back':  ('E0_sw.jpg', 2351.0, 93.7, 2238.7, 94.6),
}
_IMG = {}
def _gray(f):
    if f not in _IMG:
        _IMG[f] = np.asarray(_PILImage.open('/mnt/user-data/uploads/' + f).convert('L'))
    return _IMG[f]

def trace(view, u0, u1, z0, z1, thick=3, thr=150, min_area=0.0025, simplify=0.004, mask=None):
    """Linien der Ansichtszeichnung im Fenster (u0..u1, z0..z1) als Polygone (u, z) übernehmen."""
    f, X0, sx, Y0, sz = VM[view]
    g = _gray(f)
    px0, px1 = int(X0 + sx * u0), int(X0 + sx * u1)
    py0, py1 = int(Y0 - sz * z1), int(Y0 - sz * z0)
    a = cv2.medianBlur(g[py0:py1, px0:px1].copy(), 3)
    bw = (a < thr).astype(np.uint8) * 255
    if thick > 1:
        bw = cv2.dilate(bw, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (thick, thick)))
    cnts, hier = cv2.findContours(bw, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    if hier is None: return None
    hier = hier[0]
    def tw(c):
        c = c[:, 0, :].astype(float)
        return [((px0 + x - X0) / sx, (Y0 - (py0 + y)) / sz) for x, y in c]
    for i, c in enumerate(cnts):
        if hier[i][3] != -1 or len(c) < 3: continue
        holes = []
        j = hier[i][2]
        while j != -1:
            if len(cnts[j]) >= 3: holes.append(tw(cnts[j]))
            j = hier[j][0]
        try:
            p = Polygon(tw(c), [h for h in holes if len(h) >= 3]).buffer(0)
        except Exception:
            continue
        if p.area >= min_area: polys.append(p.simplify(simplify))
    if not polys: return None
    res = unary_union(polys)
    if mask is not None: res = res.intersection(mask)
    return res

def _ext_local(poly, d0, d1):
    return ext(poly, d0, d1)

def relief(view, poly, s_fn, d0, d1, cat, eps=0.0):
    """Polygone (u, z) als Relief auf eine Oberfläche legen; s_fn(u) = Lage der Oberfläche (y bzw. x)."""
    if poly is None or poly.is_empty: return
    m = _ext_local(poly, d0 + eps, d1 + eps)
    if m is None: return
    v = m.vertices.copy(); u, z, d = v[:, 0], v[:, 1], v[:, 2]
    s = np.array([s_fn(t) for t in u])
    if view == 'front':   X, Y = u, s - d
    elif view == 'right': X, Y = s + d, u
    elif view == 'left':  X, Y = s - d, 14.5 - u
    else:                 X, Y = 11.48 - u, s + d
    m.vertices = np.c_[X, Y, z]
    m.fix_normals()
    add(cat, m)

def const(c): return lambda u: c

# ---------------------------------------------------------------- Detailfenster
def surround(F, g, uc, w, z0, ztop, sp, top, fw=0.14, ears=None, feet=None, embed=0.0, depth=0.085):
    """Profilierte Fensterfasche (dreistufig), optional mit Ohren oben und Füßen unten."""
    fo = g.buffer(fw, **MITRE)
    if ears:
        ew, eh = ears
        if top == 'flat':
            fo = unary_union([fo, sbox(uc - w/2 - fw - ew, ztop + fw - eh, uc + w/2 + fw + ew, ztop + fw)])
        else:
            fo = unary_union([fo, sbox(uc - w/2 - fw - ew, sp - eh, uc + w/2 + fw + ew, sp + 0.03)])
    if feet:
        fwid, fh = feet
        fo = unary_union([fo, sbox(uc - w/2 - fw - fwid, z0 - fw, uc + w/2 + fw + fwid, z0 - fw + fh)])
    mid = g.buffer(fw * 0.6, **MITRE)
    inn = g.buffer(fw * 0.22, **MITRE)
    add('trim', F.ext(fo.difference(mid), -embed, depth * 0.55))
    add('trim', F.ext(mid.difference(inn), -embed, depth))
    add('trim', F.ext(inn.difference(g), -embed, depth * 0.75))
    return fo

def sill(F, uc, wt, zs, embed=0.0, consoles=False, apron=None):
    """Profilierte Sohlbank (zweistufig), optional Konsolen und Brüstungsfeld."""
    add('trim', F.box(uc - wt/2 - 0.06, uc + wt/2 + 0.06, zs - 0.07, zs + 0.005, -embed, 0.17))
    add('trim', F.box(uc - wt/2, uc + wt/2, zs - 0.13, zs - 0.07, -embed, 0.1))
    if consoles:
        for s in (-1, 1):
            u = uc + s * (wt/2 - 0.1)
            prof = Polygon([(0, 0), (0.1, 0), (0.1, -0.06), (0.04, -0.3), (0, -0.3)])
            box_ = F.box(u - 0.06, u + 0.06, zs - 0.43, zs - 0.13, -embed, 0.09)
            add('trim', box_)
            add('trim', F.box(u - 0.07, u + 0.07, zs - 0.47, zs - 0.43, -embed, 0.05))
    if apron:
        a0, aw = apron
        panel(F, uc, a0, aw, zs - 0.16 - a0, ring=0.04, raised=0.015)

def hood_cornice(F, uc, wt, zb, embed=0.0, consoles=True):
    add('trim', F.box(uc - wt/2, uc + wt/2, zb, zb + 0.1, -embed, 0.07))
    add('trim', F.box(uc - wt/2 - 0.05, uc + wt/2 + 0.05, zb + 0.1, zb + 0.15, -embed, 0.13))
    add('trim', F.box(uc - wt/2 - 0.12, uc + wt/2 + 0.12, zb + 0.15, zb + 0.24, -embed, 0.2))
    add('trim', F.box(uc - wt/2 - 0.1, uc + wt/2 + 0.1, zb + 0.24, zb + 0.27, -embed, 0.16))
    if consoles:
        for s in (-1, 1):
            u = uc + s * (wt/2 - 0.02)
            add('trim', F.box(u - 0.06, u + 0.06, zb - 0.3, zb + 0.15, -embed, 0.12))

def win(F, uc, z0, w, h, top='flat', rise=None, cols=2, upper=2, lower_rows=1, transom='auto',
        fw=0.14, ears=None, feet=None, sill_=True, consoles=False, apron=None, hood=None, key=False,
        door=False, blind=False, gd=0.15, embed=0.0, cut=True, cut_out=0.16, grille=False, mull=None, upper_n=None):
    """Fenster mit Laibung, Blendrahmen, Flügeln, Kämpfer, Sprossen und profilierter Fasche."""
    g = affinity.translate(outline(w, h, top, rise), uc, z0)
    sp = z0 + springline(w, h, top, rise)
    ztop = z0 + h
    if cut: CUT.append(F.ext(g, -0.36, cut_out))
    if blind:
        add('stuc', F.ext(g, -0.07, -0.05))
    else:
        zt = None
        if transom == 'auto' and (sp - z0) > 1.3: zt = z0 + 0.7 * (sp - z0)
        elif isinstance(transom, (int, float)): zt = transom
        # Glas / Türfüllung
        if door:
            zl = z0 + 0.45 * ((zt or sp) - z0)
            low = g.intersection(sbox(uc - w, z0 - 1, uc + w, zl))
            add('wood', F.ext(low, -gd - 0.03, -gd + 0.01))
            nl = max(1, cols)
            for i in range(nl):
                a = uc - w/2 + i * w / nl
                for (b0, b1) in [(z0 + 0.1, z0 + 0.45 * (zl - z0)), (z0 + 0.55 * (zl - z0), zl - 0.08)]:
                    if b1 - b0 > 0.08:
                        add('wood2', F.box(a + 0.09, a + w/nl - 0.09, b0, b1, -gd + 0.01, -gd + 0.035))
            add('glass', F.ext(g.difference(low), -gd - 0.02, -gd))
            add('sash', F.box(uc - w/2, uc + w/2, zl - 0.04, zl + 0.04, -gd, -gd + 0.06))
        else:
            add('glass', F.ext(g, -gd - 0.02, -gd))
        # Blendrahmen
        add('sash', F.ext(g.difference(g.buffer(-0.06, **MITRE)), -gd, -gd + 0.07))
        bars = []
        top_lim = zt if zt else ztop + 1
        # Mittelpfosten / Flügel
        mw = 0.07 if mull is None else mull
        for i in range(1, cols):
            u = uc - w/2 + i * w / cols
            bars.append(sbox(u - mw/2, z0, u + mw/2, top_lim))
        # Flügelrahmen je Feld (unterer Teil)
        for i in range(cols):
            a = uc - w/2 + i * w / cols; b = a + w / cols
            cell = sbox(a, z0, b, top_lim).intersection(g)
            if cell.is_empty: continue
            inner = cell.buffer(-0.1, **MITRE)
            if not inner.is_empty:
                add('sash', F.ext(cell.buffer(-0.055, **MITRE).difference(inner), -gd + 0.005, -gd + 0.05))
            for r in range(1, lower_rows):
                zr = z0 + r * ((top_lim if zt else ztop) - z0) / lower_rows
                bars.append(sbox(a, zr - 0.015, b, zr + 0.015))
        if zt:
            add('sash', F.ext(sbox(uc - w, zt - 0.05, uc + w, zt + 0.05).intersection(g), -gd, -gd + 0.085))
            # Oberlicht-Sprossen
            n = upper_n or (upper * cols)
            for i in range(1, n):
                u = uc - w/2 + i * w / n
                bars.append(sbox(u - 0.016, zt, u + 0.016, ztop + 1))
            if top != 'flat' and (ztop - zt) > 0.55:
                bars.append(sbox(uc - w, sp - 0.016, uc + w, sp + 0.016))
            up = g.intersection(sbox(uc - w, zt, uc + w, ztop + 1))
            if not up.is_empty:
                add('sash', F.ext(up.difference(up.buffer(-0.04, **MITRE)), -gd + 0.005, -gd + 0.045))
        if bars:
            add('sash', F.ext(unary_union(bars).intersection(g), -gd, -gd + 0.055))
        if grille:
            gl = [sbox(uc - w/2 + i * 0.12 - 0.012, z0, uc - w/2 + i * 0.12 + 0.012, ztop) for i in range(1, int(w / 0.12) + 1)]
            add('iron', F.ext(unary_union(gl).intersection(g), -0.08, -0.055))
    fo = None
    if fw > 0:
        fo = surround(F, g, uc, w, z0, ztop, sp, top, fw=fw, ears=ears, feet=feet, embed=embed)
    if sill_ and not door:
        sill(F, uc, w + 2 * fw + 0.08, z0 - fw * 0.6, embed=embed, consoles=consoles, apron=apron)
    ft = ztop + fw
    if hood == 'cornice':
        hood_cornice(F, uc, w + 2 * fw + 0.1, ft, embed=embed)
    elif hood == 'arch':
        ring = g.buffer(fw + 0.12).difference(g.buffer(fw - 0.01)).intersection(sbox(uc - w, sp - 0.02, uc + w, ztop + 1))
        add('trim', F.ext(ring, -embed, 0.12))
        ring2 = g.buffer(fw + 0.18).difference(g.buffer(fw + 0.11)).intersection(sbox(uc - w, sp - 0.02, uc + w, ztop + 1))
        add('trim', F.ext(ring2, -embed, 0.07))
    elif hood in ('pediment', 'ornate'):
        Wp = w + 2 * fw + 0.34; rp = 0.34
        seg_o = affinity.translate(profile(Wp, rp, 'barrel'), uc, ft + 0.1)
        seg_i = affinity.translate(profile(Wp - 0.3, rp - 0.12, 'barrel'), uc, ft + 0.1)
        add('trim', F.ext(seg_o.difference(seg_i), -embed, 0.16))
        add('trim', F.box(uc - Wp/2, uc + Wp/2, ft, ft + 0.1, -embed, 0.17))
    if key:
        kb = ztop - 0.1
        ks = Polygon([(uc - 0.09, kb), (uc + 0.09, kb), (uc + 0.14, ft + 0.1), (uc - 0.14, ft + 0.1)])
        add('trim', F.ext(ks, -embed, 0.13))
        add('trim', F.ext(Polygon([(uc - 0.07, kb + 0.05), (uc + 0.07, kb + 0.05), (uc + 0.1, ft + 0.06), (uc - 0.1, ft + 0.06)]), -embed, 0.17))
    return g

def kwin(F, uc, w, ztop, zbot=0.0, zsplit=None, embed=0.0):
    """Hohes Kellerfenster nach Vorderansicht: verglaster Oberteil mit Gitter, unten geschlossenes Feld."""
    zsplit = zsplit or ztop - 0.58
    win(F, uc, zsplit, w, ztop - zsplit, 'flat', cols=2, upper=1, transom=None, fw=0.08, sill_=False,
        grille=True, cut_out=0.18, embed=embed)
    g2 = sbox(uc - w/2, zbot, uc + w/2, zsplit - 0.06)
    add('sockel', F.ext(g2, 0.06, 0.1))
    add('trim', F.ext(sbox(uc - w/2 - 0.08, zbot, uc + w/2 + 0.08, zsplit).difference(g2).difference(sbox(uc - w/2, zsplit - 0.06, uc + w/2, zsplit + 1)), 0.06, 0.14))
    add('trim', F.box(uc - w/2 - 0.1, uc + w/2 + 0.1, zsplit - 0.08, zsplit, 0.05, 0.18))

# ---------------------------------------------------------------- Rustika-Sockel mit Stoßfugen
def rustica(S, z_courses, block=1.15, proj=0.09, joint=0.035):
    line = S.exterior
    L = line.length
    out = []
    for k, (za, zb) in enumerate(z_courses):
        off = 0.0 if k % 2 == 0 else block / 2
        s = -off
        while s < L:
            a, b = max(0.0, s + joint / 2), min(L, s + block - joint / 2)
            if b - a > 0.08:
                seg = _ops.substring(line, a, b)
                strip = seg.buffer(proj + 0.3, cap_style='flat', join_style='mitre')
                blk = S.buffer(proj, **MITRE).difference(S.buffer(-0.25, **MITRE)).intersection(strip)
                if not blk.is_empty and blk.area > 1e-4:
                    out.append(ext(blk, za + joint / 2, zb - joint / 2))
            s += block
    return out

def from_points(half):
    """Halbprofil [(hw, z), ...] von unten rechts bis Scheitel -> symmetrisches Polygon (u relativ, z absolut)."""
    right = [(float(a), float(b)) for a, b in half]
    left = [(-a, b) for (a, b) in reversed(right[:-1])]
    return Polygon(right + left).buffer(0)

def smooth(pts, n=6):
    """Punktfolge mit Catmull-Rom glätten."""
    P = np.array(pts, float); out = []
    for i in range(len(P) - 1):
        p0 = P[max(i - 1, 0)]; p1 = P[i]; p2 = P[i + 1]; p3 = P[min(i + 2, len(P) - 1)]
        for t in np.linspace(0, 1, n, endpoint=False):
            t2, t3 = t * t, t * t * t
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    out.append(P[-1])
    return [tuple(p) for p in out]

def gable_v3(F, uc, half, roof_half=None, back=4.0, wall_w0=-0.35, fin=None, coping=0.09, ridge=None, hwb=None):
    """Ziergiebel (Wand nach gemessenem Halbprofil), dahinter ein normales Satteldach mit geraden Flächen."""
    prof = affinity.translate(from_points(half), uc, 0)
    WALL.append(F.ext(prof, wall_w0, 0.0))
    zb = prof.bounds[1]
    cop = prof.buffer(coping, **MITRE).difference(prof).intersection(sbox(uc - 9, zb + 0.3, uc + 9, 99))
    add('trim', F.ext(cop, wall_w0, 0.07))
    hw0 = max(a for a, b in half)
    hwb = hwb or hw0 + 0.25
    ridge = ridge or ZC + hwb * 1.2
    tri = Polygon([(uc - hwb, zb - 0.08), (uc + hwb, zb - 0.08), (uc, ridge)])
    add('roof', F.ext(tri, -back, wall_w0 + 0.02))
    # Firstblech
    add('metal', F.ext(Polygon([(uc - 0.07, ridge - 0.06), (uc + 0.07, ridge - 0.06), (uc, ridge + 0.05)]), -back, wall_w0 + 0.02))
    ROOF_CUT.append(F.box(prof.bounds[0], prof.bounds[2], zb - 0.3, prof.bounds[3] + 2, -0.02, 1.6))
    if fin:
        p = F.pt(uc, prof.bounds[3] + 0.02, wall_w0 / 2); finial(p[0], p[1], p[2], fin)
    return prof

def ball(p, r=0.17):
    add('trim', trimesh.creation.icosphere(subdivisions=2, radius=r).apply_translation(p))

def capital(F, u, wdt=0.42, z0=9.72, z1=10.52, proj=0.12):
    """Pilasterkopf/Konsolblock im Fries mit Tropfen."""
    add('trim', F.box(u - wdt/2, u + wdt/2, z0, z1, 0, proj))
    add('trim', F.box(u - wdt/2 - 0.03, u + wdt/2 + 0.03, z0 + 0.12, z0 + 0.2, 0, proj + 0.03))
    for s in (-1, 1):
        add('trim', F.box(u + s * wdt * 0.25 - 0.04, u + s * wdt * 0.25 + 0.04, z0 - 0.09, z0, 0, proj - 0.03))

def gframe(F, u0, u1, z0, z1, holes, ears=None, feet=None, depth=0.075, embed=0.0):
    """Gemeinsame Fasche um eine Fenstergruppe (zweistufig), Öffnungen ausgespart."""
    outer = sbox(u0, z0, u1, z1)
    if ears:
        ew, eh = ears; outer = unary_union([outer, sbox(u0 - ew, z1 - eh, u1 + ew, z1)])
    if feet:
        fw_, fh = feet; outer = unary_union([outer, sbox(u0 - fw_, z0, u1 + fw_, z0 + fh)])
    H = unary_union(holes)
    add('trim', F.ext(outer.difference(H.buffer(0.07, **MITRE)), -embed, depth * 0.6))
    add('trim', F.ext(H.buffer(0.07, **MITRE).difference(H).intersection(outer), -embed, depth))

# ================================================================= GEOMETRIE (v3: an kalibrierten Ansichten gemessen)
ZC = 10.82
ROOFS[:] = [
    Frustum((-0.5, -0.62, 10.45, 15.0), (4.9, 4.6, 5.6, 9.0), ZC, 17.0),     # Hauptdach mit Plattform
    Frustum((4.1, 2.7, 11.98, 15.0), (7.95, 6.35, 8.1, 12.0), ZC, 15.2),     # Walmdach rechter Flügel inkl. Turm (Traufe vor der Turmfront)
]

P_sq = Polygon([(0,0),(9.95,0),(9.95,3.2),(11.48,3.2),(11.48,14.5),(4.72,14.5),(4.72,13.93),(0,13.93)])
P_eg = Polygon([(0,0),(9.95,0),(9.95,3.2),(10.85,3.2),(11.48,3.85),(11.48,14.5),(4.72,14.5),(4.72,13.93),(0,13.93)])
NOTCH = sbox(9.95, 0, 11.48, 3.2)
P_full = unary_union([P_sq, NOTCH])
P = P_sq
BC = np.array([2.62, 0.48]); BR = 1.75
a0, a1 = np.radians(-144.8), np.radians(-35.2)
arc = [tuple(BC + BR*np.array([np.cos(a), np.sin(a)])) for a in np.linspace(a0, a1, 41)]
B1 = Polygon([(1.19, 0.25), (1.19, -0.53)] + arc[1:-1] + [(4.05, -0.53), (4.05, 0.25)]).buffer(0)
def bay_face(deg, shift=0.0):
    a = np.radians(deg); n = np.array([np.cos(a), np.sin(a)])
    p = BC + (BR + shift) * n
    return Face((p[0], p[1], 0), (-np.sin(a), np.cos(a), 0))
def bay_y(x, off=0.0):
    """Vorderkante des Erkers (y) an Stelle x, um off nach außen versetzt."""
    R = BR + off
    if 1.19 - off <= x <= 4.05 + off:
        dx = x - BC[0]
        if abs(dx) < R * np.sin(np.radians(54.8)):
            return BC[1] - np.sqrt(max(R*R - dx*dx, 0))
        return -0.53 - off * 0.0
    return 0.0
BAY_ANG = (-125.0, -90.0, -55.0)

F_front = Face((0, 0, 0), (1, 0, 0))
F_ns    = Face((9.95, 0, 0), (0, 1, 0))
F_nf    = Face((0, 3.2, 0), (1, 0, 0))
F_ch, L_ch = seg_face((10.85, 3.2), (11.48, 3.85))
F_right = Face((11.48, 0, 0), (0, 1, 0))
F_back  = Face((11.48, 14.5, 0), (-1, 0, 0))
F_backL = Face((11.48, 13.93, 0), (-1, 0, 0))
F_left  = Face((0, 14.5, 0), (0, -1, 0))

# --- Mauerwerk
WALL.append(ext(P_eg, -1.0, 6.02))
WALL.append(ext(P_sq, 6.0, 10.75))
WALL.append(ext(B1, -1.0, 6.0))

# --- Sockel: Rustika-Quader mit Stoßfugen, profiliertes Sockelgesims
S = unary_union([P_full, B1])
SOCKEL_U.append(ext(S.buffer(0.05, **MITRE), -1.0, 1.98))
courses = [(-0.9, -0.42), (-0.42, 0.06), (0.06, 0.54), (0.54, 1.02), (1.02, 1.5), (1.5, 1.98)]
SOCKEL_U.extend(rustica(S, courses, block=1.15, proj=0.09, joint=0.035))
SOCKEL_U.append(ext(S.buffer(0.11, **MITRE), 1.98, 2.08))
SOCKEL_U.append(ext(S.buffer(0.16, **MITRE), 2.08, 2.18))

# --- Gurtgesims (nur vorne/rechts/Nische), Architrav, Hauptgesims (profiliert)
def ring(poly, o_out, o_in=-0.25): return poly.buffer(o_out, **MITRE).difference(poly.buffer(o_in, **MITRE))
GURT_ZONE = sbox(0.32, -2, 13, 13.6)
TRIM_U.append(ext(ring(P, 0.08).intersection(GURT_ZONE), 6.0, 6.1))
TRIM_U.append(ext(ring(P, 0.12).intersection(GURT_ZONE), 6.1, 6.22))
TRIM_U.append(ext(ring(P, 0.06).intersection(GURT_ZONE), 6.22, 6.3))
TRIM_U.append(ext(ring(P, 0.04), 9.98, 10.12))
TRIM_U.append(ext(ring(P, 0.07), 10.06, 10.12))
TRIM_U.append(ext(ring(P, 0.1), 10.5, 10.56))
TRIM_U.append(ext(ring(P, 0.17), 10.56, 10.62))
TRIM_U.append(ext(ring(P, 0.42), 10.62, 10.76))
TRIM_U.append(ext(ring(P, 0.46), 10.76, 10.82))
# Eck- und Giebelpilaster (Lisenen) bis zum Architrav
coords = list(P.exterior.coords)[:-1]; n = len(coords)
for i, (vx, vy) in enumerate(coords):
    a = np.array(coords[i-1]); b = np.array([vx, vy]); c = np.array(coords[(i+1) % n])
    cr = (b-a)[0]*(c-b)[1] - (b-a)[1]*(c-b)[0]
    if cr > 0 and not (abs(vx-11.48) < 0.01 and abs(vy-3.2) < 0.01):
        sq = sbox(vx-0.32, vy-0.32, vx+0.32, vy+0.32)
        TRIM_U.append(ext(P.buffer(0.045, **MITRE).intersection(sq), 2.18, 9.98))
TRIM_U.append(ext(sbox(4.5, -0.045, 4.82, 0.1), 2.18, 9.98))
# Pilasterköpfe / Friesblöcke (in allen Ansichten an Ecken und unter den Giebelschultern)
for (F, u) in [(F_front, 0.16), (F_front, 4.66), (F_front, 9.8), (F_nf, 11.32),
               (F_right, 3.36), (F_right, 5.7), (F_right, 9.57), (F_right, 14.34),
               (F_left, 0.73), (F_left, 5.4), (F_left, 10.46), (F_left, 14.34),
               (F_back, 0.16), (F_back, 6.6), (F_backL, 6.92), (F_backL, 11.32)]:
    capital(F.shift(0.045), u)

# --- Terrasse in der offenen Ecknische (über dem Keller), Geländer durchgehend
iron_rail([(9.97, 0.04), (11.44, 0.04), (11.44, 3.18)], 2.18, 0.9)

# =========================================================== VORDERANSICHT
# Salon-Erker: Gesimse, Brüstung, Ornamentfries, Fenster
TRIM_U.append(ext(B1.buffer(0.16).difference(P), 5.95, 6.3))
TRIM_U.append(ext(B1.buffer(0.1).difference(B1.buffer(-0.2)).intersection(sbox(-5, -5, 20, -0.02)), 2.84, 3.0))
TRIM_U.append(ext(B1.buffer(0.14).difference(B1.buffer(-0.2)).intersection(sbox(-5, -5, 20, -0.02)), 2.78, 2.84))
par = B1.buffer(0.08).difference(B1.buffer(-0.2)).intersection(sbox(-5, -5, 20, -0.02))
TRIM_U.append(ext(par, 6.3, 7.3))
TRIM_U.append(ext(B1.buffer(0.15).difference(B1.buffer(-0.26)).intersection(sbox(-5, -5, 20, -0.02)), 7.3, 7.38))
TRIM_U.append(ext(B1.buffer(0.19).difference(B1.buffer(-0.26)).intersection(sbox(-5, -5, 20, -0.02)), 7.38, 7.45))
for ang, ww in zip(BAY_ANG, (0.8, 0.95, 0.8)):
    F = bay_face(ang)
    win(F, 0, 3.02, ww, 2.4, 'round', cols=2, upper=1, transom=4.5, fw=0.09, sill_=False, embed=0.1)
    kwin(F, 0, ww - 0.05, 1.45, 0.0, 0.85, embed=0.08)
for ang in (-107.5, -72.5):
    F = bay_face(ang, -0.03)
    add('trim', F.box(-0.12, 0.12, 3.0, 5.5, -0.05, 0.08))
    add('stuc', F.box(-0.06, 0.06, 3.15, 4.4, -0.05, 0.095))
    add('trim', F.box(-0.14, 0.14, 5.5, 5.58, -0.05, 0.11))
# Ornamente aus der Zeichnung (Fries, Brüstungskartusche, Brüstungsfeld unten)
relief('front', trace('front', 1.12, 4.18, 5.5, 6.0, thick=4), lambda u: bay_y(u, 0.0), 0.0, 0.035, 'trim', eps=0.005)
relief('front', trace('front', 1.12, 4.18, 6.4, 7.27, thick=4), lambda u: bay_y(u, 0.08), 0.0, 0.035, 'trim', eps=0.005)
relief('front', trace('front', 2.2, 3.05, 2.2, 2.78, thick=3), lambda u: bay_y(u, 0.0), 0.0, 0.03, 'trim', eps=0.005)

# Obergeschoss über dem Erker: Fenster + Balkontür in gemeinsamer Rahmung
gl = win(F_front, 2.08, 7.35, 0.84, 1.71, 'flat', cols=2, upper=3, upper_n=3, transom=8.36, fw=0.0, sill_=False)
gd_ = win(F_front, 3.19, 7.3, 0.83, 1.76, 'flat', cols=2, upper=3, upper_n=3, transom=8.36, fw=0.0, sill_=False, door=True)
gframe(F_front, 1.41, 3.89, 7.3, 9.34, [gl, gd_])
add('trim', F_front.box(1.96, 3.33, 9.34, 9.43, 0, 0.08))
hood_cornice(F_front, 2.65, 2.55, 9.43, consoles=False)
for u in (1.535, 3.745, 2.635):
    add('stuc', F_front.box(u - 0.06, u + 0.06, 7.55, 8.75, 0.045, 0.06))

# Obergeschoss Mitte: Doppelfenster mit Ohrenrahmung
g1 = win(F_front, 6.475, 7.27, 1.08, 1.85, 'flat', cols=2, upper=3, upper_n=3, transom=8.22, fw=0.0, sill_=False)
g2 = win(F_front, 7.915, 7.27, 1.08, 1.85, 'flat', cols=2, upper=3, upper_n=3, transom=8.22, fw=0.0, sill_=False)
gframe(F_front, 5.68, 8.81, 7.12, 9.34, [g1, g2], ears=(0.08, 0.18), feet=(0.05, 0.25))
add('stuc', F_front.box(7.15, 7.29, 7.6, 8.75, 0.045, 0.07))
sill(F_front, 7.245, 3.2, 7.12)

# Erdgeschoss Mitte (Herrenzimmer): Segmentbogenfenster mit Ohren, Schlussstein-Kartusche, Tropfen
win(F_front, 7.385, 3.0, 1.99, 2.3, 'seg', rise=0.5, cols=3, upper=2, transom=4.38, fw=0.28,
    ears=(0.06, 0.2), feet=(0.05, 0.35), sill_=True, hood='arch')
relief('front', trace('front', 7.1, 7.7, 5.25, 5.92, thick=4), const(0.0), 0.0, 0.18, 'trim')
relief('front', trace('front', 7.0, 7.75, 2.45, 2.74, thick=3), const(0.0), 0.0, 0.05, 'trim')
for u in (6.14, 8.46):
    kwin(F_front, u, 1.02, 1.4, 0.0, 0.82)

# Turm: EG-Fenster (Stirnseite + Abschrägung), Bekrönungen, Loggia im OG
win(F_nf, 10.42, 2.92, 0.62, 2.08, 'round', cols=1, upper=2, transom=4.27, fw=0.07, sill_=True)
win(F_ch, L_ch/2, 2.92, 0.48, 2.08, 'round', cols=1, upper=2, transom=4.27, fw=0.07, sill_=True)
def tower_s(u):
    return 3.2 if u <= 10.85 else 3.2 + (u - 10.85) * (0.65 / 0.63)
relief('front', trace('front', 10.0, 11.45, 5.05, 5.65, thick=4), tower_s, 0.0, 0.04, 'trim', eps=0.005)
for u in (10.42, 11.17):
    kwin(F_front, u, 0.5, 1.4, 0.0, 0.75)
CUT.append(bx(10.1, 11.62, 3.05, 5.15, 6.35, 9.3))
add('trim', ext(sbox(10.1, 3.2, 11.48, 5.15), 6.3, 6.36))
for (x0, x1, y0, y1) in [(10.1, 11.48, 3.2, 3.45), (11.23, 11.48, 3.2, 5.15)]:
    add('trim', bx(x0, x1, y0, y1, 6.36, 7.28)); add('trim', bx(x0-0.03, x1+0.04, y0-0.05, y1, 7.28, 7.38))
relief('front', trace('front', 10.02, 11.45, 6.42, 7.27, thick=4), const(3.2), 0.0, 0.035, 'trim', eps=0.002)
relief('right', trace('right', 3.3, 4.25, 6.3, 7.22, thick=4), const(11.48), 0.0, 0.035, 'trim', eps=0.002)
panel(F_right.shift(0.01), 4.85, 6.4, 1.05, 0.78, ring=0.04, raised=0.015)
for (x, y) in [(11.38, 3.3), (10.75, 3.3), (11.38, 4.2), (11.38, 5.05)]:
    add('trim', trimesh.creation.cylinder(radius=0.07, segment=[[x, y, 7.38], [x, y, 8.95]], sections=12))
    add('trim', bx(x-0.1, x+0.1, y-0.1, y+0.1, 7.38, 7.48))
    add('trim', bx(x-0.11, x+0.11, y-0.11, y+0.11, 8.95, 9.05))
# Bögen zwischen den Loggiasäulen
for (F, a, b) in [(F_nf.shift(0.02), 10.12, 10.68), (F_nf.shift(0.02), 10.82, 11.31),
                  (F_right.shift(-0.05), 3.37, 4.13), (F_right.shift(-0.05), 4.27, 4.98)]:
    span = b - a; R = (span/2)**2/(2*0.22) + 0.11
    plate = sbox(a, 8.7, b, 9.3).difference(Point((a+b)/2, 8.92 - (R - 0.22) + 0.0).buffer(R, quad_segs=24).intersection(sbox(a, 0, b, 9.14)))
    add('trim', F.ext(plate, -0.08, 0.04))
can = sbox(9.95, 2.82, 11.88, 5.4).difference(sbox(9.0, 3.2, 11.48, 6.0))
add('roof', ext(can, 9.3, 9.42))
add('roof', ext(sbox(9.95, 2.95, 11.75, 5.4).difference(sbox(9.0, 3.25, 11.43, 6.0)), 9.42, 9.62))
add('trim', ext(can.buffer(0.03, **MITRE).difference(can), 9.25, 9.33))
F_ls = Face((10.1, 0, 0), (0, 1, 0))   # Innenwand der Loggia (x = 10,1, Blick nach rechts) – Balkontür laut Seitenansicht
win(F_ls, 4.62, 6.37, 0.85, 2.55, 'flat', cols=2, upper=2, transom=8.4, fw=0.06, door=True, sill_=False, gd=0.1, cut_out=0.05)

# =========================================================== RECHTE SEITE
panel(F_ns, 1.75, 3.0, 1.7, 2.85, 'chamfer')
panel(F_ns, 1.75, 6.32, 1.7, 3.18, 'round')
panel(F_right, 4.97, 2.7, 0.95, 2.95, 'chamfer')
# OG-Bogenfenster mit Ohren und Volutenkonsolen
win(F_right, 7.63, 7.15, 1.66, 2.14, 'seg', rise=0.38, cols=2, upper=2, transom=8.5, fw=0.13,
    ears=(0.06, 0.22), sill_=True)
relief('right', trace('right', 6.35, 6.8, 6.95, 7.95, thick=4), const(11.48), 0.0, 0.1, 'trim')
relief('right', trace('right', 8.42, 8.85, 6.95, 7.95, thick=4), const(11.48), 0.0, 0.1, 'trim')
# EG-Zwillingsfenster unter gemeinsamer Bogenfasche
t1 = win(F_right, 7.07, 3.0, 0.94, 2.29, 'round', cols=2, upper=1, transom=4.58, fw=0.0, sill_=False)
t2 = win(F_right, 8.22, 3.0, 0.94, 2.29, 'round', cols=2, upper=1, transom=4.58, fw=0.0, sill_=False)
fr = unary_union([t1.buffer(0.14, **MITRE), t2.buffer(0.14, **MITRE), sbox(7.5, 3.0, 7.8, 4.9)])
add('trim', F_right.ext(fr.difference(unary_union([t1, t2]).buffer(0.06, **MITRE)), 0, 0.05))
add('trim', F_right.ext(unary_union([t1, t2]).buffer(0.06, **MITRE).difference(unary_union([t1, t2])), 0, 0.085))
add('stuc', F_right.box(7.58, 7.71, 3.25, 4.3, 0.05, 0.07))
sill(F_right, 7.645, 2.55, 2.98)
for u in (4.8, 6.58, 8.4):
    win(F_right, u, 0.83, 0.78, 0.79, 'flat', cols=2, upper=1, transom=None, fw=0.08, sill_=False, grille=True, cut_out=0.18)
# Veranda (Speisezimmer) – Maße nach Seitenansicht
vx0, vx1, vy0, vy1 = 11.48, 12.89, 10.05, 13.05
win(F_right, 11.6, 2.16, 1.6, 3.17, 'round', cols=2, upper=2, transom=4.38, fw=0.25, door=True)
relief('right', trace('right', 11.3, 11.9, 5.3, 5.95, thick=4), const(11.48), 0.0, 0.14, 'trim')
gc = win(F_right, 11.57, 7.15, 0.93, 2.2, 'round', cols=2, upper=2, transom=8.5, fw=0.0, door=True, sill_=False)
gs1 = win(F_right, 10.45, 7.15, 0.81, 2.05, 'flat', cols=2, upper=3, upper_n=3, transom=8.5, fw=0.0, sill_=False)
gs2 = win(F_right, 12.7, 7.15, 0.77, 2.05, 'flat', cols=2, upper=3, upper_n=3, transom=8.5, fw=0.0, sill_=False)
gframe(F_right, 9.83, 13.33, 7.1, 9.45, [gc, gs1, gs2])
seg = affinity.translate(profile(1.6, 0.25, 'barrel'), 11.57, 9.45)
add('trim', F_right.ext(seg, 0, 0.075))
for u in (10.97, 12.17):
    add('stuc', F_right.box(u - 0.06, u + 0.06, 7.45, 8.9, 0.045, 0.07))
relief('right', trace('right', 9.78, 10.02, 7.3, 7.8, thick=4), const(11.48), 0.0, 0.1, 'trim')
relief('right', trace('right', 13.12, 13.38, 7.3, 7.8, thick=4), const(11.48), 0.0, 0.1, 'trim')
SOCKEL_U.append(bx(vx0-0.1, vx1, vy0, vy1, -0.5, 2.0))
SOCKEL_U.append(bx(vx0-0.1, vx1+0.06, vy0-0.06, vy1+0.06, 2.0, 2.16))
for y in (vy0+0.08, vy1-0.08):
    rod([vx1-0.06, y, 2.16], [vx1-0.06, y, 5.75], r=0.04, sections=10)
    add('iron', trimesh.creation.cylinder(radius=0.075, segment=[[vx1-0.06, y, 2.16], [vx1-0.06, y, 2.36]], sections=10))
    add('iron', trimesh.creation.cylinder(radius=0.07, segment=[[vx1-0.06, y, 5.55], [vx1-0.06, y, 5.75]], sections=10))
add('iron', bx(vx0, vx1, vy0, vy0+0.08, 5.95, 6.05)); add('iron', bx(vx0, vx1, vy1-0.08, vy1, 5.95, 6.05))
add('iron', bx(vx1-0.08, vx1, vy0, vy1, 5.95, 6.05))
add('trim', bx(vx0, vx1+0.06, vy0-0.06, vy1+0.06, 6.05, 6.28))
# Schmiedeeisen aus den Ansichten: Konsolbögen, Balkongeländer (bombiert), Brüstungsgitter EG
br_mask = sbox(10.0, 5.5, 13.15, 6.05).difference(sbox(11.2, 5.0, 11.95, 5.98))
relief('right', trace('right', 10.0, 13.15, 5.55, 6.03, thick=3, min_area=0.0015, mask=br_mask), const(vx1 - 0.06), -0.012, 0.012, 'iron')
relief('right', trace('right', 10.0, 13.15, 6.28, 7.08, thick=3, min_area=0.0015), const(vx1 + 0.07), -0.012, 0.012, 'iron')
relief('right', trace('right', 10.15, 12.95, 2.2, 2.84, thick=3, min_area=0.0015), const(vx1 - 0.02), -0.012, 0.012, 'iron')
relief('front', trace('front', 11.5, 12.95, 6.28, 7.08, thick=3, min_area=0.0015), const(vy0 - 0.02), -0.012, 0.012, 'iron')
relief('front', trace('front', 11.5, 12.95, 2.2, 2.84, thick=3, min_area=0.0015), const(vy0 + 0.02), -0.012, 0.012, 'iron')
relief('back', trace('back', -1.5, 0.0, 6.28, 7.08, thick=3, min_area=0.0015), const(vy1 + 0.02), -0.012, 0.012, 'iron')
relief('back', trace('back', -1.5, 0.0, 2.2, 2.84, thick=3, min_area=0.0015), const(vy1 - 0.02), -0.012, 0.012, 'iron')
for (pts, z0, h) in [([(vx0, vy0+0.02), (vx1+0.07, vy0+0.02), (vx1+0.07, vy1-0.02), (vx0, vy1-0.02)], 6.28, 0.82),
                     ([(vx0, vy0+0.02), (vx1-0.02, vy0+0.02), (vx1-0.02, vy1-0.02), (vx0, vy1-0.02)], 2.16, 0.68)]:
    for A, B in zip(pts[:-1], pts[1:]):
        Fs, Ls = seg_face(A, B)
        add('iron', Fs.box(0, Ls, z0 + h - 0.05, z0 + h, -0.03, 0.03)); add('iron', Fs.box(0, Ls, z0 + 0.03, z0 + 0.07, -0.02, 0.02))

# =========================================================== RÜCKSEITE
hb = []
for u in (2.75, 4.03):
    hb.append(win(F_back, u, 3.17, 0.99, 2.45, 'round', cols=2, upper=1, transom=4.83, fw=0.1, key=True, sill_=False).buffer(0.14))
sill(F_back, 3.39, 2.5, 3.12)
hb.append(sbox(2.0, 2.9, 4.8, 3.2))
panel(F_back, 3.35, 2.26, 4.59, 4.49, 'chamfer', ring=0.06, raised=0.015, holes=hb)
g_og = win(F_back, 3.38, 7.0, 1.86, 2.4, 'seg', rise=0.38, cols=3, upper=2, transom=8.55, fw=0.12, ears=(0.05, 0.2), sill_=True)
panel(F_back, 3.35, 7.61, 4.59, 2.06, 'chamfer', ring=0.06, raised=0.015, holes=[g_og.buffer(0.3)])
for u in (1.8, 4.87):
    win(F_back, u, 1.62, 0.72, 0.28, 'flat', cols=2, transom=None, fw=0.05, grille=True, cut_out=0.16, sill_=False)
for u in (7.7, 10.0):
    win(F_backL, u, 1.69, 0.75, 0.26, 'flat', cols=2, transom=None, fw=0.05, grille=True, cut_out=0.16, sill_=False)
win(F_backL, 7.76, 3.21, 0.9, 2.6, 'round', cols=2, upper=1, transom=4.95, fw=0.09, sill_=True)
add('trim', F_backL.box(7.04, 8.47, 4.95, 5.15, 0, 0.14))
win(F_backL, 7.76, 6.95, 0.9, 2.05, 'round', cols=2, upper=1, transom=8.25, fw=0.09)
win(F_backL, 10.1, 6.1, 1.23, 1.85, 'flat', cols=2, upper=2, transom=7.28, fw=0.1, ears=(0.05, 0.18), cut_out=0.4)
win(F_backL, 10.25, 3.32, 0.65, 1.42, 'round', cols=1, upper=2, transom=4.27, fw=0.08)
# Zwerchhaus hinten links
zw = affinity.translate(outline(1.75, 1.95, 'seg', rise=0.4), 10.0, 9.4)
WALL.append(F_backL.ext(zw, -1.5, 0.0))
ROOF_CUT.append(F_backL.box(9.12, 10.88, 10.4, 11.5, -0.02, 1.2))
add('trim', F_backL.ext(zw.buffer(0.08, **MITRE).difference(zw).intersection(sbox(0, 10.75, 20, 20)), -1.5, 0.08))
add('roof', F_backL.ext(Polygon([(8.95, 11.05), (11.05, 11.05), (10.0, 11.95)]), -2.2, 0.15))
win(F_backL, 10.0, 9.08, 1.2, 2.01, 'round', cols=2, upper=1, transom=10.35, fw=0.09, cut_out=0.65, sill_=False)

# =========================================================== LINKE SEITE
panel(F_left, 2.2, 3.1, 1.6, 6.5, 'round', ring=0.1, raised=0.02)        # hohe Blendarkade
win(F_left, 5.44, 7.12, 1.0, 2.24, 'flat', cols=2, upper=2, transom=8.67, fw=0.24, ears=(0.05, 0.2), feet=(0.04, 0.35))
win(F_left, 7.86, 7.5, 0.85, 1.95, 'round', cols=2, upper=1, transom=8.71, fw=0.2, ears=(0.05, 0.2))
relief('left', trace('left', 7.55, 8.2, 9.42, 10.05, thick=4), const(0.0), 0.0, 0.16, 'trim')
win(F_left, 11.27, 7.12, 1.13, 2.28, 'flat', cols=2, upper=2, transom=8.71, fw=0.24, ears=(0.05, 0.2), feet=(0.04, 0.35))
e1 = win(F_left, 4.57, 3.88, 0.87, 1.87, 'round', cols=2, upper=1, transom=5.31, fw=0.0, sill_=False)
e2 = win(F_left, 5.68, 3.51, 0.83, 2.02, 'round', cols=2, upper=1, transom=5.12, fw=0.0, sill_=False)
fr = unary_union([e1.buffer(0.15, **MITRE), e2.buffer(0.15, **MITRE), sbox(4.95, 3.51, 5.3, 5.3)])
add('trim', F_left.ext(fr.difference(unary_union([e1, e2]).buffer(0.06, **MITRE)), 0, 0.05))
add('trim', F_left.ext(unary_union([e1, e2]).buffer(0.06, **MITRE).difference(unary_union([e1, e2])), 0, 0.085))
add('stuc', F_left.box(5.06, 5.19, 3.7, 4.9, 0.05, 0.07))
sill(F_left, 4.57, 1.25, 3.8); sill(F_left, 5.68, 1.2, 3.43)
# Hauptportal
win(F_left, 7.9, 2.05, 1.48, 3.5, 'round', cols=2, upper=2, transom=4.44, fw=0.22, door=True, ears=(0.06, 0.25))
add('trim', F_left.box(6.95, 8.85, 5.55, 5.68, 0, 0.16))
pseg_o = affinity.translate(profile(1.95, 0.42, 'barrel'), 7.9, 5.68)
pseg_i = affinity.translate(profile(1.65, 0.3, 'barrel'), 7.9, 5.68)
add('trim', F_left.ext(pseg_o.difference(pseg_i), 0, 0.17))
add('trim', F_left.ext(pseg_o.buffer(0.04).difference(pseg_o).intersection(sbox(0, 5.75, 20, 9)), 0, 0.2))
add('stuc', F_left.ext(pseg_i, 0, 0.04))
for u in (7.25, 8.55):
    panel(F_left, u, 5.74, 0.42, 0.18, ring=0.02, raised=0.04)
relief('left', trace('left', 7.62, 8.2, 5.62, 6.38, thick=4), const(0.0), 0.0, 0.14, 'trim')
# Salonfenster (in Zeichnung und Farbfoto geschlossen -> Blendfenster)
win(F_left, 11.3, 3.15, 1.1, 2.38, 'seg', rise=0.32, fw=0.26, blind=True, key=True, feet=(0.04, 0.35), cut_out=0.2)
kd = sbox(5.48, 0.42, 6.48, 2.33)
CUT.append(F_left.ext(kd, -0.36, 0.22))
add('wood', F_left.ext(kd, -0.14, -0.1))
for i in range(5):                                   # schräge Brettfüllung
    a_ = 0.55 + i * 0.28
    add('wood2', F_left.ext(LineString([(5.55, a_), (6.41, a_ + 0.45)]).buffer(0.03).intersection(kd.buffer(-0.08)), -0.1, -0.08))
add('wood2', F_left.box(5.56, 6.4, 1.6, 2.25, -0.1, -0.07))
add('trim', F_left.ext(kd.buffer(0.1, **MITRE).difference(kd), 0, 0.08))
add('trim', F_left.box(5.33, 6.63, 2.33, 2.45, 0, 0.13))
for (u, w_) in ((3.28, 0.8), (4.7, 1.15)):
    win(F_left, u, 1.6, w_, 0.38, 'flat', cols=2, transom=None, fw=0.05, grille=True, cut_out=0.16, sill_=False)
win(F_left, 13.0, 0.6, 0.62, 0.75, 'flat', cols=1, transom=None, fw=0.06, grille=True, cut_out=0.16)

# Eingangspodest + Freitreppe links
lx0, ly0, ly1 = -1.45, 5.45, 7.8
SOCKEL_U.append(bx(lx0, 0.05, ly0, ly1, -0.5, 1.95))
SOCKEL_U.append(bx(lx0-0.05, 0.05, ly0-0.05, ly1+0.05, 1.95, 2.12))
F_land = Face((lx0, 14.5, 0), (0, -1, 0))
win(F_land, 14.5 - 6.62, 0.55, 1.55, 1.25, 'seg', rise=0.35, cols=3, upper=1, transom=None, fw=0.22, cut_out=0.2, sill_=False, hood='arch')
iron_rail([(lx0+0.03, ly0+0.02), (lx0+0.03, ly1-0.03), (-0.05, ly1-0.03)], 2.12, 0.9)
zg = zt(3.0); nr = 12; r = (2.12 - zg) / nr
for j in range(1, nr):
    ztop = 2.12 - j*r
    add('trim', bx(lx0+0.02, -0.02, ly0 - j*0.22, ly0 - (j-1)*0.22 + 0.03, zg - 0.4, ztop))
    add('trim', bx(lx0+0.02, -0.02, ly0 - j*0.22 - 0.03, ly0 - j*0.22 + 0.02, ztop - 0.04, ztop))
xr = lx0 + 0.05
for j in range(0, nr, 2):
    yb = ly0 - j*0.22 - 0.1; zb = 2.12 - j*r
    rod([xr, yb, zb], [xr, yb, zb + 0.9], r=0.018, sections=6)
yb_end = ly0 - (nr-1)*0.22
rail_pts = np.array([[xr-0.03, ly0+0.02, 3.02], [xr+0.03, ly0+0.02, 3.02], [xr-0.03, ly0+0.02, 3.07], [xr+0.03, ly0+0.02, 3.07],
                     [xr-0.03, yb_end, zg+0.9], [xr+0.03, yb_end, zg+0.9], [xr-0.03, yb_end, zg+0.95], [xr+0.03, yb_end, zg+0.95]])
add('iron', trimesh.convex.convex_hull(rail_pts))
for k in range(1, 12):
    t = k/12; yy = ly0 + 0.02 + t*(yb_end - ly0 - 0.02); zz = 3.02 + t*(zg + 0.9 - 3.02)
    zz0 = 2.12 - (ly0 - yy)/0.22*r
    rod([xr, yy, zz0], [xr, yy, zz], r=0.009, sections=4)
add('sockel', bx(lx0-0.05, lx0+0.12, yb_end, ly0, zg-0.3, zg+0.15))

# =========================================================== GIEBEL (Halbprofile aus den Ansichten gemessen)
# Vorderer Barockgiebel, Mitte x = 7.3
fh = smooth([(2.42, ZC), (2.42, 11.05), (2.66, 11.3), (2.72, 11.5), (2.58, 11.85), (2.4, 12.15), (2.18, 12.55), (1.95, 12.88), (1.85, 13.04)], 5) \
   + smooth([(1.55, 13.33), (1.38, 13.75), (1.12, 14.2), (0.75, 14.62), (0.32, 14.95), (0.0, 15.12)], 5)
gable_v3(F_front, 7.3, fh, back=4.6, fin=1.4, ridge=14.06, hwb=2.7)
for s in (-1, 1): volute(F_front, 7.3 + s*2.56, 11.25, 0.0, r=0.26)
add('trim', F_front.box(5.4, 9.2, 13.04, 13.2, -0.35, 0.16)); add('trim', F_front.box(5.35, 9.25, 13.2, 13.33, -0.35, 0.21))
for s in (-1, 1): ball(F_front.pt(7.3 + s*1.72, 13.52, 0.02), 0.15)
crown_mask = affinity.translate(from_points(fh), 7.3, 0).intersection(sbox(0, 13.38, 20, 99)).buffer(-0.1)
relief('front', trace('front', 5.85, 8.75, 13.36, 14.98, thick=3, min_area=0.002, mask=crown_mask), const(0.0), 0.0, 0.03, 'iron')
add('trim', F_front.box(7.2, 7.4, 13.33, 14.95, -0.05, 0.06))
relief('front', trace('front', 7.15, 7.45, 12.55, 12.98, thick=3), const(0.0), 0.0, 0.05, 'trim')
ga = win(F_front, 6.77, 10.88, 0.82, 1.57, 'flat', cols=2, upper=2, transom=11.72, fw=0.0, sill_=False, cut_out=0.1)
gb = win(F_front, 7.78, 10.88, 0.82, 1.57, 'flat', cols=2, upper=2, transom=11.72, fw=0.0, sill_=False, cut_out=0.1)
gframe(F_front, 6.11, 8.43, 10.82, 12.51, [ga, gb], ears=(0.06, 0.16))
hood_cornice(F_front, 7.27, 2.3, 12.51, consoles=False)

# Rechter Kielbogengiebel, Mitte y = 7.65
rh = [(2.15, ZC), (2.15, 11.22), (1.99, 11.24)] + smooth([(1.99, 11.24), (1.62, 11.93), (1.25, 12.48)], 5)[1:] \
   + smooth([(1.18, 12.62), (0.88, 13.1), (0.55, 13.5), (0.25, 13.82), (0.15, 13.9)], 5) + [(0.15, 14.36), (0.0, 14.36)]
gable_v3(F_right, 7.65, rh, back=3.8, ridge=13.76, hwb=2.45)
for s in (-1, 1):
    add('trim', F_right.box(7.65 + s*2.2 - 0.2, 7.65 + s*2.2 + 0.2, ZC, 11.3, -0.3, 0.1))
    ball(F_right.pt(7.65 + s*2.2, 11.5, -0.1), 0.18)
    volute(F_right, 7.65 + s*1.22, 12.55, 0.0, r=0.13)
add('trim', F_right.box(7.5, 7.8, 12.67, 14.36, -0.05, 0.07))
add('stuc', F_right.box(7.57, 7.73, 12.9, 14.1, 0.0, 0.08))
p = F_right.pt(7.65, 14.36, -0.05); finial(p[0], p[1], p[2], 0.9)
win(F_right, 7.64, 10.88, 0.92, 1.53, 'round', cols=2, upper=1, transom=11.83, fw=0.11, cut_out=0.1, sill_=True)

# Linker Mansardgiebel mit Haube, Mitte y = 6.78 (u = 7.72)
lh = smooth([(2.33, ZC), (2.01, 11.52), (1.75, 12.26), (1.53, 12.89), (1.32, 13.23)], 5) + [(1.0, 13.33), (0.5, 13.4), (0.0, 13.43)]
lroof = lh[:-3] + [(1.62, 13.23), (1.2, 13.5), (0.6, 13.95), (0.0, 14.26)]
gable_v3(F_left, 7.72, lh, back=3.8, ridge=14.26, hwb=2.6)
p = F_left.pt(7.72, 14.26, 0.0); finial(p[0], p[1], p[2], 0.6)
win(F_left, 7.8, 10.93, 0.85, 1.55, 'round', cols=2, upper=1, transom=11.85, fw=0.12, ears=(0.05, 0.15), cut_out=0.1)
relief('left', trace('left', 6.15, 6.85, 10.82, 11.7, thick=4), const(0.0), 0.0, 0.1, 'trim')
relief('left', trace('left', 8.95, 9.65, 10.82, 11.7, thick=4), const(0.0), 0.0, 0.1, 'trim')

# --- Gauben
dormer(F_front, 2.62, 11.19, 1.0, 0.7, 'flat', 'bell', cols=2, hoodH=1.1)
hood_mask = affinity.translate(profile(1.7, 1.1, 'bell'), 2.62, 11.97).buffer(-0.08)
relief('front', trace('front', 2.0, 3.25, 11.92, 12.95, thick=3, min_area=0.002, mask=hood_mask), lambda u: -0.6, 0.0, 0.02, 'iron')
dormer(F_right, 11.75, 11.2, 0.75, 0.85, 'flat', 'bell', cols=2, hoodH=0.8)
dormer(F_back, 3.53, 10.96, 1.6, 0.78, 'flat', 'barrel', cols=4, hoodH=0.35, fin=False)
dormer(F_back, 7.73, 10.96, 0.6, 0.78, 'flat', 'barrel', cols=2, hoodH=0.3, fin=False)
dormer(F_left, 3.5, 11.0, 0.7, 0.75, 'flat', 'bell', cols=2, hoodH=0.6)
dormer(F_left, 6.48, 14.84, 0.45, 0.32, 'flat', 'tri', cols=1, hoodH=0.45, fin=True)

# --- Dächer
roofU = trimesh.boolean.union([rf.mesh(ZC - 0.12) for rf in ROOFS], engine='manifold')
roofU = trimesh.boolean.difference([roofU, trimesh.boolean.union(ROOF_CUT, engine='manifold')], engine='manifold')
add('roof', roofU)
def in_cut(q):
    for m in ROOF_CUT:
        if np.all(np.einsum('ij,ij->i', m.face_normals, q - m.triangles[:, 0]) <= 1e-6): return True
    return False
for rf in ROOFS:
    def vis(q): return rf.h(q[0], q[1]) >= z_roof(q[0], q[1]) - 0.02 and not in_cut(q - [0, 0, 0.05])
    def clipped_rod(p0, p1, crock=False):
        p0 = np.array(p0, float); p1 = np.array(p1, float); L = np.linalg.norm(p1 - p0); k = max(1, int(L / 0.25))
        for i in range(k):
            a = p0 + (p1 - p0) * i / k; b = p0 + (p1 - p0) * (i + 1) / k
            if vis((a + b) / 2): rod(a, b, r=0.05, cat='metal', sections=6)
        if crock:
            for i in range(1, int(L / 0.6)):
                q = p0 + (p1 - p0) * i * 0.6 / L
                if vis(q): add('metal', trimesh.creation.cone(radius=0.06, height=0.18, sections=5).apply_translation(q + [0, 0, 0.03]))
    for (b, t) in zip(rf.base_corners(), rf.top_corners()):
        clipped_rod([b[0], b[1], rf.z0 + 0.02], [t[0], t[1], rf.z1 + 0.02], crock=True)
    tc = rf.top_corners()
    for a, b in zip(tc, tc[1:] + tc[:1]):
        clipped_rod([a[0], a[1], rf.z1 + 0.02], [b[0], b[1], rf.z1 + 0.02])
iron_rail([(4.9, 4.6), (5.6, 4.6), (5.6, 9.0), (4.9, 9.0)], 17.0, 0.4, rings=False, closed=True)
finial(5.25, 4.6, 17.0, 1.2); finial(5.25, 9.0, 17.0, 1.2)
finial(8.02, 6.35, 15.2, 1.0); finial(8.02, 12.0, 15.2, 1.0)
finial(10.95, 4.1, 9.62, 0.5)

# --- Dachrinnen + Fallrohre
eave = unary_union([sbox(*rf.b) for rf in ROOFS])
gut = eave.buffer(0.07, **MITRE).difference(eave.buffer(-0.05, **MITRE))
for m in ROOF_CUT:
    b = m.bounds; gut = gut.difference(sbox(b[0][0], b[0][1], b[1][0], b[1][1]))
add('metal', ext(gut, ZC - 0.22, ZC - 0.07))
for (x, y, ex_, ey_) in [(-0.13, 0.4, -0.5, 0.4), (-0.13, 13.5, -0.5, 13.5), (11.61, 14.1, 11.98, 14.1),
                         (11.61, 5.4, 11.98, 5.4), (0.4, 14.06, 0.4, 14.43), (9.6, -0.13, 9.6, -0.45)]:
    rod([x, y, zt(y) + 0.05], [x, y, ZC - 0.42], r=0.05, cat='metal', sections=8)
    rod([x, y, ZC - 0.42], [ex_, ey_, ZC - 0.17], r=0.05, cat='metal', sections=8)
    for zz in np.arange(2.6, 10.0, 2.0):
        add('metal', trimesh.creation.cylinder(radius=0.065, segment=[[x, y, zz], [x, y, zz + 0.05]], sections=8))

# --- Schornsteine (Lage nach DG-Plan und Ansichten)
for (cx, cy, wx, wy, top) in [(7.45, 5.65, 0.85, 0.7, 17.2), (2.65, 9.75, 1.2, 0.7, 16.4)]:
    add('chimney', bx(cx-wx/2, cx+wx/2, cy-wy/2, cy+wy/2, 9.0, top - 0.75))
    add('trim', bx(cx-wx/2-0.05, cx+wx/2+0.05, cy-wy/2-0.05, cy+wy/2+0.05, top - 0.85, top - 0.75))
    for sx_ in np.linspace(-wx/2 + 0.06, wx/2 - 0.06, max(3, int(wx / 0.2) + 1)):      # Schlitzzone
        add('chimney', bx(cx + sx_ - 0.04, cx + sx_ + 0.04, cy-wy/2, cy+wy/2, top - 0.75, top - 0.37))
    for sy_ in np.linspace(-wy/2 + 0.06, wy/2 - 0.06, 3):
        add('chimney', bx(cx-wx/2, cx+wx/2, cy + sy_ - 0.04, cy + sy_ + 0.04, top - 0.75, top - 0.37))
    add('trim', bx(cx-wx/2-0.12, cx+wx/2+0.12, cy-wy/2-0.12, cy+wy/2+0.12, top - 0.37, top - 0.22))
    add('trim', bx(cx-wx/2-0.08, cx+wx/2+0.08, cy-wy/2-0.08, cy+wy/2+0.08, top - 0.22, top))

# --- Gelände + Weg
xs = np.arange(-10, 22.01, 0.5); ys = np.arange(-10, 26.01, 0.5)
X, Y = np.meshgrid(xs, ys)
Zg = np.vectorize(zt)(Y)
V = np.c_[X.ravel(), Y.ravel(), Zg.ravel()]
nx = len(xs); faces = []
for j in range(len(ys)-1):
    for i in range(nx-1):
        a = j*nx + i; b = a + 1; c = a + nx; d = c + 1
        faces += [[a, b, d], [a, d, c]]
add('grass', trimesh.Trimesh(V, np.array(faces), process=False))
def drape(poly, cat, dz=0.03):
    ys_ = np.arange(poly.bounds[1], poly.bounds[3] + 1e-6, 0.5)
    xs_ = np.arange(poly.bounds[0], poly.bounds[2] + 1e-6, 0.5)
    for y in ys_[:-1]:
        for x in xs_[:-1]:
            c = sbox(x, y, x+0.5, y+0.5).intersection(poly)
            if c.is_empty or c.area < 1e-4: continue
            for cc in ([c] if c.geom_type == 'Polygon' else [q for q in c.geoms if q.geom_type == 'Polygon']):
                v2, f2 = trimesh.creation.triangulate_polygon(cc)
                v = np.c_[v2, [zt(yy) + dz for yy in v2[:, 1]]]
                add(cat, trimesh.Trimesh(v, f2, process=False))
drape(sbox(-2.6, -10, -0.4, 3.1), 'path', 0.05)
drape(sbox(-0.4, -10, 1.2, -1.2), 'path', 0.05)
drape(S.buffer(0.7, **MITRE).difference(S.buffer(0.05, **MITRE)).difference(sbox(-1.6, 2.5, 0.2, 8.0)), 'path', 0.025)


# ================================================================= Booleans
def U(lst):
    ms = [m for m in lst if m is not None]
    return trimesh.boolean.union(ms, engine='manifold')
cutter = U(CUT)
for name, lst, cat in [('wall', WALL, 'wall'), ('trimU', TRIM_U, 'trim'), ('sockelU', SOCKEL_U, 'sockel')]:
    u = U(lst)
    d = trimesh.boolean.difference([u, cutter], engine='manifold')
    print(name, len(u.faces), '->', len(d.faces))
    add(cat, d)

MAT = {  # sRGB, metallic, roughness  – Farbgebung nach historischem S/W-Foto (Helligkeitswerte kalibriert, Farbtöne angenommen)
    'wall':   ((205, 188, 152), 0.0, 0.92),   # Putz sandocker (Farbfoto, Weißabgleich am Himmel, Belichtung korrigiert)
    'trim':   ((178, 162, 128), 0.0, 0.88),   # Gesimse/Faschen/Sohlbänke: gleicher Ockerton, ca. 20 % dunkler als der Putz
    'stuc':   ((207, 190, 155), 0.0, 0.9),
    'sockel': ((120, 116, 96), 0.0, 0.95),    # Sockelsteine graubraun (Farbfoto)
    'roof':   ((49, 62, 78), 0.0, 0.55),      # Schiefer dunkel blaugrau (Farbfoto)
    'metal':  ((150, 156, 156), 0.6, 0.45),   # Zink
    'glass':  ((40, 48, 56), 0.3, 0.08),
    'sash':   ((238, 236, 228), 0.0, 0.6),    # weiß gestrichene Holzfenster (Gaube im Foto klar hell)
    'iron':   ((34, 34, 36), 0.5, 0.5),
    'wood':   ((78, 52, 36), 0.0, 0.7), 'wood2': ((96, 64, 44), 0.0, 0.7),
    'chimney':((150, 118, 100), 0.0, 0.95),   # Schornsteinköpfe dunkler als Fassade (Ziegel/verputzt)
    'grass':  ((98, 128, 72), 0.0, 1.0), 'path': ((192, 184, 166), 0.0, 1.0),
}
NAMES = {'wall': 'Fassade_Putz', 'trim': 'Gesimse_Faschen_Sandstein', 'sockel': 'Sockel_Naturstein',
         'stuc': 'Putzspiegel_Ornamentfelder', 'roof': 'Dach_Schiefer', 'metal': 'Zink_Grate_Spitzen',
         'glass': 'Fenster_Glas', 'sash': 'Fenster_Rahmen_Sprossen', 'iron': 'Gelaender_Schmiedeeisen',
         'wood': 'Tueren_Holz', 'wood2': 'Tueren_Fuellungen', 'chimney': 'Schornsteine',
         'grass': 'Gelaende', 'path': 'Wege_Kies'}

def build_scene():
    sc = trimesh.Scene()
    T = np.eye(4)
    T[:3, :3] = [[1, 0, 0], [0, 0, 1], [0, -1, 0]]          # Z-up -> Y-up (glTF)
    C = np.eye(4); C[:3, 3] = [-5.74, -7.25, 0]
    for cat, ms in G.items():
        m = trimesh.util.concatenate(ms)
        m.merge_vertices()
        m.apply_transform(T @ C)
        col, met, rough = MAT[cat]
        mat = trimesh.visual.material.PBRMaterial(name=NAMES[cat], baseColorFactor=[c/255 for c in col] + [1.0],
                                                  metallicFactor=met, roughnessFactor=rough, doubleSided=True)
        m.visual = trimesh.visual.TextureVisuals(material=mat)
        sc.add_geometry(m, node_name=NAMES[cat], geom_name=NAMES[cat])
    return sc

if __name__ == '__main__':
    import pickle
    sc = build_scene()
    glb = trimesh.exchange.gltf.export_glb(sc, include_normals=False)
    open('/home/claude/villa_callenberg.glb', 'wb').write(glb)
    print('GLB MB', len(glb)/1e6, 'faces', sum(len(g.faces) for g in sc.geometry.values()))
    pickle.dump({k: trimesh.util.concatenate(v) for k, v in G.items()}, open('/home/claude/geo.pkl', 'wb'))
