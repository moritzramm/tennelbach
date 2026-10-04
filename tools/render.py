import pickle, numpy as np, sys
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
sys.path.insert(0, '/home/claude')
from build import MAT
G = pickle.load(open('/home/claude/geo.pkl', 'rb'))

def render(fn, eye, target, up=(0, 0, 1), ortho=None, size=(10, 10), skip=('grass',), xlim=None, ylim=None):
    eye = np.array(eye, float); target = np.array(target, float)
    f = target - eye; f /= np.linalg.norm(f)
    r = np.cross(f, up); r /= np.linalg.norm(r); u = np.cross(r, f)
    L = -f * 0.6 + np.array([-0.4, -0.5, 0.7]); L /= np.linalg.norm(L)
    polys, cols, depth = [], [], []
    for cat, m in G.items():
        if cat in skip: continue
        V = m.vertices[m.faces]
        n = m.face_normals
        c = (V - eye) @ np.c_[r, u, f]
        if ortho:
            vis = (n @ f) < 0 if cat not in ('glass',) else np.ones(len(n), bool)
            P2 = c[:, :, :2]
        else:
            vis = ((n @ f) < 0) & (c[:, :, 2].min(1) > 0.5)
            P2 = c[:, :, :2] / c[:, :, 2:3]
        if cat in ('iron', 'metal', 'sash', 'glass', 'path'): vis = np.ones(len(n), bool) & (c[:, :, 2].min(1) > 0.5)
        base = np.array(MAT[cat][0]) / 255
        sh = 0.45 + 0.55 * np.clip(np.abs(n @ L) if cat in ('iron', 'metal') else n @ L, 0, 1)
        col = np.clip(base[None, :] * sh[:, None], 0, 1)
        polys.append(P2[vis]); cols.append(col[vis]); depth.append(c[vis, :, 2].mean(1))
    P = np.concatenate(polys); C = np.concatenate(cols); D = np.concatenate(depth)
    o = np.argsort(-D)
    fig, ax = plt.subplots(figsize=size, dpi=110)
    ax.add_collection(PolyCollection(P[o], facecolors=C[o], edgecolors=C[o], linewidths=0.15))
    if xlim: ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    else: ax.autoscale()
    ax.set_aspect('equal'); ax.axis('off'); ax.set_facecolor((0.85, 0.9, 0.95)); fig.patch.set_facecolor((0.86, 0.9, 0.95))
    plt.tight_layout(); plt.savefig(fn); plt.close()

if __name__ == '__main__':
    which = sys.argv[1:] or ['front', 'right', 'left', 'back', 'persp']
    c = (5.74, 7.25, 8)
    if 'front' in which: render('v_front.png', (5.74, -60, 9), (5.74, 7, 9), ortho=True, xlim=(-7.5, 8), ylim=(-9.5, 10.5))
    if 'right' in which: render('v_right.png', (60, 7.25, 9), (5, 7.25, 9), ortho=True, xlim=(-9, 9), ylim=(-9.5, 10.5))
    if 'left' in which:  render('v_left.png', (-60, 7.25, 9), (5, 7.25, 9), ortho=True, xlim=(-9, 9), ylim=(-9.5, 10.5))
    if 'back' in which:  render('v_back.png', (5.74, 70, 9), (5.74, 7, 9), ortho=True, xlim=(-8.5, 8), ylim=(-9.5, 10.5))
    if 'persp' in which:
        render('v_persp.png', (-11, -14, 11), (5.5, 6, 8.5), skip=('grass','path'))
        render('v_persp2.png', (23, 25, 12), (6, 7, 8.5), skip=('grass','path'))
