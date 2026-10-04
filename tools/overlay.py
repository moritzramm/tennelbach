"""Orthografische Rasterung des Modells, pixelgenau auf die Ansichtszeichnungen ausgerichtet."""
import pickle, sys, numpy as np
from PIL import Image, ImageDraw
sys.path.insert(0, '/home/claude')

COLS = {'wall': (232, 214, 170), 'trim': (200, 160, 110), 'sockel': (150, 140, 120), 'stuc': (225, 200, 160),
        'roof': (120, 140, 170), 'metal': (150, 170, 170), 'glass': (90, 130, 170), 'sash': (250, 250, 250),
        'iron': (60, 60, 60), 'wood': (150, 100, 60), 'wood2': (170, 120, 80), 'chimney': (200, 130, 110),
        'grass': (160, 200, 140), 'path': (220, 210, 190)}

# Ansicht: (Bilddatei, u-Funktion, Tiefe (größer = weiter weg), X0, sx, Y0, sz, Crop)
VIEWS = {
    'front': ('D0_sw.jpg', lambda v: v[..., 0], lambda v: v[..., 1], 242.0, 92.9, 2178.7, 94.6, (150, 400, 1580, 2300)),
    'right': ('D0_sw.jpg', lambda v: v[..., 1], lambda v: -v[..., 0], 1989.0, 92.9, 2178.7, 94.6, (1830, 400, 3450, 2300)),
    'left':  ('E0_sw.jpg', lambda v: 14.5 - v[..., 1], lambda v: v[..., 0], 328.0, 94.5, 2238.7, 94.6, (230, 450, 1950, 2380)),
    'back':  ('E0_sw.jpg', lambda v: 11.48 - v[..., 0], lambda v: -v[..., 1], 2351.0, 93.7, 2238.7, 94.6, (2150, 450, 3650, 2380)),
}

def raster(view, G, skip=('grass', 'path')):
    f, U, D, X0, sx, Y0, sz, crop = VIEWS[view]
    W, H = crop[2] - crop[0], crop[3] - crop[1]
    img = Image.new('RGB', (W, H), (255, 255, 255)); dr = ImageDraw.Draw(img)
    polys = []
    for cat, m in G.items():
        if cat in skip: continue
        V = m.vertices[m.faces]
        u = U(V); z = V[..., 2]; d = D(V).mean(1)
        px = X0 + sx * u - crop[0]; py = Y0 - sz * z - crop[1]
        n = m.face_normals
        base = np.array(COLS[cat])
        sh = 0.55 + 0.45 * np.clip(np.abs(n @ np.array([-0.4, -0.5, 0.75])), 0, 1)
        for k in range(len(V)):
            polys.append((d[k], [(px[k, 0], py[k, 0]), (px[k, 1], py[k, 1]), (px[k, 2], py[k, 2])], tuple((base * sh[k]).astype(int))))
    polys.sort(key=lambda p: -p[0])
    for _, pts, c in polys:
        dr.polygon(pts, fill=c)
    return img

def overlay(view, G, out, alpha=0.55):
    f, U, D, X0, sx, Y0, sz, crop = VIEWS[view]
    dwg = Image.open('/mnt/user-data/uploads/' + f).convert('L').crop(crop)
    mdl = raster(view, G)
    # Linien der Zeichnung dunkelrot über das Modell legen
    a = np.asarray(dwg).astype(float) / 255.0
    m = np.asarray(mdl).astype(float)
    ink = np.clip((0.75 - a) / 0.5, 0, 1)[..., None]
    col = np.array([200, 0, 0], float)
    res = m * (1 - alpha * 0.4) + 255 * alpha * 0.4
    res = res * (1 - ink) + col * ink
    Image.fromarray(res.astype(np.uint8)).save(out)
    return out

if __name__ == '__main__':
    G = pickle.load(open('/home/claude/geo.pkl', 'rb'))
    for v in (sys.argv[1:] or VIEWS.keys()):
        print(overlay(v, G, f'/home/claude/ov_{v}.png'))
