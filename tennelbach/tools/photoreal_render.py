"""Fotorealistisches Rendering (Blender Cycles) der Villa aus der Blickrichtung des Screenshots."""
import bpy, math, sys
from mathutils import Vector

RES = int(sys.argv[-3]) if len(sys.argv) > 3 else 1000
SPP = int(sys.argv[-2]) if len(sys.argv) > 3 else 64
OUT = sys.argv[-1] if len(sys.argv) > 3 else '/home/claude/photo.png'
OFF = Vector((5.74, 7.25, 0.0))          # Plan -> Blender (glTF zentriert)

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath='/home/claude/villa_callenberg.glb')
sc = bpy.context.scene

# --- Gelände großflächig fortsetzen (gleiche Hangfunktion wie im Modell)
def zt(y):
    if y < 0: return max(-0.9, 0.1 + 0.05 * y)
    if y <= 14.5: return 0.1 + 1.7 * (y / 14.5) ** 2
    return min(3.6, 1.8 + 0.12 * (y - 14.5))
for o in list(bpy.data.objects):
    if o.name.startswith('Gelaende') and not o.name.startswith('Gelaender'):
        bpy.data.objects.remove(o, do_unlink=True)
import bmesh
me = bpy.data.meshes.new('Gelaende_gross'); bm = bmesh.new()
xs = [-3000.0, -1200.0, -600.0, -300.0, -200.0] + [x * 1.0 for x in range(-160, 161, 2)] + [200.0, 300.0, 600.0, 1200.0, 3000.0]; ys = xs
grid = {}
for i, x in enumerate(xs):
    for j, y in enumerate(ys):
        # feiner in Hausnähe
        grid[i, j] = bm.verts.new((x - OFF.x, y - OFF.y, zt(y)))
for i in range(len(xs) - 1):
    for j in range(len(ys) - 1):
        bm.faces.new((grid[i, j], grid[i + 1, j], grid[i + 1, j + 1], grid[i, j + 1]))
bm.to_mesh(me); bm.free()
ground = bpy.data.objects.new('Gelaende_gross', me); sc.collection.objects.link(ground)
sub = ground.modifiers.new('sub', 'SUBSURF'); sub.levels = 0; sub.render_levels = 1; sub.subdivision_type = 'SIMPLE'

# --- Glättung nach Winkel (Säulen, Kugeln, Bögen), Kanten bleiben scharf
for o in bpy.data.objects:
    if o.type == 'MESH':
        for p in o.data.polygons: p.use_smooth = True
        try:
            o.data.set_sharp_from_angle(angle=math.radians(35))
        except Exception:
            pass

# --- Materialien (prozedural, ohne UVs: Objektkoordinaten)
def node_mat(name):
    m = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree; nt.nodes.clear()
    out = nt.nodes.new('ShaderNodeOutputMaterial'); bsdf = nt.nodes.new('ShaderNodeBsdfPrincipled')
    nt.links.new(bsdf.outputs['BSDF'], out.inputs['Surface'])
    tc = nt.nodes.new('ShaderNodeTexCoord')
    return m, nt, bsdf, tc

def srgb(h):
    h = h.lstrip('#'); c = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    return [((v + 0.055) / 1.055) ** 2.4 if v > 0.04045 else v / 12.92 for v in c] + [1.0]

def textured(name, base, rough, var=0.08, fine_scale=60.0, bump=0.15, coarse=0.6, dirt=0.0, metallic=0.0, bands=None):
    m, nt, bsdf, tc = node_mat(name)
    L = nt.links
    n1 = nt.nodes.new('ShaderNodeTexNoise'); n1.inputs['Scale'].default_value = coarse; n1.inputs['Detail'].default_value = 4
    L.new(tc.outputs['Object'], n1.inputs['Vector'])
    ramp = nt.nodes.new('ShaderNodeValToRGB')
    b = srgb(base)
    dark = [max(0, c * (1 - var)) for c in b[:3]] + [1]; light = [min(1, c * (1 + var * 0.6)) for c in b[:3]] + [1]
    ramp.color_ramp.elements[0].color = dark; ramp.color_ramp.elements[1].color = light
    L.new(n1.outputs['Fac'], ramp.inputs['Fac'])
    col = ramp.outputs['Color']
    if dirt > 0:   # leichte Verschmutzung nach unten / Laufspuren
        sep = nt.nodes.new('ShaderNodeSeparateXYZ'); L.new(tc.outputs['Object'], sep.inputs['Vector'])
        n3 = nt.nodes.new('ShaderNodeTexNoise'); n3.inputs['Scale'].default_value = 3.0; n3.inputs['Detail'].default_value = 8
        mapn = nt.nodes.new('ShaderNodeMapping'); mapn.inputs['Scale'].default_value = (6, 6, 0.6)
        L.new(tc.outputs['Object'], mapn.inputs['Vector']); L.new(mapn.outputs['Vector'], n3.inputs['Vector'])
        mr = nt.nodes.new('ShaderNodeMapRange'); mr.inputs['From Min'].default_value = 0.45; mr.inputs['From Max'].default_value = 0.75
        mr.inputs['To Min'].default_value = 0.0; mr.inputs['To Max'].default_value = dirt
        L.new(n3.outputs['Fac'], mr.inputs['Value'])
        mix = nt.nodes.new('ShaderNodeMix'); mix.data_type = 'RGBA'; mix.blend_type = 'MULTIPLY'
        mix.inputs['B'].default_value = (0.55, 0.52, 0.47, 1)
        L.new(mr.outputs['Result'], mix.inputs['Factor']); L.new(col, mix.inputs['A']); col = mix.outputs['Result']
    if bands:      # Schieferreihen
        w = nt.nodes.new('ShaderNodeTexWave'); w.bands_direction = 'Z'; w.inputs['Scale'].default_value = bands
        w.inputs['Distortion'].default_value = 0.4; w.inputs['Detail'].default_value = 1
        L.new(tc.outputs['Object'], w.inputs['Vector'])
        mr2 = nt.nodes.new('ShaderNodeMapRange'); mr2.inputs['To Min'].default_value = 0.82; mr2.inputs['To Max'].default_value = 1.0
        L.new(w.outputs['Fac'], mr2.inputs['Value'])
        mix2 = nt.nodes.new('ShaderNodeMix'); mix2.data_type = 'RGBA'; mix2.blend_type = 'MULTIPLY'; mix2.inputs['Factor'].default_value = 1.0
        L.new(col, mix2.inputs['A'])
        cr = nt.nodes.new('ShaderNodeCombineColor'); L.new(mr2.outputs['Result'], cr.inputs[0]); L.new(mr2.outputs['Result'], cr.inputs[1]); L.new(mr2.outputs['Result'], cr.inputs[2])
        L.new(cr.outputs['Color'], mix2.inputs['B']); col = mix2.outputs['Result']
        bw = nt.nodes.new('ShaderNodeBump'); bw.inputs['Strength'].default_value = 0.35
        L.new(w.outputs['Fac'], bw.inputs['Height']); L.new(bw.outputs['Normal'], bsdf.inputs['Normal'])
    L.new(col, bsdf.inputs['Base Color'])
    bsdf.inputs['Roughness'].default_value = rough
    bsdf.inputs['Metallic'].default_value = metallic
    if bump > 0 and not bands:
        n2 = nt.nodes.new('ShaderNodeTexNoise'); n2.inputs['Scale'].default_value = fine_scale; n2.inputs['Detail'].default_value = 6
        L.new(tc.outputs['Object'], n2.inputs['Vector'])
        bp = nt.nodes.new('ShaderNodeBump'); bp.inputs['Strength'].default_value = bump; bp.inputs['Distance'].default_value = 0.01
        L.new(n2.outputs['Fac'], bp.inputs['Height']); L.new(bp.outputs['Normal'], bsdf.inputs['Normal'])
    return m

def lawn(name):
    m, nt, bsdf, tc = node_mat(name); L = nt.links
    cols = []
    prev = None
    for sc_, lo, hi in [(0.04, '#3E5124', '#566A30'), (0.35, '#43562A', '#5A6E34'), (4.0, '#3B4E22', '#61763A')]:
        n_ = nt.nodes.new('ShaderNodeTexNoise'); n_.inputs['Scale'].default_value = sc_; n_.inputs['Detail'].default_value = 6
        L.new(tc.outputs['Object'], n_.inputs['Vector'])
        r_ = nt.nodes.new('ShaderNodeValToRGB'); r_.color_ramp.elements[0].color = srgb(lo); r_.color_ramp.elements[1].color = srgb(hi)
        r_.color_ramp.elements[0].position = 0.35; r_.color_ramp.elements[1].position = 0.65
        L.new(n_.outputs['Fac'], r_.inputs['Fac'])
        if prev is None: prev = r_.outputs['Color']
        else:
            mx = nt.nodes.new('ShaderNodeMix'); mx.data_type = 'RGBA'; mx.inputs['Factor'].default_value = 0.5
            L.new(prev, mx.inputs['A']); L.new(r_.outputs['Color'], mx.inputs['B']); prev = mx.outputs['Result']
    L.new(prev, bsdf.inputs['Base Color'])
    bsdf.inputs['Roughness'].default_value = 0.95
    n2 = nt.nodes.new('ShaderNodeTexNoise'); n2.inputs['Scale'].default_value = 900; L.new(tc.outputs['Object'], n2.inputs['Vector'])
    bp = nt.nodes.new('ShaderNodeBump'); bp.inputs['Strength'].default_value = 0.25; bp.inputs['Distance'].default_value = 0.002
    L.new(n2.outputs['Fac'], bp.inputs['Height']); L.new(bp.outputs['Normal'], bsdf.inputs['Normal'])
    return m

def foliage(name, hexcol):
    m, nt, bsdf, tc = node_mat(name); L = nt.links
    n_ = nt.nodes.new('ShaderNodeTexNoise'); n_.inputs['Scale'].default_value = 6; n_.inputs['Detail'].default_value = 8
    L.new(tc.outputs['Object'], n_.inputs['Vector'])
    r_ = nt.nodes.new('ShaderNodeValToRGB'); c = srgb(hexcol)
    r_.color_ramp.elements[0].color = [v * 0.45 for v in c[:3]] + [1]; r_.color_ramp.elements[1].color = [min(1, v * 1.25) for v in c[:3]] + [1]
    L.new(n_.outputs['Fac'], r_.inputs['Fac']); L.new(r_.outputs['Color'], bsdf.inputs['Base Color'])
    bsdf.inputs['Roughness'].default_value = 0.75
    n2 = nt.nodes.new('ShaderNodeTexVoronoi'); n2.inputs['Scale'].default_value = 25; L.new(tc.outputs['Object'], n2.inputs['Vector'])
    bp = nt.nodes.new('ShaderNodeBump'); bp.inputs['Strength'].default_value = 0.9
    L.new(n2.outputs['Distance'], bp.inputs['Height']); L.new(bp.outputs['Normal'], bsdf.inputs['Normal'])
    return m

def glass(name):
    m, nt, bsdf, tc = node_mat(name)
    bsdf.inputs['Base Color'].default_value = (0.012, 0.016, 0.02, 1)
    bsdf.inputs['Roughness'].default_value = 0.02
    bsdf.inputs['IOR'].default_value = 1.52
    return m

MATS = {
    'Fassade_Putz':               lambda n: textured(n, '#CDBC98', 0.9, var=0.07, fine_scale=90, bump=0.22, dirt=0.18),
    'Putzspiegel_Ornamentfelder': lambda n: textured(n, '#CFBE9B', 0.85, var=0.05, fine_scale=120, bump=0.08, dirt=0.12),
    'Gesimse_Faschen_Sandstein':  lambda n: textured(n, '#B2A280', 0.8, var=0.1, fine_scale=45, bump=0.3, dirt=0.25),
    'Sockel_Naturstein':          lambda n: textured(n, '#787460', 0.9, var=0.18, fine_scale=25, bump=0.45, coarse=1.6, dirt=0.2),
    'Dach_Schiefer':              lambda n: textured(n, '#3E4247', 0.7, var=0.2, coarse=2.5, bands=5.5),
    'Zink_Grate_Spitzen':         lambda n: textured(n, '#8A9294', 0.38, var=0.08, bump=0.0, metallic=0.85),
    'Fenster_Glas':               glass,
    'Fenster_Rahmen_Sprossen':    lambda n: textured(n, '#EDEBE3', 0.45, var=0.03, bump=0.0),
    'Gelaender_Schmiedeeisen':    lambda n: textured(n, '#222224', 0.45, var=0.05, bump=0.0, metallic=0.7),
    'Tueren_Holz':                lambda n: textured(n, '#4E3424', 0.55, var=0.12, fine_scale=12, bump=0.1),
    'Tueren_Fuellungen':          lambda n: textured(n, '#5E402A', 0.55, var=0.12, fine_scale=12, bump=0.1),
    'Schornsteine':               lambda n: textured(n, '#8E6E5E', 0.9, var=0.15, fine_scale=30, bump=0.35, dirt=0.35),
    'Wege_Kies':                  lambda n: textured(n, '#B8AE98', 0.95, var=0.2, fine_scale=180, bump=0.5, coarse=3.0),
    'Gelaende':                   lambda n: lawn(n),
}
for o in bpy.data.objects:
    if o.type != 'MESH': continue
    key = next((k for k in MATS if o.name.startswith(k)), None)
    if o.name == 'Gelaende_gross': key = 'Gelaende'
    if key is None: continue
    m = MATS[key](key + '_PBR')
    o.data.materials.clear(); o.data.materials.append(m)

# --- Umgebung: einfache Laub- und Nadelbäume (frei angenommen, nicht aus den Plänen)
tex = bpy.data.textures.new('Wolken', 'CLOUDS'); tex.noise_scale = 0.7; tex.noise_depth = 5
bark = textured('Rinde', '#4A3B30', 0.9, var=0.2, fine_scale=40, bump=0.4)
leafA = foliage('Laub', '#3E5A22'); leafB = foliage('Nadel', '#2C4220')
import random; random.seed(7)
def place(o):
    sc.collection.objects.link(o)
def tree(x, y, h, r, conifer=False):
    x0, y0 = x - OFF.x, y - OFF.y; z0 = zt(y) - 0.1
    me_t = bpy.data.meshes.new('stamm'); bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, segments=10, radius1=r * 0.09, radius2=r * 0.05, depth=h * 0.6)
    bmesh.ops.translate(bm, verts=bm.verts, vec=(x0, y0, z0 + h * 0.3)); bm.to_mesh(me_t); bm.free()
    st = bpy.data.objects.new('Stamm', me_t); st.data.materials.append(bark); place(st)
    me_k = bpy.data.meshes.new('krone'); bm = bmesh.new()
    if conifer:
        bmesh.ops.create_cone(bm, cap_ends=True, segments=24, radius1=r, radius2=0.05, depth=h * 0.85)
        for v in bm.verts: v.co.z += 0.0
        bmesh.ops.translate(bm, verts=bm.verts, vec=(x0, y0, z0 + h * 0.55))
    else:
        for k_ in range(3):   # unregelmäßige Krone aus mehreren Ballen
            rr = r * (1.0 if k_ == 0 else random.uniform(0.55, 0.75))
            dx, dy, dz = (0, 0, 0) if k_ == 0 else (random.uniform(-0.7, 0.7) * r, random.uniform(-0.7, 0.7) * r, random.uniform(-0.5, 0.35) * r)
            g_ = bmesh.ops.create_icosphere(bm, subdivisions=3 if r < 6 else 4, radius=rr)
            vs = g_['verts']
            bmesh.ops.scale(bm, vec=(1.0, 1.0, random.uniform(1.0, 1.25)), verts=vs)
            bmesh.ops.translate(bm, verts=vs, vec=(x0 + dx, y0 + dy, z0 + h - r * 1.05 + dz))
    bm.to_mesh(me_k); bm.free()
    kr = bpy.data.objects.new('Krone', me_k); kr.data.materials.append(leafB if conifer else leafA); place(kr)
    if conifer:
        ss = kr.modifiers.new('ss', 'SUBSURF'); ss.levels = 2; ss.render_levels = 3; ss.subdivision_type = 'SIMPLE'
    d = kr.modifiers.new('d', 'DISPLACE'); d.texture = tex; d.strength = r * (0.35 if conifer else 0.55); d.texture_coords = 'GLOBAL'
    for p in me_k.polygons: p.use_smooth = True
for (x, y, h, r, c) in [(24, -6, 16, 4.5, False), (31, 6, 13, 4.0, False), (15, -15, 12, 3.8, False), (-8, -14, 17, 2.6, True),
                        (-2, -20, 14, 4.2, False), (36, -12, 18, 5.0, False), (28, 30, 11, 3.5, False), (34, 22, 15, 2.2, True),
                        (-22, 2, 12, 3.8, False), (-26, -10, 16, 2.4, True), (44, 12, 14, 4.5, False), (8, -30, 15, 5.0, False),
                        (-30, 18, 9, 3.0, False), (20, 44, 12, 4.0, False), (-12, 50, 10, 3.5, False)]:
    tree(x, y, h, r, c)
# entfernte Baumreihe am Horizont (in Blickrichtung hinter dem Haus)
import math as _m
fx_, fy_ = 0.77, -0.64; lx_, ly_ = 0.64, 0.77
for k in range(40):
    d_ = random.uniform(120, 230); lat = random.uniform(-160, 160)
    x = 5.7 + fx_ * d_ + lx_ * lat; y = 7.2 + fy_ * d_ + ly_ * lat
    tree(x, y, random.uniform(13, 19), random.uniform(3.5, 5.5), False)

# --- Licht: physikalischer Himmel + Sonne (Sonne von links hinten, wie im Screenshot)
world = bpy.data.worlds.new('Welt'); sc.world = world; world.use_nodes = True
wn = world.node_tree; wn.nodes.clear()
sky = wn.nodes.new('ShaderNodeTexSky')
try: sky.sky_type = 'NISHITA'
except Exception: pass
sun_dir = Vector((-0.78, 0.38, 0.5)).normalized()
elev = math.asin(sun_dir.z); azi = math.atan2(sun_dir.x, sun_dir.y)
try:
    sky.sun_elevation = elev; sky.sun_rotation = -azi; sky.sun_disc = False; sky.air_density = 1.2; sky.dust_density = 1.5
except Exception: pass
bg = wn.nodes.new('ShaderNodeBackground'); bg.inputs['Strength'].default_value = 0.22
wo = wn.nodes.new('ShaderNodeOutputWorld')
wn.links.new(sky.outputs['Color'], bg.inputs['Color']); wn.links.new(bg.outputs['Background'], wo.inputs['Surface'])
sun = bpy.data.lights.new('Sonne', 'SUN'); sun.energy = 4.2; sun.angle = math.radians(0.8)
sun.color = (1.0, 0.96, 0.9)
so = bpy.data.objects.new('Sonne', sun); sc.collection.objects.link(so)
so.rotation_euler = (-sun_dir).to_track_quat('-Z', 'Y').to_euler()

# --- Kamera (aus dem Screenshot rückgerechnet, Plan-Koordinaten)
cam_plan = Vector((-37.8, 43.6, 25.9)); yaw, pitch = math.radians(-39.7), math.radians(-15.3)
fwd = Vector((math.cos(pitch) * math.cos(yaw), math.cos(pitch) * math.sin(yaw), math.sin(pitch)))
cd = bpy.data.cameras.new('Kamera'); cd.lens = 72.3; cd.sensor_width = 36; cd.sensor_fit = 'AUTO'
co = bpy.data.objects.new('Kamera', cd); sc.collection.objects.link(co)
co.location = cam_plan - OFF
co.rotation_euler = fwd.to_track_quat('-Z', 'Y').to_euler()
sc.camera = co

# --- Render
sc.render.engine = 'CYCLES'
cy = sc.cycles
cy.device = 'CPU'; cy.samples = SPP; cy.use_adaptive_sampling = True; cy.adaptive_threshold = 0.03
cy.max_bounces = 6; cy.diffuse_bounces = 3; cy.glossy_bounces = 3; cy.transparent_max_bounces = 4
try:
    cy.use_denoising = True; cy.denoiser = 'OPENIMAGEDENOISE'
except Exception: pass
sc.render.resolution_x = RES; sc.render.resolution_y = RES; sc.render.resolution_percentage = 100
try:
    sc.view_settings.view_transform = 'AgX'; sc.view_settings.look = 'AgX - Medium High Contrast'
except Exception:
    sc.view_settings.view_transform = 'Filmic'
sc.view_settings.exposure = -0.25
sc.render.image_settings.file_format = 'PNG'
sc.render.filepath = OUT
bpy.ops.render.render(write_still=True)
print('DONE', OUT)
