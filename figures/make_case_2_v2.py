"""Rebuild Case Study 2 figure: same content as case_1.pdf, molecules redrawn from SMILES at one fixed bond length.
Monomer SMILES regenerated with retrosynthesis/retro.py (depolymerize, Imide_split template);
SAScores and CIDs identical to the original figure."""
import io, sys, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
from PIL import Image
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
rdDepictor.SetPreferCoordGen(True)
DPI = 600
RADAR = sys.argv[1]; OUT = sys.argv[2]
C = [
 dict(name='Candidate 1', radar=f'{RADAR}/case_2_1.png',
      smi='*c1ccc2c(c1)C(=O)N(c1ccc([SH](=O)(c3ccccc3)c3ccccc3)c(N3C(=O)c4ccc(C(*)(C(F)(F)F)C(F)(F)F)cc4C3=O)c1)C2=O',
      sets=[([('Nc1ccc([SH](=O)(c2ccccc2)c2ccccc2)c(N)c1','UNK',3.50),('O=C1OC(=O)c2cc(C(c3ccc4c(c3)C(=O)OC4=O)(C(F)(F)F)C(F)(F)F)ccc21','70677',2.90)],3.17,'dash')]),
 dict(name='Candidate 2', radar=f'{RADAR}/case_2_2.png',
      smi='*c1ccc(-c2ccc(-c3cc(*)c(Cl)cc3Cl)cc2)cc1',
      sets=[([],None,'none')]),
 dict(name='Candidate 3', radar=f'{RADAR}/case_2_3.png',
      smi='*c1ccc2c(c1)C(=O)N(c1cccc(C(F)(F)C(F)(F)C(F)(F)c3cccc(N4C(=O)c5ccc(C(*)(C(F)(F)F)C(F)(F)F)cc5C4=O)c3)c1)C2=O',
      sets=[([('Nc1cccc(C(F)(F)C(F)(F)C(F)(F)c2cccc(N)c2)c1','19993866',2.61),('O=C1OC(=O)c2cc(C(c3ccc4c(c3)C(=O)OC4=O)(C(F)(F)F)C(F)(F)F)ccc21','70677',2.90)],2.75,'green')]),
]
CLASSIC = {'Nc1ccc([SH](=O)(c2ccccc2)c2ccccc2)c(N)c1'}   # CoordGen overlaps its sulfoxide labels
def oriented(smi, angle_deg=90):
    m = Chem.MolFromSmiles(smi)
    rdDepictor.SetPreferCoordGen(smi not in CLASSIC); rdDepictor.Compute2DCoords(m); rdDepictor.SetPreferCoordGen(True)
    conf = m.GetConformer(); xy = np.array([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y] for i in range(m.GetNumAtoms())])
    xy -= xy.mean(0); u, s, vt = np.linalg.svd(xy, full_matrices=False)
    ax = vt[0]; th = np.deg2rad(angle_deg) - np.arctan2(ax[1], ax[0])
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]); xy = xy @ R.T
    for i, (x, y) in enumerate(xy): conf.SetAtomPosition(i, (float(x), float(y), 0.0))
    b = np.mean([np.linalg.norm(xy[a.GetBeginAtomIdx()] - xy[a.GetEndAtomIdx()]) for a in m.GetBonds()])
    ext = (xy.max(0) - xy.min(0)) / b + 2.0   # in bond lengths, incl. label room
    return m, ext
def render(m, ext, L):
    """Draw with bond length L inches; return RGBA image and its size in inches."""
    w, h = int(round(ext[0] * L * DPI)), int(round(ext[1] * L * DPI))
    d = rdMolDraw2D.MolDraw2DCairo(w, h); o = d.drawOptions()
    o.fixedBondLength = L * DPI; o.padding = 0.02; o.bondLineWidth = max(2, int(L * DPI / 22)); o.clearBackground = True
    o.minFontSize = int(L * DPI * 0.66); o.maxFontSize = int(L * DPI * 0.72); o.additionalAtomLabelPadding = 0.1
    d.DrawMolecule(m); d.FinishDrawing()
    from PIL import ImageChops
    im = Image.open(io.BytesIO(d.GetDrawingText())).convert('RGB'); bb = ImageChops.difference(im, Image.new('RGB', im.size, 'white')).getbbox(); im = im.crop(bb)
    return im, im.size[0] / DPI, im.size[1] / DPI
W = 7.8; cw = W / 3
H_head, H_radar, H_struct, H_smi, H_arrow, H_box, H_ps = 0.24, 1.30, 1.38, 0.30, 0.28, 3.05, 0.20
H = H_head + H_radar + H_struct + H_smi + H_arrow + H_box + H_ps + 0.06
gap = 0.05; LAB = 0.30   # label block under each molecule
# --- one bond length for every monomer: largest that fits every box
mons = {}
Lmax = 1.0
for c in C:
    n = len(c['sets']); bw = min((cw - 0.12 - gap * (n - 1)) / n, 1.5)
    for mols, _, _ in c['sets']:
        if not mols: continue
        exts = []
        for smi, _, _ in mols:
            m, e = oriented(smi, 90); mons[smi] = (m, e); exts.append(e)
        inner_w, inner_h = bw - 0.08, H_box - 0.10 - LAB * len(mols) - 0.06 * (len(mols) - 1)
        Lmax = min(Lmax, inner_w / max(e[0] for e in exts), inner_h / sum(e[1] for e in exts))
L_mon = Lmax
print(f'monomer bond length = {L_mon:.3f} in ({L_mon*25.4:.2f} mm)')
fig = plt.figure(figsize=(W, H), dpi=DPI); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis('off')
def place(im, wi, hi, cx, cy):
    ax.imshow(im, extent=(cx - wi / 2, cx + wi / 2, cy - hi / 2, cy + hi / 2), interpolation='none', zorder=2)
mono = dict(family='DejaVu Sans Mono')
for k, c in enumerate(C):
    x0 = k * cw; cx = x0 + cw / 2; y = H - 0.03
    if k: ax.plot([x0, x0], [0.05, H - 0.05], color='0.35', lw=1.0)
    ax.text(x0 + 0.12, y - H_head / 2, c['name'], fontsize=10, family='serif', va='center', ha='left',
            bbox=dict(boxstyle='square,pad=0.25', fc='0.88', ec='0.4', lw=0.6))
    y -= H_head
    r0 = Image.open(c['radar']).convert('RGBA'); r = Image.new('RGBA', r0.size, 'white'); r.alpha_composite(r0); r = r.convert('RGB'); place(r, H_radar * r.size[0] / r.size[1], H_radar, cx, y - H_radar / 2); y -= H_radar
    ang = np.rad2deg(np.arctan2(H_struct, cw - 0.25))
    m, e = oriented(c['smi'], ang); L = min((cw - 0.2) / e[0], (H_struct - 0.05) / e[1])
    im, wi, hi = render(m, e, L); place(im, wi, hi, cx, y - H_struct / 2); y -= H_struct
    s = c['smi']; lines = [s[i:i + 40] for i in range(0, len(s), 40)]
    ax.text(cx, y - H_smi / 2, '\n'.join(lines), fontsize=6.3, ha='center', va='center', linespacing=1.15, **mono); y -= H_smi
    n = len(c['sets']); bw = min((cw - 0.12 - gap * (n - 1)) / n, 1.5)
    bx0 = cx - (n * bw + gap * (n - 1)) / 2; ytop_box = y - H_arrow
    for j in range(n):
        bcx = bx0 + j * (bw + gap) + bw / 2
        ax.add_patch(FancyArrowPatch((cx, y - 0.02), (bcx, ytop_box + 0.02), arrowstyle='-|>,head_width=2.2,head_length=3.5',
                                     mutation_scale=1, lw=1.1, color='k', shrinkA=0, shrinkB=0, zorder=3))
    y = ytop_box
    for j, (mols, ps, style) in enumerate(c['sets']):
        bxl = bx0 + j * (bw + gap)
        ec, ls, lw = {'dash': ('0.35', (0, (3, 2)), 0.7), 'none': ('0.35', (0, (3, 2)), 0.7), 'solid': ('0.25', '-', 0.9), 'green': ('#1a9a3a', '-', 2.2)}[style]
        ax.add_patch(Rectangle((bxl, y - H_box), bw, H_box, fill=False, ec=ec, ls=ls, lw=lw, zorder=1))
        if not mols:   # no route found: crossed-out box, as in the original figure
            ax.plot([bxl, bxl + bw], [y, y - H_box], color='0.2', lw=0.7); ax.plot([bxl, bxl + bw], [y - H_box, y], color='0.2', lw=0.7)
            continue
        items = [(render(*mons[smi], L_mon), cid, sa) for smi, cid, sa in mols]
        tot = sum(it[0][2] + LAB for it in items) + 0.06 * (len(items) - 1)
        yy = y - (H_box - tot) / 2
        for (im, wi, hi), cid, sa in items:
            place(im, wi, hi, bxl + bw / 2, yy - hi / 2); yy -= hi
            ax.text(bxl + bw / 2, yy - LAB / 2, f'CID={cid}\nSAScore={sa:.2f}', fontsize=6.3, ha='center', va='center', linespacing=1.2, **mono)
            yy -= LAB + 0.06
        ax.text(bxl + bw / 2, y - H_box - H_ps / 2 - 0.02, f'PolyScore={ps:.2f}', fontsize=6.3, ha='center', va='center',
                weight='bold' if style == 'green' else 'normal', **mono)
fig.savefig(OUT + '.pdf', dpi=DPI); fig.savefig(OUT + '.png', dpi=110)
print('wrote', OUT, f'{W:.2f} x {H:.2f} in')
