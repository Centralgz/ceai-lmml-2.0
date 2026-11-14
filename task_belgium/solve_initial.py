"""Run: python3 solve.py [path/to/potato_fields.csv]

Uses only the Python standard library. Writes an SVG supporting figure.
If Pillow is installed, also writes a PNG. No rows are filtered or rearranged.
"""
import csv
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
source = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / 'potato_fields.csv'
with source.open(newline='', encoding='utf-8-sig') as stream:
    rows = list(csv.DictReader(stream))

# Area = long side * short side; 1 hectare = 10,000 square metres.
x = [2 * (float(r['long_side_m']) + 10000 * float(r['area_ha']) /
          float(r['long_side_m'])) for r in rows]
y = [float(r['edge_m']) for r in rows]
n = len(x)
xbar, ybar = sum(x) / n, sum(y) / n
slope = sum((a-xbar)*(b-ybar) for a, b in zip(x, y)) / sum((a-xbar)**2 for a in x)
intercept = ybar - slope * xbar
fitted = [intercept + slope * a for a in x]
residual = [b-p for b, p in zip(y, fitted)]
r2 = 1 - sum(e*e for e in residual) / sum((b-ybar)**2 for b in y)
rmse = math.sqrt(sum(e*e for e in residual) / n)
summary = (f'n = {n}\nedge_m = {intercept:.9f} + {slope:.9f} * perimeter_m\n'
           f'R2 = {r2:.9f}\nRMSE = {rmse:.6f} m\n'
           'Read the residual-versus-perimeter plot: REGRESSION\n'
           'Flag: SIGMOID_REGRESSION\n')
print(summary)
(HERE / 'results.txt').write_text(summary)

width, height = 1800, 650
left, right, top, bottom = 100, 40, 110, 100
xmin, xmax = min(x)-15, max(x)+15
ymin, ymax = min(residual)-2, max(residual)+2
def position(a, b):
    return (left+(a-xmin)/(xmax-xmin)*(width-left-right),
            top+(ymax-b)/(ymax-ymin)*(height-top-bottom))
parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
         '<rect width="100%" height="100%" fill="white"/>',
         '<g font-family="Arial,sans-serif" fill="#17253c">',
         '<text x="100" y="38" font-size="27">Fry-ndship: the measurements spell REGRESSION</text>',
         f'<text x="100" y="73" font-size="18">OLS: measured edge ~ intercept + geometric perimeter | n = {n} | R² = {r2:.6f}</text>']
for tick in range(300, 1200, 100):
    px, _ = position(tick, 0)
    parts += [f'<line x1="{px}" y1="{top}" x2="{px}" y2="{height-bottom}" stroke="#e6e9ef"/>',
              f'<text x="{px}" y="{height-bottom+30}" text-anchor="middle" font-size="16">{tick}</text>']
for tick in range(-20, 21, 5):
    if not ymin <= tick <= ymax:
        continue
    _, py = position(xmin, tick)
    parts += [f'<line x1="{left}" y1="{py}" x2="{width-right}" y2="{py}" stroke="#e6e9ef"/>',
              f'<text x="{left-14}" y="{py+5}" text-anchor="end" font-size="16">{tick}</text>']
for a, b in zip(x, residual):
    px, py = position(a, b)
    parts.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="1.6" fill="#173f80" fill-opacity="0.8"/>')
parts += [f'<text x="{width/2}" y="{height-32}" text-anchor="middle" font-size="20">Geometric perimeter = 2 × (long side + 10,000 × area_ha / long side), metres</text>',
          f'<text transform="translate(28,{height/2}) rotate(-90)" text-anchor="middle" font-size="20">Measured − fitted edge, metres</text>', '</g></svg>']
(HERE / 'residuals.svg').write_text('\n'.join(parts))

# Optional raster preview; the SVG above is always produced.
try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    pass
else:
    im = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(im)
    font = ImageFont.load_default(size=22)
    draw.text((left, 25), 'Fry-ndship: residuals reveal REGRESSION', fill='#17253c', font=font)
    draw.text((left, 65), f'OLS edge ~ geometric perimeter | n={n} | R2={r2:.6f}', fill='#17253c', font=font)
    for a, b in zip(x, residual):
        px, py = position(a, b)
        draw.ellipse((px-1.5, py-1.5, px+1.5, py+1.5), fill='#173f80')
    draw.text((left, height-65), 'X: geometric perimeter (m). Y: measured minus fitted edge (m).', fill='#17253c', font=font)
    im.save(HERE / 'residuals.png')
