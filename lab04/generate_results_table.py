#!/usr/bin/env python3
import csv
from pathlib import Path

CSV = Path(__file__).resolve().parent / 'results.csv'
OUT = Path(__file__).resolve().parent / 'results_table.tex'

def map_expected_return(v):
    # thresholds chosen to match representative values (low=3, med=10, high=20)
    if v < 6: return 'niska'
    if v < 16: return 'średnia'
    return 'wysoka'

def map_risk(v):
    if v < 3: return 'niski'
    if v < 7: return 'średni'
    return 'wysoki'

def map_liquidity(v):
    if v < 4: return 'niska'
    if v < 7: return 'średnia'
    return 'wysoka'

def map_attr(v):
    # simple 3-way split
    if v < 33.3333: return 'niska'
    if v < 66.6666: return 'średnia'
    return 'wysoka'

def escape_tex(s):
    # Escape common LaTeX special characters
    if s is None:
        return ''
    repl = {
        '\\': '\\textbackslash{}',
        '&': '\\&',
        '%': '\\%',
        '$': '\\$',
        '#': '\\#',
        '_': '\\_',
        '{': '\\{',
        '}': '\\}',
        '~': '\\textasciitilde{}',
        '^': '\\textasciicircum{}',
    }
    out = str(s)
    for k, v in repl.items():
        out = out.replace(k, v)
    return out

def main():
    if not CSV.exists():
        print('Brak pliku:', CSV)
        return

    rows = []
    with CSV.open(newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                er = float(r['expected_return'])
                risk = float(r['risk'])
                liq = float(r['liquidity'])
                attr = float(r['attractiveness'])
            except Exception as e:
                print('Błąd parsowania wiersza:', r, e)
                continue

            rows.append({
                'case': r.get('case',''),
                'er_label': map_expected_return(er),
                'risk_label': map_risk(risk),
                'liq_label': map_liquidity(liq),
                'attr_label': map_attr(attr),
            })

    with OUT.open('w', encoding='utf8') as out:
        out.write('% Ta tabela została wygenerowana automatycznie przez generate_results_table.py\n')
        out.write('\\begin{table}[h]\n')
        out.write('\\centering\n')
        out.write('\\caption{Wyniki (opisowo) dla modelu uproszczonego}\n')
        out.write('\\label{tab:results_desc}\n')
        out.write('\\begin{tabular}{ccccc}\n')
        out.write('\\toprule\n')
        out.write('Lp. & Stopa zwrotu & Ryzyko & Płynność & Atrakcyjność \\\\ \n')
        out.write('\\midrule\n')
        for r in rows:
            line = f"{r['case']} & {escape_tex(r['er_label'])} & {escape_tex(r['risk_label'])} & {escape_tex(r['liq_label'])} & {escape_tex(r['attr_label'])} \\\\ \n"
            out.write(line)
        out.write('\\bottomrule\n')
        out.write('\\end{tabular}\n')
        out.write('\\end{table}\n')

    print('Wygenerowano', OUT)

if __name__ == '__main__':
    main()
