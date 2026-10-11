"""Downloadable review brief: plain-language content, laid out as a branded A4 PDF.

Content comes only from the validated row and the explanation already shown on screen;
nothing here scores, ranks or estimates impact.
"""
from __future__ import annotations

import io
import re
from datetime import datetime
from pathlib import Path

import matplotlib
from fpdf import FPDF, XPos, YPos
from fpdf.enums import MethodReturnValue

from src.llm.payload import certainty_band, fmt_number, format_evidence
from src.llm.phrases import clause

ASSETS = Path(__file__).parent / 'assets'
PRODUCT_ART = {'GP01': 'ev.svg', 'GP02': 'home.svg', 'GP03': 'solar.svg', 'GP04': 'paperless.svg'}
# DejaVu ships with matplotlib (already a requirement) and covers CO₂, bullets and accents.
FONT_DIR = Path(matplotlib.get_data_path()) / 'fonts' / 'ttf'

FOREST, GREEN, LEAF = (24, 60, 50), (22, 99, 76), (95, 154, 92)
INK, MUTED, LINE = (36, 56, 47), (96, 118, 104), (215, 227, 218)
SAGE, SAND, CREAM = (234, 243, 226), (249, 241, 220), (245, 250, 241)


def _sentence(text: str) -> str:
    text = text.strip()
    return text[0].upper() + text[1:] + ('' if text.endswith('.') else '.')


def _date(value) -> str:
    try:
        return datetime.strptime(str(value), '%Y-%m-%d').strftime('%d %B %Y')
    except ValueError:
        return 'date unavailable'


def brief_content(row, insight, steps) -> dict:
    """Everything the brief says, in customer language. Technical identifiers are kept separate."""
    impact = row.get('impact') or {}
    tiles = []
    if impact.get('status') == 'ESTIMATED':
        for field, unit, caption in [('kg_co2e_per_year', 'kg CO₂e', 'emissions avoided'),
                                     ('energy_kwh_per_year', 'kWh', 'less energy bought'),
                                     ('paper_sheets_per_year', 'sheets', 'paper saved')]:
            if impact.get(field) is not None:
                tiles.append((fmt_number(impact[field]), unit, caption))
    return {
        'product': row['product_name'], 'product_code': row['product_code'],
        'customer_id': row['customer_id'], 'prepared': datetime.now().strftime('%d %B %Y'),
        'data_date': _date(row.get('data_timestamp')),
        'headline': (insight or {}).get('headline'), 'text': (insight or {}).get('text'),
        'reasons': [_sentence(clause(code)) for code in row.get('reason_codes') or []],
        'evidence': [(_sentence(e['label'])[:-1], e['value'])
                     for e in (format_evidence(k, v) for k, v in (row.get('evidence') or {}).items() if v is not None)],
        'impact': tiles,
        'method': _sentence(impact['method']) if tiles and impact.get('method') else None,
        'assumptions': list(impact.get('assumptions') or []) if tiles else [],
        'likelihood': f'{row["propensity"]:.0%}', 'confidence': f'{row["confidence"]:.0%}',
        'band': certainty_band(row.get('confidence')),
        'steps': list(steps),
        'notes': ['This brief does not submit an application or change your account settings.',
                  'Being shown this opportunity is not a credit approval. Financial and installation '
                  'decisions need the usual authorised checks.',
                  'Prepared from synthetic customer data for demonstration.'],
        'technical': (f'Model {row.get("model_version")} · Policy {row.get("policy_version", "n/a")} · '
                      f'Data {row.get("data_timestamp")} · Generated {datetime.now().astimezone().isoformat(timespec="seconds")}'),
    }


class _BriefPDF(FPDF):
    def multi_cell(self, *args, **kwargs):
        kwargs.setdefault('align', 'L')  # ragged-right: justified text leaves wide gaps around long tokens
        return super().multi_cell(*args, **kwargs)

    def footer(self):
        self.set_y(-13)
        self.set_font('DejaVu', '', 7.5)
        self.set_text_color(*MUTED)
        self.cell(0, 5, 'Verdant · Green Customer Insights · Estimates are potential, not achieved savings',
                  align='L')
        self.cell(0, 5, f'Page {self.page_no()} of {{nb}}', align='R')


def _pdf() -> _BriefPDF:
    pdf = _BriefPDF(format='A4')
    pdf.add_font('DejaVu', '', str(FONT_DIR / 'DejaVuSans.ttf'))
    pdf.add_font('DejaVu', 'B', str(FONT_DIR / 'DejaVuSans-Bold.ttf'))
    pdf.set_margins(18, 16, 18)
    pdf.set_auto_page_break(True, margin=20)
    pdf.alias_nb_pages()
    pdf.add_page()
    return pdf


def _height(pdf, width, line, text) -> float:
    return pdf.multi_cell(width, line, text, dry_run=True, output=MethodReturnValue.HEIGHT)


def _keep(pdf, needed):
    if pdf.get_y() + needed > pdf.h - pdf.b_margin:
        pdf.add_page()


def _heading(pdf, title):
    _keep(pdf, 22)
    pdf.ln(3.5)
    y = pdf.get_y()
    pdf.set_fill_color(*LEAF)
    pdf.ellipse(pdf.l_margin, y + 1.3, 3.2, 3.2, style='F')
    pdf.set_xy(pdf.l_margin + 5.5, y)
    pdf.set_font('DejaVu', 'B', 11)
    pdf.set_text_color(*FOREST)
    pdf.cell(0, 6, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_draw_color(*LINE)
    pdf.line(pdf.l_margin, pdf.get_y() + 1, pdf.w - pdf.r_margin, pdf.get_y() + 1)
    pdf.ln(2.5)


def _bullet(pdf, text, marker='•', size=9.5):
    pdf.set_font('DejaVu', '', size)
    width = pdf.epw - 7
    _keep(pdf, _height(pdf, width, 5, text))
    y = pdf.get_y()
    pdf.set_text_color(*LEAF)
    pdf.set_xy(pdf.l_margin + 1, y)
    pdf.cell(6, 5, marker)
    pdf.set_text_color(*INK)
    pdf.set_xy(pdf.l_margin + 7, y)
    pdf.multi_cell(width, 5, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(0.6)


def _tiles(pdf, tiles):
    """Equal-width cards: (big value, unit, caption, note)."""
    gap = 4
    width = (pdf.epw - gap * (len(tiles) - 1)) / len(tiles)
    pdf.set_font('DejaVu', '', 8)
    height = 21 + max(_height(pdf, width - 8, 4, note) if note else 0 for *_, note in tiles)
    _keep(pdf, height + 2)
    y = pdf.get_y()
    for i, (value, unit, caption, note) in enumerate(tiles):
        x = pdf.l_margin + i * (width + gap)
        pdf.set_fill_color(*SAGE)
        pdf.rect(x, y, width, height, style='F', round_corners=True, corner_radius=3)
        pdf.set_fill_color(*LEAF)
        pdf.rect(x, y + 3, 1.2, height - 6, style='F')
        pdf.set_xy(x + 5, y + 3.5)
        pdf.set_font('DejaVu', 'B', 17)
        pdf.set_text_color(*FOREST)
        pdf.cell(pdf.get_string_width(value) + 1.5, 9, value)
        pdf.set_font('DejaVu', '', 9)
        pdf.set_text_color(*GREEN)
        pdf.cell(width - 8, 10, unit)
        pdf.set_xy(x + 5, y + 12.5)
        pdf.set_font('DejaVu', 'B', 8.5)
        pdf.set_text_color(*INK)
        pdf.cell(width - 8, 5, caption)
        if note:
            pdf.set_xy(x + 5, y + 18)
            pdf.set_font('DejaVu', '', 8)
            pdf.set_text_color(*MUTED)
            pdf.multi_cell(width - 8, 4, note)
    pdf.set_xy(pdf.l_margin, y + height + 2)


def review_pdf(row, insight, steps) -> bytes:
    c = brief_content(row, insight, steps)
    pdf = _pdf()

    # Header band with the Verdant mark.
    pdf.set_fill_color(*FOREST)
    pdf.rect(0, 0, pdf.w, 30, style='F')
    pdf.set_fill_color(213, 233, 181)
    pdf.rect(18, 9, 12, 12, style='F', round_corners=('TOP_LEFT', 'TOP_RIGHT', 'BOTTOM_RIGHT'), corner_radius=4)
    pdf.set_xy(18, 9.6)
    pdf.set_font('DejaVu', 'B', 15)
    pdf.set_text_color(*FOREST)
    pdf.cell(12, 10, 'v', align='C')
    pdf.set_xy(33, 8.5)
    pdf.set_text_color(*CREAM)
    pdf.set_font('DejaVu', 'B', 16)
    pdf.cell(60, 8, 'verdant')
    pdf.set_xy(33, 16.5)
    pdf.set_font('DejaVu', '', 8)
    pdf.set_text_color(200, 220, 196)
    pdf.cell(60, 5, 'Green Customer Insights')
    pdf.set_xy(pdf.w - 98, 10)
    pdf.set_font('DejaVu', 'B', 10)
    pdf.set_text_color(*CREAM)
    pdf.cell(80, 5, 'Your greener next step', align='R')
    pdf.set_xy(pdf.w - 98, 16)
    pdf.set_font('DejaVu', '', 8)
    pdf.set_text_color(200, 220, 196)
    pdf.cell(80, 5, f'Review brief · {c["prepared"]}', align='R')

    # Title with the product illustration.
    art = ASSETS / PRODUCT_ART.get(c['product_code'], 'home.svg')
    try:
        # fpdf2 warns on the accessibility <title>; the PDF has no use for it.
        svg = re.sub(r'<title>.*?</title>', '', art.read_text(), flags=re.S)
        pdf.image(io.BytesIO(svg.encode()), x=pdf.w - pdf.r_margin - 46, y=36, w=46)
    except Exception:  # illustration is decorative; never block the download
        pass
    pdf.set_xy(pdf.l_margin, 40)
    pdf.set_font('DejaVu', 'B', 22)
    pdf.set_text_color(*FOREST)
    pdf.multi_cell(pdf.epw - 52, 10, c['product'], new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font('DejaVu', '', 9.5)
    pdf.set_text_color(*MUTED)
    pdf.cell(0, 6, f'Prepared for customer {c["customer_id"]}', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.cell(0, 5, f'Based on your data up to {c["data_date"]}', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_y(max(pdf.get_y() + 3, 64))

    # The explanation the customer saw on screen.
    if c['headline']:
        inner = pdf.epw - 14
        pdf.set_font('DejaVu', 'B', 12.5)
        head_h = _height(pdf, inner, 6.5, c['headline'])
        pdf.set_font('DejaVu', '', 10)
        text_h = _height(pdf, inner, 5.6, c['text'] or '')
        y = pdf.get_y()
        pdf.set_fill_color(*SAGE)
        pdf.rect(pdf.l_margin, y, pdf.epw, head_h + text_h + 12, style='F', round_corners=True, corner_radius=4)
        pdf.set_xy(pdf.l_margin + 7, y + 4.5)
        pdf.set_font('DejaVu', 'B', 12.5)
        pdf.set_text_color(*FOREST)
        pdf.multi_cell(inner, 6.5, c['headline'], new_x=XPos.LEFT, new_y=YPos.NEXT)
        pdf.set_xy(pdf.l_margin + 7, pdf.get_y() + 2)
        pdf.set_font('DejaVu', '', 10)
        pdf.set_text_color(*INK)
        pdf.multi_cell(inner, 5.6, c['text'] or '', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_y(y + head_h + text_h + 13)

    _heading(pdf, 'Why we are suggesting this')
    for reason in c['reasons']:
        _bullet(pdf, reason)
    if c['evidence']:
        pdf.ln(1)
        pdf.set_font('DejaVu', 'B', 9)
        pdf.set_text_color(*MUTED)
        pdf.cell(0, 5, 'What we looked at', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(1)
        for label, value in c['evidence']:
            _bullet(pdf, f'{label}: {value}')

    _heading(pdf, 'What it could mean each year')
    if c['impact']:
        _tiles(pdf, [(value, unit, caption, None) for value, unit, caption in c['impact']])
        pdf.set_font('DejaVu', '', 8.5)
        pdf.set_text_color(*MUTED)
        pdf.multi_cell(0, 4.5, 'These are estimates of potential, not guaranteed savings.'
                       + (f' How we worked it out: {c["method"]}' if c['method'] else ''),
                       new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        if c['assumptions']:
            pdf.ln(2)
            pdf.set_font('DejaVu', 'B', 9)
            pdf.set_text_color(*MUTED)
            pdf.cell(0, 5, 'What this assumes', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.ln(1)
            for i, text in enumerate(c['assumptions'], 1):
                _bullet(pdf, text, marker=f'{i}.', size=8.5)
    else:
        _bullet(pdf, 'We do not have enough information to estimate the impact yet, so no saving has been assumed.')

    _heading(pdf, 'How sure are we?')
    _tiles(pdf, [(c['likelihood'], '', 'Likelihood of take-up',
                  'Our model’s estimate of how likely customers like you are to take this up. '
                  'It is not an approval or affordability check.'),
                 (c['confidence'], c['band'], 'Confidence',
                  'How complete and stable the data behind this suggestion is.')])

    _heading(pdf, 'Your next steps')
    for i, step in enumerate(c['steps'], 1):
        _bullet(pdf, step, marker=f'{i}.', size=10)

    # Good to know, on a warm sand panel; technical identifiers close it in small print.
    pdf.set_font('DejaVu', '', 9)
    notes_h = sum(_height(pdf, pdf.epw - 17, 4.8, n) + 1 for n in c['notes'])
    pdf.set_font('DejaVu', '', 7)
    tech = 'For your relationship team: ' + c['technical']
    tech_h = _height(pdf, pdf.epw - 12, 3.6, tech)
    panel_h = notes_h + tech_h + 13
    _keep(pdf, panel_h + 4)
    pdf.ln(3)
    y = pdf.get_y()
    pdf.set_fill_color(*SAND)
    pdf.rect(pdf.l_margin, y, pdf.epw, panel_h, style='F', round_corners=True, corner_radius=4)
    pdf.set_xy(pdf.l_margin + 6, y + 4)
    pdf.set_font('DejaVu', 'B', 10)
    pdf.set_text_color(*FOREST)
    pdf.cell(0, 5, 'Good to know', new_x=XPos.LEFT, new_y=YPos.NEXT)
    pdf.ln(1)
    for note in c['notes']:
        ny = pdf.get_y()
        pdf.set_font('DejaVu', '', 9)
        pdf.set_text_color(173, 123, 32)
        pdf.set_xy(pdf.l_margin + 6, ny)
        pdf.cell(5, 4.8, '•')
        pdf.set_text_color(*INK)
        pdf.set_xy(pdf.l_margin + 11, ny)
        pdf.multi_cell(pdf.epw - 17, 4.8, note, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(1)
    pdf.set_xy(pdf.l_margin + 6, pdf.get_y() + 0.5)
    pdf.set_font('DejaVu', '', 7)
    pdf.set_text_color(*MUTED)
    pdf.multi_cell(pdf.epw - 12, 3.6, tech, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_y(y + panel_h + 2)

    pdf.set_title(f'Verdant review brief: {c["product"]}')
    pdf.set_author('Verdant · Green Customer Insights')
    return bytes(pdf.output())
