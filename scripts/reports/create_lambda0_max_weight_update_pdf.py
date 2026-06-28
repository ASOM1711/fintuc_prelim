from datetime import datetime
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "resultados_diagnostico"
OUTPUT_DIR = ROOT / "output" / "pdf"
OUTPUT = OUTPUT_DIR / "actualizacion_lambda0_max_weight.pdf"

ANNUAL = RESULTS / "max_weight_decision_train5_metricas_anuales.csv"
SUMMARY = RESULTS / "max_weight_decision_train5_resumen.csv"

WEIGHTS = [0.01, 0.025, 0.05, 0.10, 0.15, 0.20]
PROFILE_ORDER = [
    "muy_conservador",
    "conservador",
    "neutro",
    "arriesgado",
    "muy_arriesgado",
]
PROFILE_LABELS = {
    "muy_conservador": "Muy conservador",
    "conservador": "Conservador",
    "neutro": "Neutro",
    "arriesgado": "Arriesgado",
    "muy_arriesgado": "Muy arriesgado",
}


def pct(x, decimals=1):
    if pd.isna(x):
        return "-"
    return f"{x * 100:.{decimals}f}%"


def weight_label(w):
    return "2,5%" if abs(w - 0.025) < 1e-9 else f"{w * 100:g}%"


def profile_label(perfil):
    return PROFILE_LABELS.get(perfil, str(perfil))


def table_style(header_color="#0f172a", band_color="#f8fafc", font_size=7.6, highlight_col=None):
    commands = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(header_color)),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ALIGN", (0, 0), (-1, 0), "CENTER"),
        ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#cbd5e1")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor(band_color)]),
        ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]
    if highlight_col is not None:
        commands.extend([
            ("BACKGROUND", (highlight_col, 1), (highlight_col, -1), colors.HexColor("#ecfdf5")),
            ("TEXTCOLOR", (highlight_col, 1), (highlight_col, -1), colors.HexColor("#065f46")),
        ])
    return TableStyle(commands)


def metric_table(df, value_col, title, formatter=pct, header_color="#1f2937"):
    pivot = df.pivot(index="perfil", columns="max_weight", values=value_col)
    data = [[title] + [weight_label(w) for w in WEIGHTS]]
    for perfil in PROFILE_ORDER:
        row = [profile_label(perfil)]
        for w in WEIGHTS:
            row.append(formatter(pivot.loc[perfil, w]))
        data.append(row)
    table = Table(data, colWidths=[4.0 * cm] + [2.0 * cm] * len(WEIGHTS), repeatRows=1)
    table.setStyle(table_style(header_color=header_color, highlight_col=2))
    return table


def annual_return_table(df, perfil):
    sub = df[df["perfil"] == perfil]
    pivot = sub.pivot(index="anio", columns="max_weight", values="retorno_anual").sort_index()
    data = [[profile_label(perfil)] + [weight_label(w) for w in WEIGHTS]]
    for year, row_values in pivot.iterrows():
        data.append([str(int(year))] + [pct(row_values[w]) for w in WEIGHTS])
    table = Table(data, colWidths=[2.5 * cm] + [2.0 * cm] * len(WEIGHTS), repeatRows=1)
    table.setStyle(table_style(header_color="#1e3a8a", font_size=7.5, highlight_col=2))
    return table


def add_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#64748b"))
    canvas.drawString(1.2 * cm, 1.0 * cm, "Fintuc prelim - Lambda 0 y sensibilidad max_weight")
    canvas.drawRightString(28.5 * cm, 1.0 * cm, f"Pagina {canvas.getPageNumber()}")
    canvas.restoreState()


def keep(title_text, h_style, flowable):
    return KeepTogether([Paragraph(title_text, h_style), flowable, Spacer(1, 0.25 * cm)])


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    annual = pd.read_csv(ANNUAL)
    summary = pd.read_csv(SUMMARY)

    annual = annual[
        (annual["train_years"] == 5)
        & (annual["max_weight"].round(6).isin([round(w, 6) for w in WEIGHTS]))
    ].copy()
    summary = summary[
        (summary["train_years"] == 5)
        & (summary["max_weight"].round(6).isin([round(w, 6) for w in WEIGHTS]))
    ].copy()

    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "TitleCustom",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#0f172a"),
        alignment=TA_CENTER,
        spaceAfter=10,
    )
    h1 = ParagraphStyle(
        "H1Custom",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=13.5,
        leading=16,
        textColor=colors.HexColor("#0f172a"),
        spaceBefore=6,
        spaceAfter=7,
    )
    body = ParagraphStyle(
        "BodyCustom",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9.0,
        leading=12,
        textColor=colors.HexColor("#334155"),
        spaceAfter=5,
    )
    bullet = ParagraphStyle("Bullet", parent=body, leftIndent=12, firstLineIndent=-8)

    doc = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=landscape(A4),
        rightMargin=1.2 * cm,
        leftMargin=1.2 * cm,
        topMargin=1.1 * cm,
        bottomMargin=1.5 * cm,
    )

    story = []
    story.append(Paragraph("Actualizacion de Resultados: Lambda 0 y max_weight", title))
    story.append(Paragraph(f"Generado el {datetime.now().strftime('%d-%m-%Y')}.", body))
    story.append(Paragraph("Configuracion de la corrida", h1))
    for text in [
        "Lambda = 0, train window = 5 anos, full_invest = True.",
        "Comparacion de max_weight = 1%, 2,5%, 5%, 10%, 15% y 20%.",
        "Anios evaluados: 2019 a 2025. Se excluye 2026 porque es parcial.",
        "La columna 2,5% aparece destacada porque fue la restriccion recomendada.",
    ]:
        story.append(Paragraph(f"- {text}", bullet))

    story.append(Spacer(1, 0.2 * cm))
    story.append(Paragraph("Comparacion promedio por perfil", h1))
    story.append(Paragraph("Estas tablas resumen el desempeno promedio de los backtests anuales independientes.", body))
    story.append(keep("Retorno promedio", h1, metric_table(summary, "retorno_promedio", "Perfil", header_color="#1e40af")))
    story.append(keep("Error predictivo MAE", h1, metric_table(summary, "mae", "Perfil", header_color="#854d0e")))

    story.append(PageBreak())
    story.append(Paragraph("Probabilidades Promedio", title))
    story.append(Paragraph("Probabilidad de aceptacion y abandono promedio por perfil y restriccion.", body))
    story.append(keep("Probabilidad de aceptacion", h1, metric_table(summary, "prob_aceptacion_promedio", "Perfil", header_color="#166534")))
    story.append(keep("Probabilidad de abandono", h1, metric_table(summary, "prob_abandono_promedio", "Perfil", header_color="#991b1b")))
    story.append(keep("Tasa de abandono observada", h1, metric_table(summary, "abandono_promedio", "Perfil", header_color="#7f1d1d")))

    story.append(PageBreak())
    story.append(Paragraph("Retornos por Ano", title))
    story.append(Paragraph("Cada tabla muestra el retorno anual independiente para un perfil. Cada ano parte nuevamente desde capital inicial.", body))
    for idx, perfil in enumerate(PROFILE_ORDER):
        story.append(keep(f"Retornos anuales - {profile_label(perfil)}", h1, annual_return_table(annual, perfil)))
        if idx in {1, 3}:
            story.append(PageBreak())

    story.append(PageBreak())
    story.append(Paragraph("Lectura Para el Informe", title))
    for text in [
        "El 2,5% mantiene una restriccion diversificada: obliga a tener al menos 40 posiciones si el portafolio esta completamente invertido.",
        "Pesos mayores como 10%, 15% y 20% permiten mas concentracion, pero tienden a subir drawdown y error en perfiles riesgosos.",
        "El 1% reduce concentracion y error en algunos perfiles, pero fuerza al menos 100 posiciones y puede diluir demasiado la senal del modelo.",
        "La eleccion de 2,5% queda como equilibrio entre diversificacion, retorno, error predictivo y explicabilidad.",
    ]:
        story.append(Paragraph(f"- {text}", bullet))

    story.append(Spacer(1, 0.25 * cm))
    sources = [
        ["Archivo", "Contenido usado"],
        ["max_weight_decision_train5_metricas_anuales.csv", "Retornos por ano y probabilidades por perfil/max_weight."],
        ["max_weight_decision_train5_resumen.csv", "Metricas promedio, MAE y comparacion de max_weight."],
    ]
    source_table = Table(sources, colWidths=[8.0 * cm, 10.0 * cm], repeatRows=1)
    source_table.setStyle(table_style(header_color="#334155", font_size=8.0))
    story.append(keep("Archivos fuente", h1, source_table))

    doc.build(story, onFirstPage=add_footer, onLaterPages=add_footer)
    print(OUTPUT)


if __name__ == "__main__":
    main()
