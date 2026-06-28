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


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "resultados_diagnostico"
OUTPUT_DIR = ROOT / "output" / "pdf"
OUTPUT = OUTPUT_DIR / "actualizacion_hybrid_bl_vs_sin_bl.pdf"

ANNUAL = RESULTS / "max_weight_hybrid_vs_sin_bl_metricas_anuales.csv"
SUMMARY = RESULTS / "max_weight_hybrid_vs_sin_bl_resumen.csv"

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
MODEL_LABELS = {
    "hybrid_bl": "Con BL nuevo",
    "sin_bl": "Sin BL",
    "delta_hybrid_minus_sin_bl": "Delta BL - sin BL",
}


def pct(x, decimals=1):
    if pd.isna(x):
        return "-"
    return f"{x * 100:.{decimals}f}%"


def num(x, decimals=2):
    if pd.isna(x):
        return "-"
    return f"{x:.{decimals}f}"


def weight_label(w):
    return "2,5%" if abs(w - 0.025) < 1e-9 else f"{w * 100:g}%"


def profile_label(perfil):
    return PROFILE_LABELS.get(perfil, str(perfil))


def model_label(modelo):
    return MODEL_LABELS.get(modelo, str(modelo))


def table_style(header_color="#0f172a", band_color="#f8fafc", font_size=7.4, highlight_col=2):
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
        ("TOPPADDING", (0, 0), (-1, -1), 3.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 3.5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3.5),
    ]
    if highlight_col is not None:
        commands.extend([
            ("BACKGROUND", (highlight_col, 1), (highlight_col, -1), colors.HexColor("#ecfdf5")),
            ("TEXTCOLOR", (highlight_col, 1), (highlight_col, -1), colors.HexColor("#065f46")),
        ])
    return TableStyle(commands)


def metric_table(df, modelo, value_col, title, formatter=pct, header_color="#1f2937"):
    sub = df[df["modelo"] == modelo]
    pivot = sub.pivot(index="perfil", columns="max_weight", values=value_col)
    data = [[title] + [weight_label(w) for w in WEIGHTS]]
    for perfil in PROFILE_ORDER:
        row = [profile_label(perfil)]
        for w in WEIGHTS:
            row.append(formatter(pivot.loc[perfil, w]))
        data.append(row)
    table = Table(data, colWidths=[4.1 * cm] + [2.0 * cm] * len(WEIGHTS), repeatRows=1)
    table.setStyle(table_style(header_color=header_color))
    return table


def annual_table(df, modelo, perfil, value_col="retorno_anual", title=None):
    sub = df[(df["modelo"] == modelo) & (df["perfil"] == perfil)]
    pivot = sub.pivot(index="anio", columns="max_weight", values=value_col).sort_index()
    data = [[title or profile_label(perfil)] + [weight_label(w) for w in WEIGHTS]]
    for year, row_values in pivot.iterrows():
        data.append([str(int(year))] + [pct(row_values[w]) for w in WEIGHTS])
    table = Table(data, colWidths=[2.5 * cm] + [2.0 * cm] * len(WEIGHTS), repeatRows=1)
    table.setStyle(table_style(header_color="#1e3a8a", font_size=7.3))
    return table


def make_annual_delta(annual):
    keys = ["max_weight", "anio", "perfil"]
    left = annual[annual["modelo"] == "hybrid_bl"][keys + ["retorno_anual"]].rename(
        columns={"retorno_anual": "retorno_bl"}
    )
    right = annual[annual["modelo"] == "sin_bl"][keys + ["retorno_anual"]].rename(
        columns={"retorno_anual": "retorno_sin_bl"}
    )
    delta = left.merge(right, on=keys, how="inner")
    delta["modelo"] = "delta_hybrid_minus_sin_bl"
    delta["retorno_anual"] = delta["retorno_bl"] - delta["retorno_sin_bl"]
    return delta[["modelo", *keys, "retorno_anual"]]


def add_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#64748b"))
    canvas.drawString(1.2 * cm, 1.0 * cm, "Fintuc prelim - BL nuevo vs sin BL")
    canvas.drawRightString(28.5 * cm, 1.0 * cm, f"Pagina {canvas.getPageNumber()}")
    canvas.restoreState()


def keep(title_text, h_style, flowable):
    return KeepTogether([Paragraph(title_text, h_style), flowable, Spacer(1, 0.20 * cm)])


def add_metric_block(story, h1, body, summary, metric_name, value_col, formatter, header_color):
    story.append(Paragraph(metric_name, h1))
    story.append(Paragraph("Delta = Con BL nuevo - Sin BL.", body))
    for modelo in ["hybrid_bl", "sin_bl", "delta_hybrid_minus_sin_bl"]:
        story.append(keep(
            model_label(modelo),
            h1,
            metric_table(summary, modelo, value_col, "Perfil", formatter, header_color),
        ))


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    annual = pd.read_csv(ANNUAL)
    summary = pd.read_csv(SUMMARY)

    annual = annual[
        (annual["train_years"] == 5)
        & (annual["max_weight"].round(6).isin([round(w, 6) for w in WEIGHTS]))
    ].copy()
    summary = summary[
        summary["max_weight"].round(6).isin([round(w, 6) for w in WEIGHTS])
    ].copy()
    annual_delta = make_annual_delta(annual)

    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "TitleCustom",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=19,
        leading=23,
        textColor=colors.HexColor("#0f172a"),
        alignment=TA_CENTER,
        spaceAfter=9,
    )
    h1 = ParagraphStyle(
        "H1Custom",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=12.2,
        leading=15,
        textColor=colors.HexColor("#0f172a"),
        spaceBefore=5,
        spaceAfter=6,
    )
    body = ParagraphStyle(
        "BodyCustom",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=8.8,
        leading=11.5,
        textColor=colors.HexColor("#334155"),
        spaceAfter=4,
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

    story = [
        Paragraph("Actualizacion de Resultados: BL nuevo vs sin BL", title),
        Paragraph(f"Generado el {datetime.now().strftime('%d-%m-%Y')}.", body),
        Paragraph("Configuracion de la corrida", h1),
    ]
    for text in [
        "BL nuevo = robust_factor_hybrid: 65% factor BL y 35% asset momentum.",
        "Sin BL = optimizacion con retornos historicos, sin ajuste Black-Litterman.",
        "Lambda = 0, train window = 5 anos, full_invest = True.",
        "max_weight comparados = 1%, 2,5%, 5%, 10%, 15% y 20%.",
        "Retornos anuales independientes: 2019 a 2025. MAE: 2019 a 2024.",
        "La columna 2,5% aparece destacada porque fue la restriccion recomendada.",
    ]:
        story.append(Paragraph(f"- {text}", bullet))
    story.append(Paragraph(
        "Lectura del delta: positivo favorece al BL en retorno y aceptacion; negativo favorece al BL en MAE, abandono y drawdown.",
        body,
    ))

    story.append(PageBreak())
    story.append(Paragraph("Retorno Promedio", title))
    add_metric_block(story, h1, body, summary, "Retorno promedio por perfil", "retorno_promedio", pct, "#1e40af")

    story.append(PageBreak())
    story.append(Paragraph("Error Predictivo", title))
    add_metric_block(story, h1, body, summary, "MAE por perfil", "mae", pct, "#854d0e")

    story.append(PageBreak())
    story.append(Paragraph("Probabilidades", title))
    add_metric_block(story, h1, body, summary, "Probabilidad de aceptacion", "prob_aceptacion_promedio", pct, "#166534")
    story.append(PageBreak())
    add_metric_block(story, h1, body, summary, "Probabilidad de abandono", "prob_abandono_promedio", pct, "#991b1b")
    story.append(PageBreak())
    add_metric_block(story, h1, body, summary, "Tasa de abandono observada", "abandono_promedio", pct, "#7f1d1d")

    story.append(PageBreak())
    story.append(Paragraph("Drawdown Promedio", title))
    add_metric_block(story, h1, body, summary, "Max drawdown promedio", "drawdown_promedio", pct, "#334155")

    story.append(PageBreak())
    story.append(Paragraph("Retornos por Ano - Con BL Nuevo", title))
    story.append(Paragraph("Cada ano parte nuevamente desde capital inicial.", body))
    for idx, perfil in enumerate(PROFILE_ORDER):
        story.append(keep(f"Con BL nuevo - {profile_label(perfil)}", h1, annual_table(annual, "hybrid_bl", perfil)))
        if idx in {1, 3}:
            story.append(PageBreak())

    story.append(PageBreak())
    story.append(Paragraph("Retornos por Ano - Sin BL", title))
    story.append(Paragraph("Misma configuracion, pero sin aplicar Black-Litterman.", body))
    for idx, perfil in enumerate(PROFILE_ORDER):
        story.append(keep(f"Sin BL - {profile_label(perfil)}", h1, annual_table(annual, "sin_bl", perfil)))
        if idx in {1, 3}:
            story.append(PageBreak())

    story.append(PageBreak())
    story.append(Paragraph("Diferencia Anual: Con BL - Sin BL", title))
    story.append(Paragraph("Delta positivo significa que el BL nuevo tuvo mayor retorno que el modelo sin BL en ese ano.", body))
    for idx, perfil in enumerate(PROFILE_ORDER):
        story.append(keep(
            f"Delta anual - {profile_label(perfil)}",
            h1,
            annual_table(annual_delta, "delta_hybrid_minus_sin_bl", perfil),
        ))
        if idx in {1, 3}:
            story.append(PageBreak())

    story.append(PageBreak())
    story.append(Paragraph("Lectura Para el Informe", title))
    for text in [
        "El 2,5% sigue siendo la restriccion elegida por equilibrio: diversifica, evita concentracion extrema y mantiene retornos competitivos.",
        "El BL nuevo ayuda especialmente al perfil arriesgado en 2,5%, 5%, 10%, 15% y 20%, pero no supera a sin BL en neutro.",
        "En muy arriesgado, el BL nuevo mejora en 5%, pero pierde contra sin BL en los pesos mas altos.",
        "En conservador y muy conservador las diferencias de retorno son casi cero porque las restricciones/probabilidades dominan mas que el ajuste BL.",
        "El MAE debe leerse como error promedio entre prediccion y retorno real mensual; menor MAE es mejor.",
    ]:
        story.append(Paragraph(f"- {text}", bullet))

    sources = [
        ["Archivo", "Contenido usado"],
        ["max_weight_hybrid_vs_sin_bl_metricas_anuales.csv", "Retornos por ano, probabilidades y abandonos por perfil/max_weight/modelo."],
        ["max_weight_hybrid_vs_sin_bl_error_predictivo.csv", "MAE, RMSE, bias e hit rate por perfil/max_weight/modelo."],
        ["max_weight_hybrid_vs_sin_bl_resumen.csv", "Metricas promedio y deltas con BL nuevo menos sin BL."],
    ]
    source_table = Table(sources, colWidths=[8.2 * cm, 15.2 * cm], repeatRows=1)
    source_table.setStyle(table_style(header_color="#334155", font_size=8.0, highlight_col=None))
    story.append(Spacer(1, 0.25 * cm))
    story.append(keep("Archivos fuente", h1, source_table))

    doc.build(story, onFirstPage=add_footer, onLaterPages=add_footer)
    print(OUTPUT)


if __name__ == "__main__":
    main()
