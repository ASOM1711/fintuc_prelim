from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "resultados_diagnostico" / "max_weight_decision_resumen.csv"
OUTPUT_DIR = ROOT / "output" / "pdf"
OUTPUT = OUTPUT_DIR / "analisis_max_weight_resultados.pdf"

WEIGHTS = [0.05, 0.10, 0.15, 0.20, 0.25]
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


def pct(x):
    if pd.isna(x):
        return "-"
    return f"{x * 100:.1f}%"


def num(x):
    if pd.isna(x):
        return "-"
    return f"{x:.1f}"


def metric_table(df, value_col, title, formatter=pct):
    pivot = df.pivot(index="perfil", columns="max_weight", values=value_col)
    data = [[title, "5%", "10%", "15%", "20%", "25%"]]
    for perfil in PROFILE_ORDER:
        row = [PROFILE_LABELS[perfil]]
        for w in WEIGHTS:
            row.append(formatter(pivot.loc[perfil, w]))
        data.append(row)

    table = Table(data, colWidths=[4.4 * cm] + [2.25 * cm] * 5, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2937")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
                ("ALIGN", (0, 0), (-1, 0), "CENTER"),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#cbd5e1")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
                ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


def best_table(df):
    best = df.sort_values("score", ascending=False).groupby("perfil").head(1)
    best = best.set_index("perfil").loc[PROFILE_ORDER].reset_index()
    data = [["Perfil", "Mejor max_weight", "Score", "Retorno prom.", "Drawdown prom.", "MAE pred."]]
    for _, row in best.iterrows():
        data.append(
            [
                PROFILE_LABELS[row["perfil"]],
                pct(row["max_weight"]),
                pct(row["score"]),
                pct(row["retorno_promedio"]),
                pct(row["drawdown_promedio"]),
                pct(row["mae"]),
            ]
        )

    table = Table(data, colWidths=[4.2 * cm, 3.0 * cm, 2.4 * cm, 2.8 * cm, 3.0 * cm, 2.8 * cm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#065f46")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#cbd5e1")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f0fdf4")]),
                ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


def add_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#64748b"))
    page = canvas.getPageNumber()
    canvas.drawRightString(28.7 * cm, 1.0 * cm, f"Pagina {page}")
    canvas.drawString(1.2 * cm, 1.0 * cm, "Analisis max_weight - Fintuc prelim")
    canvas.restoreState()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(INPUT)

    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "TitleCustom",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=22,
        leading=26,
        textColor=colors.HexColor("#0f172a"),
        alignment=TA_CENTER,
        spaceAfter=12,
    )
    h1 = ParagraphStyle(
        "H1Custom",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=18,
        textColor=colors.HexColor("#0f172a"),
        spaceBefore=6,
        spaceAfter=8,
    )
    body = ParagraphStyle(
        "BodyCustom",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor("#334155"),
        spaceAfter=7,
    )
    bullet = ParagraphStyle(
        "BulletCustom",
        parent=body,
        leftIndent=12,
        firstLineIndent=-8,
        spaceAfter=5,
    )

    doc = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=landscape(A4),
        rightMargin=1.2 * cm,
        leftMargin=1.2 * cm,
        topMargin=1.1 * cm,
        bottomMargin=1.4 * cm,
    )

    story = []
    story.append(Paragraph("Analisis de Restriccion de Peso Maximo por Accion", title))
    story.append(
        Paragraph(
            "Objetivo: comparar max_weight = 5%, 10%, 15%, 20% y 25% para decidir si el 15% usado en el modelo se justifica.",
            body,
        )
    )
    story.append(Paragraph("Supuestos de la corrida", h1))
    for text in [
        "Backtests anuales independientes: cada año parte desde capital inicial, aunque el perfil abandone en otro año.",
        "Periodo de metricas: 2019 a 2025, excluyendo 2026 por ser un año parcial. El error predictivo se mide en 2019 a 2024.",
        "Lambda usado: 0. Se evalua la restriccion de max_weight sin penalizacion adicional por varianza.",
        "full_invest=True. Black-Litterman uso pesos iguales porque no se encontro stocks_info.txt en esta maquina.",
    ]:
        story.append(Paragraph(f"- {text}", bullet))

    story.append(Paragraph("Como leer las metricas", h1))
    for text in [
        "Retorno promedio: ganancia o perdida promedio anual del portafolio.",
        "Drawdown promedio: caida promedio desde el maximo previo. Mide dolor/riesgo vivido por el cliente.",
        "MAE predictivo: error absoluto promedio entre retorno predicho y retorno real del portafolio. Menor es mejor.",
        "Abandono promedio: fraccion de años en que el perfil abandono durante el backtest anual.",
        "Score: retorno - 0.5*drawdown - 0.5*MAE - 0.3*abandono. Sirve solo para ordenar alternativas.",
    ]:
        story.append(Paragraph(f"- {text}", bullet))

    story.append(Spacer(1, 0.2 * cm))
    story.append(Paragraph("Mejor alternativa segun score", h1))
    story.append(best_table(df))

    story.append(PageBreak())
    story.append(Paragraph("Tablas de Resultados", title))
    for col, label in [
        ("retorno_promedio", "Retorno promedio"),
        ("drawdown_promedio", "Drawdown promedio"),
    ]:
        story.append(Paragraph(label, h1))
        story.append(metric_table(df, col, label))
        story.append(Spacer(1, 0.35 * cm))

    story.append(PageBreak())
    story.append(Paragraph("Tablas de Resultados", title))
    for col, label in [
        ("mae", "MAE del error predictivo"),
        ("abandono_promedio", "Abandono promedio"),
    ]:
        story.append(Paragraph(label, h1))
        story.append(metric_table(df, col, label))
        story.append(Spacer(1, 0.35 * cm))

    story.append(PageBreak())
    story.append(Paragraph("Tablas de Resultados", title))
    story.append(Paragraph("Score de decision", h1))
    story.append(metric_table(df, "score", "Score de decision"))

    story.append(PageBreak())
    story.append(Paragraph("Conclusiones", title))
    conclusions = [
        "El 15% no aparece como el mejor valor empirico en esta muestra. Funciona como regla intermedia, pero no domina a 5% o 10%.",
        "Para arriesgado, 5% obtuvo mejor balance: mayor retorno promedio, menor drawdown y menor error predictivo.",
        "Para muy arriesgado, 10% obtuvo el mejor balance: retorno promedio mas alto y menor error que 15%, 20% y 25%.",
        "Para conservador y neutro, max_weight casi no cambia los resultados. La restriccion que manda es la tolerancia de riesgo y la probabilidad de abandono.",
        "Si se quiere una regla unica y defendible, 10% parece mas robusto que 15%: reduce concentracion y error predictivo sin sacrificar claramente el retorno.",
        "Si el equipo mantiene 15%, la justificacion debe ser conceptual: balance entre flexibilidad y diversificacion, no superioridad historica.",
    ]
    for text in conclusions:
        story.append(Paragraph(f"- {text}", bullet))

    story.append(Paragraph("Archivos fuente usados", h1))
    for text in [
        "resultados_diagnostico/max_weight_decision_resumen.csv",
        "resultados_diagnostico/max_weight_decision_metricas_anuales.csv",
        "resultados_diagnostico/max_weight_decision_error_predictivo.csv",
    ]:
        story.append(Paragraph(f"- {text}", bullet))

    doc.build(story, onFirstPage=add_footer, onLaterPages=add_footer)
    print(OUTPUT)


if __name__ == "__main__":
    main()
