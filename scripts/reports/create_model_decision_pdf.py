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
OUTPUT = OUTPUT_DIR / "informe_decision_parametros_modelo.pdf"

TRAIN_SUMMARY = RESULTS / "train_window_decision_resumen.csv"
MAX_WEIGHT_SUMMARY = RESULTS / "max_weight_decision_train5_resumen.csv"
FINAL_ANNUAL = RESULTS / "train5_maxweight_2_5_anual_independiente.csv"
NO_PANDEMIC = RESULTS / "train5_maxweight_1_2_5_5_10_no_pandemia.csv"
PANDEMIC = RESULTS / "train5_maxweight_1_2_5_5_10_pandemia.csv"

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


def num(x, decimals=1):
    if pd.isna(x):
        return "-"
    return f"{x:.{decimals}f}"


def bool_pct(x):
    if pd.isna(x):
        return "-"
    return "100%" if bool(x) else "0%"


def weight_label(w):
    return f"{w * 100:g}%"


def profile_label(perfil):
    return PROFILE_LABELS.get(perfil, str(perfil))


def styled_table(
    data,
    col_widths,
    header_color="#0f172a",
    band_color="#f8fafc",
    font_size=7.8,
    align_from_col=1,
):
    table = Table(data, colWidths=col_widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(header_color)),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("ALIGN", (0, 0), (-1, 0), "CENTER"),
                ("ALIGN", (align_from_col, 1), (-1, -1), "RIGHT"),
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
        )
    )
    return table


def section(title, h_style, flowable):
    return KeepTogether([Paragraph(title, h_style), flowable, Spacer(1, 0.25 * cm)])


def pivot_metric_table(df, col_key, value_col, columns, title, formatter=pct, header_color="#1f2937"):
    pivot = df.pivot(index="perfil", columns=col_key, values=value_col)
    data = [[title] + [str(c) if col_key == "train_years" else weight_label(c) for c in columns]]
    for perfil in PROFILE_ORDER:
        row = [profile_label(perfil)]
        for c in columns:
            row.append(formatter(pivot.loc[perfil, c]) if c in pivot.columns else "-")
        data.append(row)
    col_widths = [4.2 * cm] + [2.2 * cm] * len(columns)
    return styled_table(data, col_widths, header_color=header_color)


def train_best_table(df):
    best = df.sort_values(["perfil", "score"], ascending=[True, False]).groupby("perfil", as_index=False).first()
    best = best.set_index("perfil").loc[PROFILE_ORDER].reset_index()
    data = [["Perfil", "Mejor ventana", "Score", "Retorno", "Drawdown", "MAE pred.", "N activos"]]
    for _, row in best.iterrows():
        data.append(
            [
                profile_label(row["perfil"]),
                f"{int(row['train_years'])} anos",
                pct(row["score"]),
                pct(row["retorno_anual"]),
                pct(row["max_drawdown"]),
                pct(row["mae"]),
                num(row["n_activos_promedio"], 0),
            ]
        )
    return styled_table(data, [4.0 * cm, 2.5 * cm, 2.2 * cm, 2.2 * cm, 2.4 * cm, 2.2 * cm, 2.2 * cm], "#065f46")


def max_weight_best_table(df):
    best = df.sort_values(["perfil", "score"], ascending=[True, False]).groupby("perfil", as_index=False).first()
    best = best.set_index("perfil").loc[PROFILE_ORDER].reset_index()
    data = [["Perfil", "Mejor restriccion", "Score", "Retorno prom.", "Drawdown prom.", "MAE pred.", "N activos"]]
    for _, row in best.iterrows():
        data.append(
            [
                profile_label(row["perfil"]),
                weight_label(row["max_weight"]),
                pct(row["score"]),
                pct(row["retorno_promedio"]),
                pct(row["drawdown_promedio"]),
                pct(row["mae"]),
                num(row["n_activos_promedio"], 0),
            ]
        )
    return styled_table(data, [4.0 * cm, 2.8 * cm, 2.1 * cm, 2.4 * cm, 2.5 * cm, 2.1 * cm, 2.2 * cm], "#065f46")


def candidate_table():
    rows = [["Restriccion", "Minimo teorico de acciones", "Lectura"]]
    for w, lectura in [
        (0.01, "Muy diversificado; puede diluir conviccion del modelo."),
        (0.025, "Compromiso final: diversifica sin forzar 100 acciones."),
        (0.05, "Mas flexible; buen resultado en neutro/no pandemia."),
        (0.10, "Permite concentracion relevante."),
        (0.15, "Restriccion original; mas concentrada que 2,5%."),
        (0.20, "Muy flexible; aumenta concentracion y drawdown."),
    ]:
        rows.append([weight_label(w), f"{int(round(1 / w))}", lectura])
    return styled_table(rows, [3.0 * cm, 4.0 * cm, 12.0 * cm], "#334155", font_size=8.0, align_from_col=1)


def annual_table(df, value_col, title, formatter=pct, years=None, header_color="#1e3a8a"):
    if years is None:
        years = sorted(df["anio"].unique())
    pivot = df.pivot(index="perfil", columns="anio", values=value_col)
    data = [[title] + [str(y) for y in years]]
    for perfil in PROFILE_ORDER:
        row = [profile_label(perfil)]
        for y in years:
            row.append(formatter(pivot.loc[perfil, y]) if y in pivot.columns else "-")
        data.append(row)
    return styled_table(data, [4.0 * cm] + [2.0 * cm] * len(years), header_color, font_size=7.7)


def final_average_table(df):
    avg = (
        df.groupby("perfil")
        .agg(
            retorno=("retorno_anual", "mean"),
            drawdown=("max_drawdown", "mean"),
            abandono=("abandono", "mean"),
            aceptacion=("frecuencia_rebalanceo", "mean"),
            p2=("prob_aceptacion_prom", "mean"),
            p1=("prob_abandono_prom", "mean"),
            meses=("meses_activos", "mean"),
        )
        .reset_index()
        .set_index("perfil")
        .loc[PROFILE_ORDER]
        .reset_index()
    )
    data = [["Perfil", "Retorno prom.", "Drawdown prom.", "Abandono obs.", "Aceptacion obs.", "Prob. acept.", "Prob. abandono", "Meses activos"]]
    for _, row in avg.iterrows():
        data.append(
            [
                profile_label(row["perfil"]),
                pct(row["retorno"]),
                pct(row["drawdown"]),
                pct(row["abandono"]),
                pct(row["aceptacion"]),
                pct(row["p2"]),
                pct(row["p1"]),
                num(row["meses"], 1),
            ]
        )
    return styled_table(data, [3.7 * cm, 2.2 * cm, 2.2 * cm, 2.2 * cm, 2.3 * cm, 2.2 * cm, 2.4 * cm, 2.0 * cm], "#7c2d12", font_size=7.4)


def period_average_table(df, title, metric, weights, formatter=pct, best_high=True, header_color="#4c1d95"):
    avg = df.groupby(["perfil", "max_weight"], as_index=False)[metric].mean()
    pivot = avg.pivot(index="perfil", columns="max_weight", values=metric)
    data = [[title] + [weight_label(w) for w in weights] + ["Mejor"]]
    for perfil in PROFILE_ORDER:
        values = [pivot.loc[perfil, w] for w in weights]
        best_i = max(range(len(values)), key=lambda i: values[i]) if best_high else min(range(len(values)), key=lambda i: values[i])
        row = [profile_label(perfil)] + [formatter(v) for v in values] + [weight_label(weights[best_i])]
        data.append(row)
    return styled_table(data, [4.0 * cm] + [2.2 * cm] * len(weights) + [2.1 * cm], header_color, font_size=7.7)


def pandemic_worst_table(df, weights):
    worst = df.groupby(["perfil", "max_weight"], as_index=False)["retorno_anual"].min()
    pivot = worst.pivot(index="perfil", columns="max_weight", values="retorno_anual")
    data = [["Peor retorno anual pandemia"] + [weight_label(w) for w in weights] + ["Menor perdida"]]
    for perfil in PROFILE_ORDER:
        values = [pivot.loc[perfil, w] for w in weights]
        best_i = max(range(len(values)), key=lambda i: values[i])
        data.append([profile_label(perfil)] + [pct(v) for v in values] + [weight_label(weights[best_i])])
    return styled_table(data, [4.0 * cm] + [2.2 * cm] * len(weights) + [2.1 * cm], "#991b1b", font_size=7.7)


def add_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#64748b"))
    page = canvas.getPageNumber()
    canvas.drawString(1.2 * cm, 1.0 * cm, "Fintuc prelim - Decision de parametros")
    canvas.drawRightString(28.5 * cm, 1.0 * cm, f"Pagina {page}")
    canvas.restoreState()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(TRAIN_SUMMARY)
    maxw = pd.read_csv(MAX_WEIGHT_SUMMARY)
    annual = pd.read_csv(FINAL_ANNUAL)
    no_pandemic = pd.read_csv(NO_PANDEMIC)
    pandemic = pd.read_csv(PANDEMIC)

    train_years = sorted(train["train_years"].unique())
    max_weights = sorted(maxw["max_weight"].unique())
    comparison_weights = [0.01, 0.025, 0.05, 0.10]
    years = sorted(annual["anio"].unique())

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
        leading=12.2,
        textColor=colors.HexColor("#334155"),
        spaceAfter=5,
    )
    bullet = ParagraphStyle(
        "BulletCustom",
        parent=body,
        leftIndent=12,
        firstLineIndent=-8,
        spaceAfter=4,
    )

    doc = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=landscape(A4),
        rightMargin=1.2 * cm,
        leftMargin=1.2 * cm,
        topMargin=1.1 * cm,
        bottomMargin=1.5 * cm,
    )

    story = []
    story.append(Paragraph("Informe de Decision: Ventana de Training y Restriccion por Accion", title))
    story.append(
        Paragraph(
            f"Generado el {datetime.now().strftime('%d-%m-%Y')}. El informe resume los tests hechos desde la eleccion de ventana de entrenamiento hasta la recomendacion final de max_weight = 2,5%.",
            body,
        )
    )
    story.append(Paragraph("Decision final", h1))
    for text in [
        "Ventana de entrenamiento recomendada: 5 anos.",
        "Restriccion recomendada por accion: max_weight = 2,5%.",
        "Lambda usado en estas corridas: 0. La penalizacion adicional no movia los resultados relevantes bajo estas restricciones.",
        "Backtesting anual independiente: cada ano parte de nuevo, para ver el desempeno por ano aun si un perfil abandono en otro periodo.",
        "Periodo evaluado: 2019 a 2025. Se excluye 2026 porque los datos llegan solo hasta marzo y seria un ano parcial.",
    ]:
        story.append(Paragraph(f"- {text}", bullet))

    story.append(Paragraph("Como leer las metricas", h1))
    for text in [
        "Retorno: ganancia o perdida anual del portafolio. Mayor es mejor.",
        "Drawdown: caida desde el maximo previo hasta el minimo posterior. Menor drawdown significa menos dolor para el cliente.",
        "MAE predictivo: error absoluto promedio entre retorno predicho y retorno real del portafolio. Menor es mejor.",
        "Aceptacion observada: frecuencia con que el cliente acepta/rebalancea segun la simulacion.",
        "Abandono observado: fraccion de anos en que el perfil abandono antes de terminar el ano.",
        "Score: retorno - 0,5*drawdown - 0,5*MAE - 0,3*abandono. Se usa solo como criterio de ordenamiento.",
    ]:
        story.append(Paragraph(f"- {text}", bullet))
    story.append(Spacer(1, 0.15 * cm))
    story.append(section("Mejor ventana de entrenamiento segun score", h1, train_best_table(train)))

    story.append(PageBreak())
    story.append(Paragraph("1. Test de Ventana de Entrenamiento", title))
    story.append(
        Paragraph(
            "Se compararon ventanas de 2 a 6 anos usando lambda = 0, max_weight = 15% y full_invest = True. El objetivo era decidir cuanta historia usar antes de volver a estimar el modelo.",
            body,
        )
    )
    story.append(section("Retorno anual por ventana", h1, pivot_metric_table(train, "train_years", "retorno_anual", train_years, "Perfil")))
    story.append(section("Drawdown por ventana", h1, pivot_metric_table(train, "train_years", "max_drawdown", train_years, "Perfil")))
    story.append(section("MAE predictivo por ventana", h1, pivot_metric_table(train, "train_years", "mae", train_years, "Perfil")))

    story.append(PageBreak())
    story.append(Paragraph("2. Test de Restriccion max_weight con Training de 5 Anos", title))
    story.append(
        Paragraph(
            "Despues de elegir 5 anos de training, se compararon restricciones de 1%, 2,5%, 5%, 10%, 15% y 20%. Una restriccion mas baja fuerza mas diversificacion; una mas alta permite posiciones mas concentradas.",
            body,
        )
    )
    story.append(section("Lectura de cada restriccion", h1, candidate_table()))
    story.append(section("Mejor restriccion por perfil segun score", h1, max_weight_best_table(maxw)))

    story.append(PageBreak())
    story.append(Paragraph("Resultados Promedio por Restriccion", title))
    story.append(section("Retorno promedio", h1, pivot_metric_table(maxw, "max_weight", "retorno_promedio", max_weights, "Perfil", header_color="#1e40af")))
    story.append(section("Drawdown promedio", h1, pivot_metric_table(maxw, "max_weight", "drawdown_promedio", max_weights, "Perfil", header_color="#7f1d1d")))
    story.append(section("MAE predictivo", h1, pivot_metric_table(maxw, "max_weight", "mae", max_weights, "Perfil", header_color="#854d0e")))

    story.append(PageBreak())
    story.append(Paragraph("3. Resultados Anuales con max_weight = 2,5%", title))
    story.append(
        Paragraph(
            "Esta seccion muestra la configuracion final: 5 anos de training, lambda = 0 y max_weight = 2,5%. Cada ano es independiente.",
            body,
        )
    )
    story.append(section("Resumen promedio final", h1, final_average_table(annual)))
    story.append(section("Retorno anual por perfil", h1, annual_table(annual, "retorno_anual", "Perfil", years=years)))
    story.append(section("Drawdown anual por perfil", h1, annual_table(annual, "max_drawdown", "Perfil", years=years, header_color="#7f1d1d")))

    story.append(PageBreak())
    story.append(Paragraph("Aceptacion y Abandono con max_weight = 2,5%", title))
    story.append(section("Aceptacion observada por ano", h1, annual_table(annual, "frecuencia_rebalanceo", "Perfil", years=years, header_color="#166534")))
    story.append(section("Abandono observado por ano", h1, annual_table(annual, "abandono", "Perfil", formatter=bool_pct, years=years, header_color="#991b1b")))
    story.append(section("Probabilidad promedio de aceptacion", h1, annual_table(annual, "prob_aceptacion_prom", "Perfil", years=years, header_color="#0369a1")))
    story.append(section("Probabilidad promedio de abandono", h1, annual_table(annual, "prob_abandono_prom", "Perfil", years=years, header_color="#9f1239")))

    story.append(PageBreak())
    story.append(Paragraph("4. Comparacion No Pandemia y Pandemia", title))
    story.append(
        Paragraph(
            "Para entender si la decision depende de anos extremos, se separaron anos no pandemia (2019, 2023, 2024, 2025) y pandemia/choque (2020, 2021, 2022).",
            body,
        )
    )
    story.append(section("Retorno promedio en anos no pandemia", h1, period_average_table(no_pandemic, "Perfil", "retorno_anual", comparison_weights, header_color="#155e75")))
    story.append(section("Retorno promedio en pandemia", h1, period_average_table(pandemic, "Perfil", "retorno_anual", comparison_weights, header_color="#7c2d12")))
    story.append(section("Peor retorno anual en pandemia", h1, pandemic_worst_table(pandemic, comparison_weights)))
    story.append(section("Drawdown promedio en pandemia", h1, period_average_table(pandemic, "Perfil", "max_drawdown", comparison_weights, best_high=False, header_color="#6b21a8")))

    story.append(PageBreak())
    story.append(Paragraph("5. Conclusiones", title))
    for text in [
        "El test de ventana muestra que 5 anos es la mejor decision global: mejora los perfiles arriesgado, muy arriesgado y conservador, aunque neutro tenga su mejor score puntual con 4 anos.",
        "Usar 5 anos no impide empezar en 2019: para entrenar 2019 se usa 2014-2018, y hay datos suficientes para casi todo el universo.",
        "La restriccion de 2,5% queda como compromiso: 1% baja el riesgo y ayuda al perfil arriesgado, pero fuerza al menos 100 posiciones; 2,5% mantiene alta diversificacion, exige al menos 40 posiciones y es el mejor score para muy arriesgado.",
        "En anos no pandemia, 2,5% es competitivo: gana en muy arriesgado y queda muy cerca de la mejor alternativa en neutro y arriesgado.",
        "En pandemia, las restricciones mas altas a veces capturan mejor recuperaciones, pero tambien permiten mas concentracion. Por estabilidad y explicabilidad, 2,5% es mas defendible que volver a 15% o 20%.",
        "La recomendacion final para presentar es: lambda = 0, ventana de training = 5 anos, max_weight = 2,5%, backtesting anual independiente 2019-2025.",
    ]:
        story.append(Paragraph(f"- {text}", bullet))

    sources = [
        ["Archivo", "Contenido usado"],
        ["train_window_decision_resumen.csv", "Comparacion de ventanas de training 2 a 6 anos."],
        ["max_weight_decision_train5_resumen.csv", "Comparacion de restricciones con training de 5 anos."],
        ["train5_maxweight_2_5_anual_independiente.csv", "Resultados anuales finales con max_weight = 2,5%."],
        ["train5_maxweight_1_2_5_5_10_no_pandemia.csv", "Corte no pandemia para 1%, 2,5%, 5% y 10%."],
        ["train5_maxweight_1_2_5_5_10_pandemia.csv", "Corte pandemia para 1%, 2,5%, 5% y 10%."],
    ]
    story.append(Spacer(1, 0.25 * cm))
    story.append(section("Archivos fuente", h1, styled_table(sources, [7.0 * cm, 12.0 * cm], "#334155", font_size=8.0, align_from_col=1)))

    doc.build(story, onFirstPage=add_footer, onLaterPages=add_footer)
    print(OUTPUT)


if __name__ == "__main__":
    main()
