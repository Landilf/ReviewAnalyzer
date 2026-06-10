from __future__ import annotations

import html
import json

import pandas as pd

from analysis_helpers.pipeline import SENTIMENT_LABELS_RU


def build_insights(reviews: pd.DataFrame, aspect_stats: pd.DataFrame) -> list[str]:
    if reviews.empty:
        return ["После применения фильтров не осталось отзывов для анализа."]

    insights = []
    sentiment_counts = reviews["predicted_sentiment"].value_counts(normalize=True)
    dominant_sentiment = sentiment_counts.idxmax()
    insights.append(
        f"Преобладающая тональность — {SENTIMENT_LABELS_RU.get(dominant_sentiment, dominant_sentiment).lower()} "
        f"({sentiment_counts.max():.1%} отзывов)."
    )

    negative_reviews = reviews[reviews["predicted_sentiment"] == "negative"]
    if not negative_reviews.empty and not aspect_stats.empty:
        problem_aspects = aspect_stats[aspect_stats["mentions"] >= 2].sort_values(
            ["negative_share", "mentions"],
            ascending=[False, False],
        )
        if not problem_aspects.empty:
            top_problem = problem_aspects.iloc[0]
            insights.append(
                f"Наиболее проблемный аспект — «{top_problem['aspect']}»: "
                f"{top_problem['negative_share']:.1%} негативных упоминаний."
            )

    if "category" in reviews.columns:
        category_sentiment = (
            reviews.assign(is_negative=reviews["predicted_sentiment"].eq("negative"))
            .groupby("category", as_index=False)["is_negative"]
            .mean()
            .sort_values("is_negative", ascending=False)
        )
        if len(category_sentiment) > 1:
            worst_category = category_sentiment.iloc[0]
            insights.append(
                f"Категория с максимальной долей негатива — «{worst_category['category']}» "
                f"({worst_category['is_negative']:.1%})."
            )

    low_confidence_share = reviews["confidence"].lt(0.55).mean()
    if low_confidence_share > 0:
        insights.append(
            f"{low_confidence_share:.1%} отзывов классифицированы с низкой уверенностью; "
            "их стоит проверить вручную."
        )

    return insights


def build_recommendations(aspect_stats: pd.DataFrame) -> list[str]:
    if aspect_stats.empty:
        return ["Добавьте больше отзывов или включите извлечение аспектов для формирования рекомендаций."]

    problem_aspects = aspect_stats[aspect_stats["mentions"] >= 2].sort_values(
        ["negative_share", "mentions"],
        ascending=[False, False],
    )
    recommendations = []
    for _, aspect in problem_aspects.head(3).iterrows():
        recommendations.append(
            f"Проверить аспект «{aspect['aspect']}»: высокая доля негатива "
            f"({aspect['negative_share']:.1%}) при {int(aspect['mentions'])} упоминаниях."
        )
    return recommendations or ["Критичных аспектов с достаточным числом упоминаний не обнаружено."]


def build_html_report(
    reviews: pd.DataFrame,
    aspect_stats: pd.DataFrame,
    topics: pd.DataFrame,
    insights: list[str],
    recommendations: list[str],
    *,
    model_name: str,
    model_metrics: dict[str, float] | None = None,
    confusion: pd.DataFrame | None = None,
    source_name: str | None = None,
    filters: dict | None = None,
    options: dict | None = None,
) -> str:
    sentiment_table = reviews["predicted_sentiment"].value_counts().rename_axis("Тональность").reset_index(name="Отзывов")
    sentiment_table["Тональность"] = sentiment_table["Тональность"].map(SENTIMENT_LABELS_RU)

    avg_conf = reviews["confidence"].mean() if not reviews.empty else 0
    total_reviews = len(reviews)
    negative_share = reviews["predicted_sentiment"].eq("negative").mean() if not reviews.empty else 0
    positive_share = reviews["predicted_sentiment"].eq("positive").mean() if not reviews.empty else 0

    experiment_meta = pd.DataFrame(
        [
            {"Параметр": "Источник данных", "Значение": source_name or "не указан"},
            {"Параметр": "Модель", "Значение": model_name},
            {"Параметр": "Количество отзывов", "Значение": total_reviews},
            {"Параметр": "Средняя уверенность", "Значение": f"{avg_conf:.1%}"},
            {"Параметр": "Доля негативных отзывов", "Значение": f"{negative_share:.1%}"},
            {"Параметр": "Доля позитивных отзывов", "Значение": f"{positive_share:.1%}"},
        ]
    )
    if filters:
        for key, value in filters.items():
            if isinstance(value, (list, tuple)):
                rendered_value = ", ".join(map(str, value))
            else:
                rendered_value = str(value)
            experiment_meta.loc[len(experiment_meta)] = {
                "Параметр": f"Фильтр: {key}",
                "Значение": rendered_value,
            }

    metrics_frame = pd.DataFrame(
        [{"Метрика": metric, "Значение": value} for metric, value in (model_metrics or {}).items()]
    )
    if not metrics_frame.empty:
        metrics_frame["Значение"] = metrics_frame["Значение"].map(lambda value: f"{value:.4f}" if isinstance(value, (int, float)) else value)

    confusion_html = ""
    if confusion is not None and not confusion.empty:
        confusion_html = confusion.to_html(index=True, classes="table", escape=True)

    report_payload = {
        "model_name": model_name,
        "source_name": source_name,
        "metrics": {key: (float(value) if isinstance(value, (int, float)) else value) for key, value in (model_metrics or {}).items()},
        "summary": {
            "total_reviews": total_reviews,
            "avg_confidence": round(float(avg_conf), 4) if total_reviews else 0,
            "negative_share": round(float(negative_share), 4) if total_reviews else 0,
            "positive_share": round(float(positive_share), 4) if total_reviews else 0,
        },
        "insights": insights,
        "recommendations": recommendations,
        "filters": filters or {},
        "sentiment_distribution": sentiment_table.to_dict(orient="records"),
        "aspect_stats_top10": aspect_stats.head(10).to_dict(orient="records"),
        "topics": topics.to_dict(orient="records"),
        "confusion_matrix": None if confusion is None or confusion.empty else confusion.reset_index().to_dict(orient="records"),
    }
    report_json = html.escape(json.dumps(report_payload, ensure_ascii=False, indent=2), quote=False)

    return f"""
<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Экспериментальный отчёт ReviewAnalyzer</title>
  <style>
    body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 40px; color: #1e293b; background: #f8fafc; line-height: 1.6; }}
    .container {{ max-width: 1100px; margin: 0 auto; background: white; padding: 40px; border-radius: 12px; box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1); }}
    h1 {{ color: #0f172a; border-bottom: 2px solid #e2e8f0; padding-bottom: 16px; margin-bottom: 18px; font-size: 28px; }}
    .subtitle {{ color: #475569; margin-top: 0; margin-bottom: 28px; }}
    h2 {{ color: #1e293b; margin-top: 34px; border-left: 4px solid #3b82f6; padding-left: 12px; font-size: 20px; }}
    .stats-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 24px; }}
    .stat-card {{ background: #f1f5f9; padding: 16px; border-radius: 8px; text-align: center; min-height: 92px; }}
    .stat-value {{ display: block; font-size: 24px; font-weight: bold; color: #3b82f6; }}
    .stat-label {{ font-size: 12px; text-transform: uppercase; color: #64748b; letter-spacing: 1px; }}
    .meta-note {{ background: #eff6ff; border-left: 4px solid #3b82f6; padding: 14px 16px; border-radius: 0 6px 6px 0; margin: 18px 0 0; }}
    .comparison-box {{ background: #f8fafc; border: 1px solid #e2e8f0; padding: 16px; border-radius: 10px; margin-top: 16px; }}
    table {{ border-collapse: collapse; width: 100%; margin: 16px 0; font-size: 14px; }}
    th {{ background: #f1f5f9; color: #475569; font-weight: 600; text-align: left; padding: 12px; border-bottom: 2px solid #e2e8f0; }}
    td {{ padding: 12px; border-bottom: 1px solid #f1f5f9; }}
    .insight-list, .recommendation-list {{ list-style: none; padding: 0; }}
    .insight-list li {{ background: #eff6ff; border-left: 4px solid #3b82f6; padding: 12px; margin-bottom: 8px; border-radius: 0 4px 4px 0; }}
    .recommendation-list li {{ background: #fff7ed; border-left: 4px solid #f97316; padding: 12px; margin-bottom: 8px; border-radius: 0 4px 4px 0; }}
    .section-grid {{ display: grid; grid-template-columns: 1.2fr 0.8fr; gap: 18px; align-items: start; }}
    .small {{ font-size: 13px; color: #64748b; }}
    pre {{ white-space: pre-wrap; word-break: break-word; background: #0f172a; color: #e2e8f0; padding: 18px; border-radius: 10px; overflow-x: auto; }}
    .footer {{ margin-top: 40px; font-size: 12px; color: #94a3b8; text-align: center; border-top: 1px solid #e2e8f0; padding-top: 20px; }}
  </style>
</head>
<body>
  <div class="container">
    <h1>Экспериментальный HTML-отчёт ReviewAnalyzer</h1>
    <p class="subtitle">Формат отчёта предназначен для сравнения нескольких прогонов модели на одном или сопоставимом наборе данных. Один HTML-файл = один эксперимент.</p>

    <div class="stats-grid">
      <div class="stat-card">
        <span class="stat-value">{total_reviews}</span>
        <span class="stat-label">Отзывов в отчёте</span>
      </div>
      <div class="stat-card">
        <span class="stat-value">{avg_conf:.1%}</span>
        <span class="stat-label">Уверенность модели</span>
      </div>
      <div class="stat-card">
        <span class="stat-value">{len(aspect_stats)}</span>
        <span class="stat-label">Извлечено аспектов</span>
      </div>
      <div class="stat-card">
        <span class="stat-value">{len(topics)}</span>
        <span class="stat-label">Тем в отчёте</span>
      </div>
    </div>

    <div class="comparison-box">
      <strong>Как использовать для сравнения моделей:</strong>
      <div class="small">
        Сравнивайте два HTML-отчёта по блоку метрик, матрице ошибок, доле негативных отзывов, качеству выводов и списку проблемных аспектов.
        Если названия моделей, источник данных и фильтры совпадают, различия в результатах отражают поведение моделей на одинаковом наборе.
      </div>
    </div>

    <h2>Паспорт эксперимента</h2>
    {experiment_meta.to_html(index=False, classes='table', escape=True)}

    <div class="section-grid">
      <div>
        <h2>Метрики модели</h2>
        {metrics_frame.to_html(index=False, classes='table', escape=True) if not metrics_frame.empty else '<p class="small">Метрики недоступны для текущего набора.</p>'}

        <h2>Матрица ошибок</h2>
        {confusion_html if confusion_html else '<p class="small">Матрица ошибок недоступна (нет истинных меток).</p>'}
      </div>
      <div>
        <h2>Автоматические выводы</h2>
        <ul class="insight-list">
          {''.join(f'<li>{html.escape(item)}</li>' for item in insights) if insights else '<li>Выводы не сформированы.</li>'}
        </ul>

        <h2>Ключевые рекомендации</h2>
        <ul class="recommendation-list">
          {''.join(f'<li>{html.escape(item)}</li>' for item in recommendations) if recommendations else '<li>Рекомендации не сформированы.</li>'}
        </ul>
      </div>
    </div>

    <h2>Распределение тональности</h2>
    {sentiment_table.to_html(index=False, classes='table', escape=True)}

    <h2>Анализ проблемных аспектов</h2>
    <p class="small">Топ-10 аспектов с наибольшей долей негативных упоминаний</p>
    {aspect_stats.head(10).to_html(index=False, classes='table', escape=True)}

    <h2>Тематические группы (LDA)</h2>
    {topics.to_html(index=False, classes='table', escape=True)}

    <h2>Сырые данные эксперимента</h2>
    <p class="small">Ниже сохранён JSON-блок с ключевыми метаданными отчёта. Он помогает быстро сравнивать несколько HTML-файлов программно или вручную.</p>
    <pre><code>{report_json}</code></pre>

    <div class="footer">
      Сгенерировано системой ReviewAnalyzer • {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
    </div>
  </div>
</body>
</html>
""".strip()
