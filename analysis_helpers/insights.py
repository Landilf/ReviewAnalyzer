from __future__ import annotations

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
) -> str:
    sentiment_table = reviews["predicted_sentiment"].value_counts().rename_axis("Тональность").reset_index(name="Отзывов")
    sentiment_table["Тональность"] = sentiment_table["Тональность"].map(SENTIMENT_LABELS_RU)
    
    # Расчет средней уверенности
    avg_conf = reviews["confidence"].mean() if not reviews.empty else 0

    return f"""
<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <title>Аналитический отчёт ReviewAnalyzer</title>
  <style>
    body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 40px; color: #1e293b; background: #f8fafc; line-height: 1.6; }}
    .container {{ max-width: 900px; margin: 0 auto; background: white; padding: 40px; border-radius: 12px; box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1); }}
    h1 {{ color: #0f172a; border-bottom: 2px solid #e2e8f0; padding-bottom: 16px; margin-bottom: 32px; font-size: 28px; }}
    h2 {{ color: #1e293b; margin-top: 32px; border-left: 4px solid #3b82f6; padding-left: 12px; font-size: 20px; }}
    .stats-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin-bottom: 32px; }}
    .stat-card {{ background: #f1f5f9; padding: 16px; border-radius: 8px; text-align: center; }}
    .stat-value {{ display: block; font-size: 24px; font-weight: bold; color: #3b82f6; }}
    .stat-label {{ font-size: 12px; text-transform: uppercase; color: #64748b; letter-spacing: 1px; }}
    table {{ border-collapse: collapse; width: 100%; margin: 16px 0; font-size: 14px; }}
    th {{ background: #f1f5f9; color: #475569; font-weight: 600; text-align: left; padding: 12px; border-bottom: 2px solid #e2e8f0; }}
    td {{ padding: 12px; border-bottom: 1px solid #f1f5f9; }}
    .insight-list, .recommendation-list {{ list-style: none; padding: 0; }}
    .insight-list li {{ background: #eff6ff; border-left: 4px solid #3b82f6; padding: 12px; margin-bottom: 8px; border-radius: 0 4px 4px 0; }}
    .recommendation-list li {{ background: #fff7ed; border-left: 4px solid #f97316; padding: 12px; margin-bottom: 8px; border-radius: 0 4px 4px 0; }}
    .footer {{ margin-top: 40px; font-size: 12px; color: #94a3b8; text-align: center; border-top: 1px solid #e2e8f0; padding-top: 20px; }}
  </style>
</head>
<body>
  <div class="container">
    <h1>Отчёт по анализу пользовательских отзывов</h1>
    
    <div class="stats-grid">
      <div class="stat-card">
        <span class="stat-value">{len(reviews)}</span>
        <span class="stat-label">Всего отзывов</span>
      </div>
      <div class="stat-card">
        <span class="stat-value">{avg_conf:.1%}</span>
        <span class="stat-label">Уверенность модели</span>
      </div>
      <div class="stat-card">
        <span class="stat-value">{len(aspect_stats)}</span>
        <span class="stat-label">Извлечено аспектов</span>
      </div>
    </div>

    <h2>Автоматические выводы</h2>
    <ul class="insight-list">
      {''.join(f'<li>{item}</li>' for item in insights)}
    </ul>

    <h2>Ключевые рекомендации</h2>
    <ul class="recommendation-list">
      {''.join(f'<li>{item}</li>' for item in recommendations)}
    </ul>

    <h2>Распределение тональности</h2>
    {sentiment_table.to_html(index=False, classes='table')}

    <h2>Анализ проблемных аспектов</h2>
    <p><small>Топ-10 аспектов с наибольшей долей негативных упоминаний</small></p>
    {aspect_stats.head(10).to_html(index=False, classes='table')}

    <h2>Тематические группы (LDA)</h2>
    {topics.to_html(index=False, classes='table')}

    <div class="footer">
      Сгенерировано системой ReviewAnalyzer • {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
    </div>
  </div>
</body>
</html>
""".strip()
