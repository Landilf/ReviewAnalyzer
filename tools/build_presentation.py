from __future__ import annotations

from copy import deepcopy
import io
from pathlib import Path
import zipfile
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "Пример презентации.pptx"
OUTPUT = ROOT / "Презентация_ВКР_Рюмин.pptx"

P_NS = "http://schemas.openxmlformats.org/presentationml/2006/main"
A_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PKG_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
TEXT_INSET_X = "110000"
TEXT_INSET_TOP = "70000"
TEXT_INSET_BOTTOM = "70000"

ET.register_namespace("a", A_NS)
ET.register_namespace("p", P_NS)
ET.register_namespace("r", R_NS)


def qn(ns: str, tag: str) -> str:
    return f"{{{ns}}}{tag}"


def find_shape_by_name(root: ET.Element, name: str) -> ET.Element:
    for shape in root.findall(f".//{qn(P_NS, 'sp')}"):
        c_nv_pr = shape.find(f"./{qn(P_NS, 'nvSpPr')}/{qn(P_NS, 'cNvPr')}")
        if c_nv_pr is not None and c_nv_pr.attrib.get("name") == name:
            return shape
    raise ValueError(f"Shape '{name}' not found")


def _max_shape_id(sp_tree: ET.Element) -> int:
    max_id = 1
    for elem in sp_tree.findall(f".//{qn(P_NS, 'cNvPr')}"):
        try:
            max_id = max(max_id, int(elem.attrib.get("id", "1")))
        except ValueError:
            continue
    return max_id


def _pick_template_paragraph(tx_body: ET.Element, preferred_index: int | None = None) -> ET.Element:
    paragraphs = tx_body.findall(qn(A_NS, "p"))
    if preferred_index is not None and preferred_index < len(paragraphs):
        return paragraphs[preferred_index]
    for paragraph in paragraphs:
        if paragraph.findall(qn(A_NS, "r")):
            return paragraph
    return paragraphs[0]


def normalize_text_margins(shape: ET.Element) -> None:
    tx_body = shape.find(qn(P_NS, "txBody"))
    if tx_body is None:
        return
    body_pr = tx_body.find(qn(A_NS, "bodyPr"))
    if body_pr is None:
        body_pr = ET.Element(qn(A_NS, "bodyPr"))
        tx_body.insert(0, body_pr)
    body_pr.attrib["lIns"] = TEXT_INSET_X
    body_pr.attrib["rIns"] = TEXT_INSET_X
    body_pr.attrib["tIns"] = TEXT_INSET_TOP
    body_pr.attrib["bIns"] = TEXT_INSET_BOTTOM


def set_shape_paragraphs(shape: ET.Element, texts: list[str], preferred_index: int | None = None) -> None:
    tx_body = shape.find(qn(P_NS, "txBody"))
    if tx_body is None:
        raise ValueError("Shape has no text body")
    normalize_text_margins(shape)

    template_paragraph = _pick_template_paragraph(tx_body, preferred_index)
    template_p_pr = template_paragraph.find(qn(A_NS, "pPr"))
    template_r_pr = template_paragraph.find(f"./{qn(A_NS, 'r')}/{qn(A_NS, 'rPr')}")
    template_end = template_paragraph.find(qn(A_NS, "endParaRPr"))

    for paragraph in list(tx_body.findall(qn(A_NS, "p"))):
        tx_body.remove(paragraph)

    for text in texts:
        paragraph = ET.Element(qn(A_NS, "p"))
        if template_p_pr is not None:
            paragraph.append(deepcopy(template_p_pr))
        run = ET.SubElement(paragraph, qn(A_NS, "r"))
        if template_r_pr is not None:
            run.append(deepcopy(template_r_pr))
        ET.SubElement(run, qn(A_NS, "t")).text = text
        if template_end is not None:
            paragraph.append(deepcopy(template_end))
        tx_body.append(paragraph)


def build_title_slide(xml_bytes: bytes, number_shape: ET.Element) -> bytes:
    root = ET.fromstring(xml_bytes)

    set_shape_paragraphs(
        find_shape_by_name(root, "Заголовок 1"),
        ["Разработка программного обеспечения для анализа и визуализации результатов обработки пользовательских отзывов"],
    )
    set_shape_paragraphs(
        find_shape_by_name(root, "Подзаголовок 2"),
        ["Министерство науки и высшего образования Российской Федерации ФГБОУ ВО «Рязанский государственный радиотехнический университет имени В. Ф. Уткина»"],
    )

    subtitle_shapes = [
        shape for shape in root.findall(f".//{qn(P_NS, 'sp')}")
        if (shape.find(f"./{qn(P_NS, 'nvSpPr')}/{qn(P_NS, 'cNvPr')}") is not None
            and shape.find(f"./{qn(P_NS, 'nvSpPr')}/{qn(P_NS, 'cNvPr')}").attrib.get("name") == "Подзаголовок 2")
    ]

    # The first subtitle was already replaced with the university line.
    # The remaining ones are the student/supervisor block, program block, type and city.
    set_shape_paragraphs(
        subtitle_shapes[1],
        [
            "Студент: Рюмин В.Р.",
            "Руководитель:",
            "к.т.н., доцент кафедры ВПМ Князьков П.А.",
        ],
    )
    set_shape_paragraphs(
        subtitle_shapes[2],
        [
            "Направление подготовки:",
            "09.03.04 Программная инженерия",
            "ОПОП: Программная инженерия",
            "Группа: 2413",
        ],
    )
    set_shape_paragraphs(subtitle_shapes[3], ["Выпускная квалификационная работа"])
    set_shape_paragraphs(subtitle_shapes[4], ["Рязань 2026"])

    sp_tree = root.find(f".//{qn(P_NS, 'spTree')}")
    if sp_tree is not None:
        title_number_shape = deepcopy(number_shape)
        c_nv_pr = title_number_shape.find(f"./{qn(P_NS, 'nvSpPr')}/{qn(P_NS, 'cNvPr')}")
        if c_nv_pr is not None:
            c_nv_pr.attrib["id"] = "20"
            c_nv_pr.attrib["name"] = "Номер слайда 1"
        set_shape_paragraphs(title_number_shape, ["1"])
        sp_tree.append(title_number_shape)

    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def build_content_slide(xml_bytes: bytes, title: str, lines: list[str], slide_number: int) -> bytes:
    root = ET.fromstring(xml_bytes)
    title_shape = find_shape_by_name(root, "Заголовок 1")
    body_shape = find_shape_by_name(root, "Объект 2")
    number_shape = find_shape_by_name(root, "Номер слайда 3")

    set_shape_paragraphs(title_shape, [title])
    set_shape_paragraphs(body_shape, lines, preferred_index=2)
    set_shape_paragraphs(number_shape, [str(slide_number)])

    sp_tree = root.find(f".//{qn(P_NS, 'spTree')}")
    if sp_tree is not None:
        for pic in list(sp_tree.findall(qn(P_NS, "pic"))):
            sp_tree.remove(pic)

    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def add_placeholder_block(
    xml_bytes: bytes,
    *,
    placeholder_text: str,
    caption_text: str,
    rect_x: int = 8200000,
    rect_y: int = 1350000,
    rect_cx: int = 3200000,
    rect_cy: int = 3200000,
    caption_gap: int = 80000,
    caption_cy: int = 320000,
) -> bytes:
    root = ET.fromstring(xml_bytes)
    sp_tree = root.find(f".//{qn(P_NS, 'spTree')}")
    if sp_tree is None:
        return xml_bytes

    next_id = _max_shape_id(sp_tree) + 1
    rect_shape = make_rect_shape(
        shape_id=next_id,
        name=f"Placeholder {next_id}",
        x=rect_x,
        y=rect_y,
        cx=rect_cx,
        cy=rect_cy,
        lines=[placeholder_text],
    )
    sp_tree.append(rect_shape)

    next_id += 1
    caption_shape = make_caption_shape(
        shape_id=next_id,
        name=f"Caption {next_id}",
        x=rect_x,
        y=rect_y + rect_cy + caption_gap,
        cx=rect_cx,
        cy=caption_cy,
        text=caption_text,
    )
    sp_tree.append(caption_shape)

    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def make_rect_shape(
    *,
    shape_id: int,
    name: str,
    x: int,
    y: int,
    cx: int,
    cy: int,
    lines: list[str],
) -> ET.Element:
    sp = ET.Element(qn(P_NS, "sp"))
    nv = ET.SubElement(sp, qn(P_NS, "nvSpPr"))
    ET.SubElement(nv, qn(P_NS, "cNvPr"), {"id": str(shape_id), "name": name})
    ET.SubElement(nv, qn(P_NS, "cNvSpPr"))
    ET.SubElement(nv, qn(P_NS, "nvPr"))

    sp_pr = ET.SubElement(sp, qn(P_NS, "spPr"))
    xfrm = ET.SubElement(sp_pr, qn(A_NS, "xfrm"))
    ET.SubElement(xfrm, qn(A_NS, "off"), {"x": str(x), "y": str(y)})
    ET.SubElement(xfrm, qn(A_NS, "ext"), {"cx": str(cx), "cy": str(cy)})
    ET.SubElement(sp_pr, qn(A_NS, "prstGeom"), {"prst": "rect"}).append(ET.Element(qn(A_NS, "avLst")))
    ET.SubElement(sp_pr, qn(A_NS, "solidFill")).append(ET.Element(qn(A_NS, "srgbClr"), {"val": "FFFFFF"}))
    ln = ET.SubElement(sp_pr, qn(A_NS, "ln"), {"w": "19050"})
    ET.SubElement(ln, qn(A_NS, "solidFill")).append(ET.Element(qn(A_NS, "srgbClr"), {"val": "808080"}))

    tx_body = ET.SubElement(sp, qn(P_NS, "txBody"))
    ET.SubElement(
        tx_body,
        qn(A_NS, "bodyPr"),
        {
            "anchor": "ctr",
            "rtlCol": "0",
            "lIns": TEXT_INSET_X,
            "rIns": TEXT_INSET_X,
            "tIns": TEXT_INSET_TOP,
            "bIns": TEXT_INSET_BOTTOM,
        },
    )
    ET.SubElement(tx_body, qn(A_NS, "lstStyle"))
    for text in lines:
        p = ET.SubElement(tx_body, qn(A_NS, "p"))
        ET.SubElement(p, qn(A_NS, "pPr"), {"algn": "ctr"})
        r = ET.SubElement(p, qn(A_NS, "r"))
        ET.SubElement(r, qn(A_NS, "rPr"), {"lang": "ru-RU", "sz": "1800", "b": "0"})
        ET.SubElement(r, qn(A_NS, "t")).text = text
        ET.SubElement(p, qn(A_NS, "endParaRPr"), {"lang": "ru-RU", "sz": "1800"})
    return sp


def make_caption_shape(
    *,
    shape_id: int,
    name: str,
    x: int,
    y: int,
    cx: int,
    cy: int,
    text: str,
) -> ET.Element:
    sp = ET.Element(qn(P_NS, "sp"))
    nv = ET.SubElement(sp, qn(P_NS, "nvSpPr"))
    ET.SubElement(nv, qn(P_NS, "cNvPr"), {"id": str(shape_id), "name": name})
    ET.SubElement(nv, qn(P_NS, "cNvSpPr"), {"txBox": "1"})
    ET.SubElement(nv, qn(P_NS, "nvPr"))

    sp_pr = ET.SubElement(sp, qn(P_NS, "spPr"))
    xfrm = ET.SubElement(sp_pr, qn(A_NS, "xfrm"))
    ET.SubElement(xfrm, qn(A_NS, "off"), {"x": str(x), "y": str(y)})
    ET.SubElement(xfrm, qn(A_NS, "ext"), {"cx": str(cx), "cy": str(cy)})
    ET.SubElement(sp_pr, qn(A_NS, "prstGeom"), {"prst": "rect"}).append(ET.Element(qn(A_NS, "avLst")))
    ET.SubElement(sp_pr, qn(A_NS, "noFill"))
    ET.SubElement(sp_pr, qn(A_NS, "ln")).append(ET.Element(qn(A_NS, "noFill")))

    tx_body = ET.SubElement(sp, qn(P_NS, "txBody"))
    ET.SubElement(
        tx_body,
        qn(A_NS, "bodyPr"),
        {
            "anchor": "ctr",
            "rtlCol": "0",
            "lIns": TEXT_INSET_X,
            "rIns": TEXT_INSET_X,
            "tIns": TEXT_INSET_TOP,
            "bIns": TEXT_INSET_BOTTOM,
        },
    )
    ET.SubElement(tx_body, qn(A_NS, "lstStyle"))
    p = ET.SubElement(tx_body, qn(A_NS, "p"))
    ET.SubElement(p, qn(A_NS, "pPr"), {"algn": "ctr"})
    r = ET.SubElement(p, qn(A_NS, "r"))
    ET.SubElement(r, qn(A_NS, "rPr"), {"lang": "ru-RU", "sz": "1400", "i": "1"})
    ET.SubElement(r, qn(A_NS, "t")).text = text
    ET.SubElement(p, qn(A_NS, "endParaRPr"), {"lang": "ru-RU", "sz": "1400", "i": "1"})
    return sp


def trim_presentation_to_12_slides(xml_bytes: bytes) -> bytes:
    root = ET.fromstring(xml_bytes)
    sld_id_lst = root.find(qn(P_NS, "sldIdLst"))
    if sld_id_lst is None:
        return xml_bytes
    for child in list(sld_id_lst)[12:]:
        sld_id_lst.remove(child)
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


SLIDES: list[tuple[str, list[str]]] = [
    (
        "Актуальность",
        [
            "Объём пользовательских отзывов в сети постоянно растёт, а ручная обработка таких данных требует значительных временных затрат.",
            "Отзывы содержат сведения о качестве товаров и услуг, проблемных аспектах и уровне удовлетворённости потребителей.",
            "Для бизнеса и конечных покупателей востребован инструмент, который автоматически собирает, анализирует и наглядно представляет результаты обработки отзывов.",
            "Разработка ReviewAnalyzer направлена на автоматизацию этого процесса для русскоязычных текстовых данных.",
        ],
    ),
    (
        "Цель и задачи работы",
        [
            "Цель работы — разработать программное обеспечение для анализа и визуализации пользовательских отзывов с использованием методов обработки естественного языка.",
            "• проанализировать предметную область и существующие методы обработки отзывов;",
            "• обосновать выбор технологий и аналитических алгоритмов;",
            "• реализовать импорт отзывов из файла, по ссылке и вручную;",
            "• реализовать тональный, аспектный, тематический и кластерный анализ;",
            "• реализовать визуализацию, экспорт результатов и проверить работу системы.",
        ],
    ),
    (
        "Обзор существующих аналогов",
        [
            "В работе рассмотрены Google Cloud Natural Language API, AWS Comprehend, Microsoft Azure Text Analytics, IBM Watson NLU, MeaningCloud, MonkeyLearn и Yandex DataLens.",
            "Ключевые ограничения аналогов: зависимость от внешних сервисов, закрытая архитектура, платная модель доступа и ограниченная адаптация к пользовательскому сценарию анализа отзывов.",
            "Часть решений ориентирована только на NLP-анализ, а часть — только на визуализацию без встроенного импорта и интерпретации результатов.",
            "Это подтверждает целесообразность разработки собственного прикладного программного средства.",
        ],
    ),
    (
        "Практическая значимость и пользователи",
        [
            "Система может использоваться в задачах бизнес-аналитики, мониторинга качества продукции, маркетинговых исследований и оценки пользовательского опыта.",
            "Предполагаемые пользователи: аналитики, специалисты маркетинга, менеджеры по качеству, владельцы товаров и конечные покупатели.",
            "При локальном сценарии система рассчитана на одного оператора; при серверном развёртывании количество пользователей определяется ресурсами хоста.",
            "Практический результат — готовый инструмент, позволяющий импортировать отзывы, проводить анализ и сохранять результаты в удобных форматах.",
        ],
    ),
    (
        "Средства разработки",
        [
            "Основной язык разработки — Python 3.12; среда разработки — PyCharm.",
            "Интерфейс и визуализация: Streamlit и Plotly.",
            "Анализ тональности: Transformers и Torch, модель cointegrated/rubert-tiny-sentiment-balanced (RuBERT tiny).",
            "Подготовка и анализ данных: pandas, scikit-learn, openpyxl.",
            "Импорт веб-данных: requests, beautifulsoup4, Playwright.",
        ],
    ),
    (
        "Функциональные возможности системы",
        [
            "ReviewAnalyzer поддерживает три источника входных данных: ссылка на веб-источник, CSV/XLSX-файл и ручной ввод текста.",
            "После загрузки выполняются нормализация структуры данных, анализ тональности, извлечение аспектов, тематическое моделирование и кластеризация.",
            "Пользователю доступны краткий обзор, расширенный анализ по аспектам, темам, кластерам и периодам, а также таблица отзывов.",
            "Результаты можно сохранить в форматах HTML, CSV и JSON.",
        ],
    ),
    (
        "Алгоритмический пайплайн",
        [
            "1) получение отзывов из файла, по ссылке или вручную;",
            "2) очистка, нормализация и формирование DataFrame;",
            "3) анализ тональности с помощью RuBERT tiny и вычисление confidence;",
            "4) извлечение аспектов и построение аспектной статистики;",
            "5) тематическое моделирование LDA и кластеризация TF-IDF + K-Means;",
            "6) фильтрация, визуализация и экспорт результатов.",
        ],
    ),
    (
        "Методы анализа текстов",
        [
            "Для анализа тональности используется трансформерная модель RuBERT tiny, обеспечивающая приемлемый баланс качества и скорости.",
            "Аспекты извлекаются эвристическим способом на основе токенизации, нормализации, фильтра по части речи и корпусной частоты.",
            "Для тематического моделирования применяется LDA, позволяющая выделять скрытые темы в массиве отзывов.",
            "Для группировки похожих отзывов используется связка TF-IDF и K-Means.",
            "При наличии label или rating дополнительно рассчитываются accuracy, precision, recall, F1 и матрица ошибок.",
        ],
    ),
    (
        "Структура программного обеспечения",
        [
            "Архитектура системы построена модульно и включает интерфейсный слой, сервисы импорта, аналитический контур и вспомогательные подсистемы.",
            "Пакет review_parser отвечает за получение отзывов из файлов, ручного ввода и веб-источников.",
            "Пакет analysis_helpers реализует нормализацию данных, анализ тональности, аспекты, темы, кластеры и отчёты.",
            "Пакет ui.dashboard формирует пользовательский интерфейс, фильтры, исследовательские вкладки и таблицы.",
            "Подсистемы app_logger и app_control обеспечивают логирование и отмену длительных операций.",
        ],
    ),
    (
        "Тестирование программного обеспечения",
        [
            "Для проекта реализован набор автоматизированных тестов, покрывающих ключевые пользовательские и технические сценарии.",
            "Проверяются импорт отзывов, ручной ввод, корректность подготовки данных, генерация HTML-отчёта, логирование и отмена операций.",
            "Отдельно тестируется специализированный парсер Otzovik и обработка граничных случаев.",
            "По результатам разработки и отладки тестовый набор составляет 27 тестов, подтверждающих работоспособность основных компонентов системы.",
        ],
    ),
    (
        "Результаты работы и направления развития",
        [
            "Разработано программное обеспечение ReviewAnalyzer для анализа и визуализации пользовательских отзывов.",
            "Система поддерживает импорт отзывов, автоматический анализ тональности, аспектов, тем и кластеров, а также интерактивную визуализацию и экспорт результатов.",
            "Практическая ценность решения состоит в ускорении анализа пользовательских мнений и повышении наглядности итоговых выводов.",
            "Дальнейшее развитие: расширение набора источников данных, улучшение аспектного анализа, углубление отчётности и развитие серверного режима использования.",
        ],
    ),
]


def build_presentation() -> None:
    files = {}
    with zipfile.ZipFile(TEMPLATE, "r") as zin:
        for name in zin.namelist():
            files[name] = zin.read(name)

    slide4_root = ET.fromstring(files["ppt/slides/slide4.xml"])
    number_shape_template = deepcopy(find_shape_by_name(slide4_root, "Номер слайда 3"))
    slide4_rels = files["ppt/slides/_rels/slide4.xml.rels"]

    files["ppt/slides/slide1.xml"] = build_title_slide(files["ppt/slides/slide1.xml"], number_shape_template)

    for index, (title, lines) in enumerate(SLIDES, start=2):
        slide_name = f"ppt/slides/slide{index}.xml"
        files[slide_name] = build_content_slide(files["ppt/slides/slide4.xml"], title, lines, index)
        files[f"ppt/slides/_rels/slide{index}.xml.rels"] = slide4_rels

    files["ppt/slides/slide8.xml"] = add_placeholder_block(
        files["ppt/slides/slide8.xml"],
        placeholder_text="Место для рисунка\nиз ПЗ",
        caption_text="Рисунок 1 – Алгоритмический конвейер обработки пользовательских отзывов",
    )
    files["ppt/slides/slide9.xml"] = add_placeholder_block(
        files["ppt/slides/slide9.xml"],
        placeholder_text="Место для рисунка\nиз ПЗ",
        caption_text="Рисунок 2 – Структура обработки текста моделью RuBERT",
    )
    files["ppt/slides/slide10.xml"] = add_placeholder_block(
        files["ppt/slides/slide10.xml"],
        placeholder_text="Место для рисунка\nиз ПЗ",
        caption_text="Рисунок 3 – Структура программного обеспечения ReviewAnalyzer",
    )

    files["ppt/presentation.xml"] = trim_presentation_to_12_slides(files["ppt/presentation.xml"])

    with zipfile.ZipFile(OUTPUT, "w", zipfile.ZIP_DEFLATED) as zout:
        for name, data in files.items():
            zout.writestr(name, data)


if __name__ == "__main__":
    build_presentation()
