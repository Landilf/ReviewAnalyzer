import unittest

from tools.prepare_absa_draft import extract_rows


class PrepareAbsADraftTests(unittest.TestCase):
    @staticmethod
    def _annotations(pros: str, cons: str):
        text = f"Достоинства: {pros}\nНедостатки: {cons}\nОсновной текст отзыва."
        return [(row["aspect"], row["aspect_normalized"], row["aspect_sentiment"]) for row in extract_rows(1, text, "url", "5")]

    def test_does_not_match_substrings_as_design_or_size(self):
        annotations = self._annotations("Видео весьма полезно", "нет")

        self.assertEqual(annotations, [])

    def test_charge_and_charger_do_not_share_one_span(self):
        annotations = self._annotations("Зарядка", "нет")

        self.assertEqual(annotations, [("Зарядка", "зарядка", "positive")])

    def test_does_not_use_action_as_performance_aspect(self):
        annotations = self._annotations("нет", "тормозит и зависает")

        self.assertEqual(annotations, [])

    def test_quality_of_camera_keeps_only_camera_aspect(self):
        annotations = self._annotations("качество камеры", "нет")

        self.assertEqual(annotations, [("камеры", "камера", "positive")])

    def test_charging_connector_is_not_a_sim_slot(self):
        annotations = self._annotations("нет", "Разъём для зарядки — micro USB")

        self.assertEqual(annotations, [("Разъём для зарядки", "зарядка", "negative")])

    def test_headphones_in_box_marks_headphones_not_box(self):
        annotations = self._annotations("нет", "Нет наушников в комплекте")

        self.assertEqual(annotations, [("наушников", "комплектация", "negative")])

    def test_no_drawbacks_phrase_does_not_produce_negative_price(self):
        annotations = self._annotations("Недорогой", "Для этой ценовой категории их нет.")

        self.assertEqual(annotations, [])


if __name__ == "__main__":
    unittest.main()
