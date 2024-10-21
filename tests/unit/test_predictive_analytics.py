import unittest
from src.predictive_analytics.risk_assessment.catboost_risk_model import RiskAssessmentModel

class TestRiskAssessmentModel(unittest.TestCase):
    def setUp(self):
        self.model = RiskAssessmentModel()

    def test_predict(self):
        predictions = self.model.predict([[10]])
        self.assertEqual(predictions, [0.75])

if __name__ == '__main__':
    unittest.main()

