import unittest
from src.ai_chatbot.chatbot import app
from src.predictive_analytics.risk_assessment.catboost_risk_model import RiskAssessmentModel
import logging

logger = logging.getLogger('tests.e2e.test_end_to_end')

class TestEndToEnd(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True
        self.model = RiskAssessmentModel()

    def test_chatbot_and_risk_assessment(self):
        # Test Chatbot
        chatbot_response = self.app.post('/chat', json={'message': 'Hello'})
        self.assertEqual(chatbot_response.status_code, 200)
        data = chatbot_response.get_json()
        self.assertIn('response', data)
        self.assertEqual(data['response'], 'Echo: Hello')
        logger.info("End-to-End Chatbot test passed.")

        # Test Risk Assessment
        risk_result = self.model.predict([[10]])
        self.assertIn('risk_score', risk_result)
        self.assertEqual(risk_result['risk_score'], 0.75)
        logger.info("End-to-End Risk Assessment test passed.")

if __name__ == '__main__':
    unittest.main()

