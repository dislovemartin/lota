import unittest
import requests
import logging

logger = logging.getLogger('tests.integration.test_api')

class TestAPI(unittest.TestCase):
    AI_CHATBOT_API_URL = "http://localhost:8000"
    PREDICTIVE_ANALYTICS_API_URL = "http://localhost:8002"

    def test_ai_chatbot_endpoint(self):
        payload = {
            "message": "Hello"
        }
        try:
            response = requests.post(f"{self.AI_CHATBOT_API_URL}/chat", json=payload)
            self.assertEqual(response.status_code, 200)
            data = response.get_json()
            self.assertIn("response", data)
            self.assertEqual(data["response"], "Echo: Hello")
            logger.info("AI Chatbot endpoint test passed.")
        except Exception as e:
            logger.error(f"AI Chatbot endpoint test failed: {type(e).__name__}: {e}")
            self.fail(e)

    def test_predictive_analytics_endpoint(self):
        payload = {
            "feature1": 1.0,
            "feature2": "A"
        }
        try:
            response = requests.post(f"{self.PREDICTIVE_ANALYTICS_API_URL}/risk_assessment", json=payload)
            self.assertEqual(response.status_code, 200)
            data = response.get_json()
            self.assertIn("risk_score", data)
            self.assertEqual(data["risk_score"], 0.75)
            logger.info("Predictive Analytics endpoint test passed.")
        except Exception as e:
            logger.error(f"Predictive Analytics endpoint test failed: {type(e).__name__}: {e}")
            self.fail(e)

if __name__ == '__main__':
    unittest.main()

