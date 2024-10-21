import unittest
import requests
import logging

logger = logging.getLogger('tests.integration.test_recommender_service')

class TestRecommenderService(unittest.TestCase):
    RECOMMENDER_SERVICE_URL = "http://localhost:8001/recommend"

    def test_recommender_service(self):
        payload = {
            "user_id": 1,
            "item_id": 101
        }
        try:
            response = requests.post(self.RECOMMENDER_SERVICE_URL, json=payload)
            self.assertEqual(response.status_code, 200)
            data = response.get_json()
            self.assertIn("recommendations", data)
            self.assertEqual(data["recommendations"], [301, 302, 303])
            logger.info("Recommender Service endpoint test passed.")
        except Exception as e:
            logger.error(f"Recommender Service endpoint test failed: {type(e).__name__}: {e}")
            self.fail(e)

if __name__ == '__main__':
    unittest.main()

