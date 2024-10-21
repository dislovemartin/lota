import unittest
from src.personalization_engine.recommender.hybrid_recommender import HybridRecommender

class TestHybridRecommender(unittest.TestCase):
    def setUp(self):
        self.recommender = HybridRecommender()

    def test_recommend(self):
        recommendations = self.recommender.recommend(user_id=1, item_id=101)
        self.assertEqual(recommendations, [301, 302, 303])

if __name__ == '__main__':
    unittest.main()

