import unittest
from src.cybersecurity_ai.anomaly_detection.anomaly_detector import AnomalyDetector

class TestAnomalyDetector(unittest.TestCase):
    def setUp(self):
        self.detector = AnomalyDetector()

    def test_detect(self):
        anomalies = self.detector.detect('Test data')
        self.assertEqual(anomalies, ['Anomaly Detected'])

if __name__ == '__main__':
    unittest.main()

