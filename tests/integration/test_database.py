import unittest
import psycopg2
import logging

logger = logging.getLogger('tests.integration.test_database')

class TestDatabase(unittest.TestCase):
    def setUp(self):
        try:
            self.conn = psycopg2.connect(
                dbname="ai_db_dev",
                user="user",
                password="password",
                host="localhost",
                port="5432"
            )
            self.cur = self.conn.cursor()
            logger.info("Database connection established for testing.")
        except Exception as e:
            logger.error(f"Error connecting to database: {type(e).__name__}: {e}")
            raise

    def test_connection(self):
        self.cur.execute("SELECT 1")
        result = self.cur.fetchone()
        self.assertEqual(result[0], 1)
        logger.info("Database connection test passed.")

    def tearDown(self):
        self.cur.close()
        self.conn.close()
        logger.info("Database connection closed after testing.")

if __name__ == '__main__':
    unittest.main()

