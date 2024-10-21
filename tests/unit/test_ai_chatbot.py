import unittest
from src.ai_chatbot.chatbot import app

class TestChatbot(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_chat_response(self):
        response = self.app.post('/chat', json={'message': 'Hello'})
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn('response', data)
        self.assertEqual(data['response'], 'Echo: Hello')

if __name__ == '__main__':
    unittest.main()

