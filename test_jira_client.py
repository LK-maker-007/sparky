# test_jira_client.py
import unittest
from unittest.mock import patch

from jira_client import JiraClient


class TestJiraClient(unittest.TestCase):
    @patch("jira_client.requests.post")
    def test_create_issue_success(self, mock_post):
        mock_post.return_value.status_code = 201
        mock_post.return_value.json.return_value = {"key": "SPARKY-123"}
        client = JiraClient()
        fields = {
            "project": {"key": "SPARKY"},
            "summary": "Test",
            "description": "Test",
            "issuetype": {"name": "Task"},
        }
        result = client.create_issue(fields)
        self.assertEqual(result["key"], "SPARKY-123")

    @patch("jira_client.requests.post")
    def test_create_issue_failure(self, mock_post):
        mock_post.return_value.status_code = 400
        mock_post.return_value.text = "Bad Request"
        client = JiraClient()
        fields = {
            "project": {"key": "SPARKY"},
            "summary": "Test",
            "description": "Test",
            "issuetype": {"name": "Task"},
        }
        result = client.create_issue(fields)
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
