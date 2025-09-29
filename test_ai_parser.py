# test_ai_parser.py
import unittest
from unittest.mock import patch

from ai_parser import parse_search_query, parse_ticket_request


class TestAIParser(unittest.TestCase):
    @patch("ai_parser._rewrite_text", side_effect=lambda text, *_args, **_kwargs: text)
    def test_basic_parse(self, _mock_rewrite):
        msg = "Create a bug ticket for login failure assigned to alice with high priority."
        assignee_map = {"alice": "account-id-alice"}
        parsed = parse_ticket_request(msg, assignee_map=assignee_map)
        self.assertEqual(parsed["issue_type"], "Bug")
        self.assertEqual(parsed["priority"], "High")
        self.assertEqual(parsed["assignee"], "account-id-alice")

    @patch("ai_parser._rewrite_text", side_effect=lambda text, *_args, **_kwargs: text)
    def test_labels_via_hashtag(self, _mock_rewrite):
        parsed = parse_ticket_request("Need improvement #backend #urgent")
        self.assertIn("backend", parsed["labels"])
        self.assertIn("urgent", parsed["labels"])

    @patch("ai_parser._rewrite_text", side_effect=lambda text, *_args, **_kwargs: text)
    def test_labels_via_explicit_list(self, _mock_rewrite):
        parsed = parse_ticket_request("Task labels: Platform, Release Planning")
        self.assertIn("platform", parsed["labels"])
        self.assertIn("release-planning", parsed["labels"])

    @patch("ai_parser._rewrite_text", side_effect=lambda text, *_args, **_kwargs: text)
    def test_empty_message(self, _mock_rewrite):
        with self.assertRaises(ValueError):
            parse_ticket_request("")

    @patch("ai_parser._rewrite_text", side_effect=lambda text, *_args, **_kwargs: text)
    def test_informal_assign_me(self, _mock_rewrite):
        msg = "hey login page error create ticket assign me urgent tag auth"
        assignee_map = {"me": "account-id-self"}
        parsed = parse_ticket_request(msg, assignee_map=assignee_map)
        self.assertEqual(parsed["assignee"], "account-id-self")
        self.assertIn("auth", parsed["labels"])
        self.assertNotIn("assign", parsed["summary"].lower())
        self.assertNotIn("assign", parsed["description"].lower())

    def test_parse_search_query_for_my_open(self):
        query = parse_search_query("show my open bugs")
        self.assertIsNotNone(query)
        assert query is not None
        self.assertIn("assignee = currentUser()", query.jql)
        self.assertIn("statusCategory != Done", query.jql)
        self.assertGreaterEqual(query.limit, 1)

    def test_parse_search_query_issue_key(self):
        query = parse_search_query("find S1-123 please")
        self.assertIsNotNone(query)
        assert query is not None
        self.assertIn("issuekey in (S1-123)", query.jql)
        self.assertNotIn("text ~", query.jql)

    @patch("ai_parser._rewrite_text", side_effect=lambda text, *_args, **_kwargs: text)
    def test_direct_name_assignment(self, _mock_rewrite):
        msg = "pls make bug ticket checkout broken assign singaraj asap"
        assignee_map = {"singa": "account-id-singa", "singaraj": "account-id-singa"}
        parsed = parse_ticket_request(msg, assignee_map=assignee_map)
        self.assertEqual(parsed["assignee"], "account-id-singa")
        self.assertNotIn("singaraj", parsed["summary"].lower())
        self.assertNotIn("singaraj", parsed["description"].lower())

    @patch("ai_parser._rewrite_text", side_effect=lambda text, *_args, **_kwargs: text)
    def test_structured_form_input(self, _mock_rewrite):
        message = """
        Add epic
        S1-30
        1
        Assign login page timeout issue to Singaraj for resolution
        To Do
        Improve work item
        Description
        The login page timeout is causing issues resulting in delayed access for users.
        Please complete the configuration changes outlined in the attached documentation
        and update the production environment.
        Subtasks
        Add subtask
        """
        parsed = parse_ticket_request(message)
        self.assertEqual(
            parsed["summary"],
            "Assign login page timeout issue to Singaraj for resolution",
        )
        self.assertIn("login page timeout is causing issues", parsed["description"].lower())


if __name__ == "__main__":
    unittest.main()
