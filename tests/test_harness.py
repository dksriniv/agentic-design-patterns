import unittest
import importlib.util
from types import SimpleNamespace
from harness.actions import validate
from harness.agents import LiveAgent, ScriptedAgent, SCENARIOS, action
from harness.runtime import Harness, Session
from harness.tools import FAQSearch


class SequenceAgent:
    def __init__(self, actions):
        self.actions = iter(actions)

    def choose_action(self, session):
        return next(self.actions)


class HarnessTests(unittest.TestCase):
    def test_direct_answer(self):
        session = Harness(ScriptedAgent("direct-answer"), FAQSearch()).run(Session(SCENARIOS["direct-answer"]))
        self.assertEqual(session.status, "answered")
        self.assertEqual(session.attempts, 2)
        self.assertEqual(session.sources, ["password-reset"])

    def test_clarification_pauses_and_resumes_without_reset(self):
        harness = Harness(ScriptedAgent("account-recovery"), FAQSearch())
        session = harness.run(Session(SCENARIOS["account-recovery"]))
        self.assertEqual(session.status, "waiting_for_user")
        self.assertEqual(session.steps_remaining, 3)
        self.assertIn("access-overview", session.retrieved)
        harness.resume(session, "I lost my phone.")
        self.assertEqual(session.status, "answered")
        self.assertEqual(session.attempts, 4)
        self.assertEqual(session.events[2]["result"]["reply"], "I lost my phone.")

    def test_empty_search_recovery(self):
        session = Harness(ScriptedAgent("unsuccessful-search"), FAQSearch()).run(Session("lunar delivery"))
        self.assertEqual(session.status, "insufficient_information")
        self.assertEqual(session.events[0]["result"]["documents"], [])
        self.assertEqual(session.events[1]["result"]["documents"], [])

    def test_invalid_actions_consume_budget(self):
        agent = SequenceAgent([{"action": "delete_account", "arguments": {}}] * 2)
        session = Harness(agent, FAQSearch()).run(Session("help", max_steps=2))
        self.assertEqual(session.status, "limit_reached")
        self.assertEqual(session.attempts, 2)
        self.assertEqual(session.events[0]["result"]["error"], "invalid_action")

    def test_invented_citation_rejected_then_corrected(self):
        agent = SequenceAgent([
            action("finish", answer="Answer", sources=["fake"], outcome="answered"),
            action("search_faq", query="forgotten password"),
            action("finish", answer="Use Forgot Password.", sources=["password-reset"], outcome="answered"),
        ])
        session = Harness(agent, FAQSearch()).run(Session("password"))
        self.assertEqual(session.events[0]["result"]["error"], "invalid_action")
        self.assertEqual(session.status, "answered")

    def test_repeated_search_is_not_executed(self):
        class CountingSearch(FAQSearch):
            calls = 0
            def search(self, query):
                self.calls += 1
                return super().search(query)
        tool = CountingSearch()
        agent = SequenceAgent([action("search_faq", query=q) for q in ["password", " PASSWORD "]])
        session = Harness(agent, tool).run(Session("help", max_steps=2))
        self.assertEqual(tool.calls, 1)
        self.assertEqual(session.events[1]["result"]["error"], "repeated_search")
        self.assertEqual(session.status, "limit_reached")

    def test_last_step_cannot_pause(self):
        agent = SequenceAgent([action("ask_user", question="What issue?")])
        session = Harness(agent, FAQSearch()).run(Session("help", max_steps=1))
        self.assertEqual(session.status, "limit_reached")
        self.assertEqual(session.pending_question, "")

    def test_tool_failure_can_be_observed_and_recovered(self):
        class BrokenSearch:
            def search(self, query):
                raise RuntimeError("internal details")
        agent = SequenceAgent([
            action("search_faq", query="password"),
            action("finish", answer="Search is unavailable.", sources=[], outcome="insufficient_information"),
        ])
        session = Harness(agent, BrokenSearch()).run(Session("help"))
        self.assertEqual(session.events[0]["result"]["error"], "tool_error")
        self.assertEqual(session.status, "insufficient_information")

    def test_agent_error_stops_cleanly(self):
        session = Harness(SequenceAgent([]), FAQSearch()).run(Session("help"))
        self.assertEqual(session.status, "agent_error")
        self.assertEqual(session.attempts, 1)

    def test_answer_requires_evidence(self):
        agent = SequenceAgent([action("finish", answer="Unsupported", sources=[], outcome="answered")])
        session = Harness(agent, FAQSearch()).run(Session("help", max_steps=1))
        self.assertEqual(session.status, "limit_reached")
        self.assertEqual(session.answer, "")

    @unittest.skipUnless(importlib.util.find_spec("langchain_core"), "optional live dependency absent")
    def test_live_adapter_parses_json_and_reports_malformed_output(self):
        class FakeModel:
            content = '{"action":"search_faq","arguments":{"query":"password"}}'
            def invoke(self, messages):
                self.messages = messages
                return SimpleNamespace(content=self.content)
        agent = LiveAgent.__new__(LiveAgent)
        agent.model = FakeModel()
        session = Session("help")
        session.attempts = 1
        self.assertEqual(validate(agent.choose_action(session)).name, "search_faq")
        self.assertIn('"actions_available_including_this_one": 5', agent.model.messages[1].content)
        agent.model.content = "not JSON"
        with self.assertRaises(ValueError):
            validate(agent.choose_action(session))

    def test_malformed_shapes_are_validation_errors(self):
        for raw in [None, [], {"action": [], "arguments": {}}, {"action": "search_faq", "arguments": []}]:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                validate(raw)


if __name__ == "__main__":
    unittest.main()
