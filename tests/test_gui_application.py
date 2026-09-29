import time
import unittest
from pathlib import Path

from gui.application import JarvisApplication
from tasks.models import TaskState


def wait_for(predicate, timeout=4.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


class GuiApplicationTests(unittest.TestCase):
    def setUp(self):
        self.app = JarvisApplication(max_concurrent_tasks=2)
        self.addCleanup(self.app.close)

    def test_cessna_analysis_and_side_question_use_separate_conversations(self):
        conversation = self.app.manager("demo").default_conversation_id
        analysis = self.app.submit("demo", "Analyze my Cessna simulation.", conversation)
        self.assertTrue(analysis["accepted"])
        self.assertTrue(wait_for(lambda: self.app.manager("demo").get_task(analysis["task_id"]).state == TaskState.RUNNING))
        side = self.app.submit("demo", "What does Cm_alpha mean?", conversation)
        self.assertTrue(side["accepted"])
        self.assertTrue(side["side_task"])
        self.assertNotEqual(analysis["conversation_id"], side["conversation_id"])
        self.assertIn(analysis["task_id"], self.app.manager("demo").get_snapshot().active_task_ids)
        self.assertTrue(wait_for(lambda: self.app.manager("demo").get_task(side["task_id"]).state == TaskState.COMPLETED))
        self.assertTrue(wait_for(lambda: self.app.manager("demo").get_task(analysis["task_id"]).state == TaskState.COMPLETED))

    def test_demo_approval_requires_gate_and_no_provider_result_is_fabricated(self):
        result = self.app.submit("demo", "Show approval flow", self.app.manager("demo").default_conversation_id)
        self.assertTrue(result["accepted"])
        manager = self.app.manager("demo")
        self.assertTrue(wait_for(lambda: manager.get_task(result["task_id"]).state == TaskState.WAITING_FOR_APPROVAL))
        task = manager.get_task(result["task_id"])
        approved = self.app.control("demo", task.task_id, "approve", {"approval_id": task.pending_approval.approval_id})
        self.assertTrue(approved["accepted"])
        self.assertTrue(wait_for(lambda: manager.get_task(task.task_id).state == TaskState.COMPLETED))
        self.assertIn("no web result was retrieved", manager.get_task(task.task_id).result)

    def test_natural_language_control_resolves_task_title_and_ordinal(self):
        manager = self.app.manager("demo")
        conversation = manager.default_conversation_id
        task = self.app.submit("demo", "Analyze my Cessna simulation.", conversation)
        self.assertTrue(wait_for(lambda: manager.get_task(task["task_id"]).state == TaskState.RUNNING))
        paused = self.app.submit("demo", "Pause the simulation.", conversation)
        self.assertTrue(paused["accepted"])
        self.assertTrue(wait_for(lambda: manager.get_task(task["task_id"]).state == TaskState.PAUSED))
        resumed = self.app.submit("demo", "Resume Task 1.", conversation)
        self.assertTrue(resumed["accepted"])
        self.assertTrue(wait_for(lambda: manager.get_task(task["task_id"]).state == TaskState.COMPLETED))

    def test_submit_rejects_non_string_messages(self):
        result = self.app.submit("demo", None)
        self.assertFalse(result["accepted"])
        self.assertEqual(result["message"], "Enter a message first.")

    def test_events_include_safe_tool_activity_and_gui_has_required_controls(self):
        manager = self.app.manager("demo")
        baseline = manager.get_snapshot()
        task = self.app.submit("demo", "Analyze local project", manager.default_conversation_id)
        self.assertTrue(task["accepted"])
        self.assertTrue(wait_for(lambda: manager.get_task(task["task_id"]).state == TaskState.COMPLETED))
        feed = self.app.events("demo", baseline.last_sequence, baseline.manager_epoch)
        tool_names = [event["payload"].get("tool_name") for event in feed["events"] if event["event_type"] == "tool_started"]
        self.assertIn("list_files", tool_names)
        self.assertIn("calculate", tool_names)
        self.assertEqual([event["sequence"] for event in feed["events"]], sorted(event["sequence"] for event in feed["events"]))
        static = Path(__file__).resolve().parents[1] / "gui" / "static"
        html = (static / "index.html").read_text(encoding="utf-8")
        js = (static / "app.js").read_text(encoding="utf-8")
        for feature in ("ACTIVE TASKS", "Pause", "Resume", "Cancel", "FOCUS MODE", "Approval required", "activity-dialog"):
            self.assertIn(feature, html)
        self.assertIn("new Notification", js)
        self.assertIn("show()", js)

    def test_demo_modes_show_failure_waiting_and_natural_task_controls(self):
        manager = self.app.manager("demo")
        conv = manager.default_conversation_id
        failed = self.app.submit("demo", "demo failure", conv)
        self.assertTrue(wait_for(lambda: manager.get_task(failed["task_id"]).state == TaskState.FAILED))
        waiting = self.app.submit("demo", "demo input", self.app.manager("demo").create_conversation())
        self.assertTrue(wait_for(lambda: manager.get_task(waiting["task_id"]).state == TaskState.WAITING_FOR_USER_INPUT))
        running = self.app.submit("demo", "Analyze project", self.app.manager("demo").create_conversation())
        self.assertTrue(wait_for(lambda: manager.get_task(running["task_id"]).state == TaskState.RUNNING))
        control = self.app.submit("demo", f"Cancel {running['task_id'][:8]}.", conv)
        self.assertTrue(control["accepted"])
        self.assertTrue(wait_for(lambda: manager.get_task(running["task_id"]).state == TaskState.CANCELLED))


if __name__ == "__main__":
    unittest.main()
