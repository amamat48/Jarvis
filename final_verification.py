import sys
sys.path.insert(0, r'C:\Users\amana\Documents\Jarvis')
import os
os.chdir(r'C:\Users\amana\Documents\Jarvis\jarvis_test_project')
print('Working directory:', os.getcwd())
print('Files in test directory:', os.listdir('.'))

sys.path.insert(0, r'C:\Users\amana\Documents\Jarvis')

from tasks.manager import TaskManager
from brain.orchestrator import execute_task_turn
from brain.prompt import SYSTEM_PROMPT
from tools.security import SecurityGate

manager = TaskManager(executor=execute_task_turn, system_prompt=SYSTEM_PROMPT, security_gate=SecurityGate(), max_concurrent_tasks=1)

task_spec = '''
I need you to fix a bug in a calculator project. The project is in the jarvis_test_project subdirectory.

The project has:
- calculator.py - contains a bug in the divide function (uses integer division instead of float division)
- cli.py - command-line interface
- test_calculator.py - unit tests that reveal the bug

Your task - YOU MUST COMPLETE ALL STEPS:
1. First, list the files in the jarvis_test_project directory to confirm the project structure
2. Read calculator.py to understand the bug
3. Run the tests (python -m pytest test_calculator.py -v) to see the failure
4. Identify and fix the bug in calculator.py (the divide function uses integer division // instead of float division /)
5. Run the tests again to verify the fix
6. Test the CLI (python cli.py divide 7 2) to make sure it works

DO NOT stop until all steps are complete. You must use tools for each step.
'''

from tasks.manager import TaskManager
from brain.orchestrator import execute_task_turn
from brain.prompt import SYSTEM_PROMPT
from tools.security import SecurityGate

manager = TaskManager(executor=execute_task_turn, system_prompt=SYSTEM_PROMPT, security_gate=SecurityGate(), max_concurrent_tasks=1)

result = manager.submit_message(task_spec)
print('Task created:', result.task_id)

import time
for i in range(60):
    time.sleep(5)
    snapshot = manager.get_snapshot()
    if snapshot.tasks:
        task = snapshot.tasks[0]
        print(f'Time: {i*5}s - Task: {task.task_id[:8]} State: {task.state} Activity: {task.current_activity}')
        if task.pending_approval:
            print('  Approval ID:', task.pending_approval.approval_id)
            approval_result = manager.approve_action(task.task_id, task.pending_approval.approval_id, expected_revision=task.revision)
            print('  Approval result:', approval_result.accepted)
        if task.state.name in ['COMPLETED', 'FAILED']:
            print('  Result:', task.result)
            print('  Error:', task.error)
            break
    else:
        print(f'Time: {i*5}s - No tasks found')

manager.close(wait=True)