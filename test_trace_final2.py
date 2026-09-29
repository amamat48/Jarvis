import sys
sys.path.insert(0, r'C:\Users\amana\Documents\Jarvis')
import os
os.chdir(r'C:\Users\amana\Documents\Jarvis\jarvis_test_project')
print('Working directory:', os.getcwd())

sys.path.insert(0, r'C:\Users\amana\Documents\Jarvis')
from tasks.manager import TaskManager
from brain.orchestrator import execute_task_turn
from brain.prompt import SYSTEM_PROMPT
from tools.security import SecurityGate

manager = TaskManager(executor=execute_task_turn, system_prompt=SYSTEM_PROMPT, security_gate=SecurityGate(), max_concurrent_tasks=1)

task_spec = '''
I need you to fix a bug in a calculator project. The project is in the current working directory (jarvis_test_project).

The project has:
- calculator.py - contains a bug in the divide function (uses integer division instead of float division)
- cli.py - command-line interface
- test_calculator.py - unit tests that reveal the bug

Your task:
1. First, list the files in the jarvis_test_project directory to confirm the project structure
2. Read calculator.py to understand the bug
3. Run the tests to see the failure
5. Identify and fix the bug in calculator.py (the divide function uses integer division // instead of float division /)
5. Run the tests again to verify the fix
6. Test the CLI to make sure it works

Start by listing the files in the jarvis_test_project directory.
'''

from tasks.manager import TaskManager
from brain.orchestrator import execute_task_turn
from brain.prompt import SYSTEM_PROMPT
from tools.security import SecurityGate

manager = TaskManager(executor=execute_task_turn, system_prompt=SYSTEM_PROMPT, security_gate=SecurityGate(), max_concurrent_tasks=1)

result = manager.submit_message(task_spec)
print('Task created:', result.task_id)

import time
for i in range(30):
    time.sleep(5)
    snapshot = manager.get_snapshot()
    if snapshot.tasks:
        task = snapshot.tasks[0]
        print('Time:', i*5, 's - Task:', task.task_id[:8], 'State:', task.state, 'Activity:', task.current_activity)
        if task.pending_approval:
            print('  Approval ID:', task.pending_approval.approval_id)
            approval_result = manager.approve_action(task.task_id, task.pending_approval.approval_id, expected_revision=task.revision)
            print('  Approval result:', approval_result.accepted)
        if task.state.name in ['COMPLETED', 'FAILED']:
            print('  Result:', task.result)
            print('  Error:', task.error)
            ctx = manager._contexts.get(task.task_id)
            if ctx:
                print('  Messages:')
                for msg in ctx.messages[-5:]:
                    role = msg.get('role')
                    content = msg.get('content', '')
                    tool_calls = msg.get('tool_calls')
                    print(f'    {role}: {str(content)[:200]}')
                    if tool_calls:
                        print(f'    Tool calls: {tool_calls}')
            break
    else:
        print('Time:', i*5, 's - No tasks found')

manager.close(wait=True)