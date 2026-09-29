import sys
sys.path.insert(0, r'C:\Users\amana\Documents\Jarvis')
import os
os.chdir(r'C:\Users\amana\Documents\Jarvis\jarvis_test_project')

from tasks.manager import TaskManager
from brain.orchestrator import execute_task_turn
from brain.prompt import SYSTEM_PROMPT
from tools.security import SecurityGate

manager = TaskManager(executor=execute_task_turn, system_prompt=SYSTEM_PROMPT, security_gate=SecurityGate(), max_concurrent_tasks=1)

result = manager.submit_message('List the files in the jarvis_test_project directory')
print('Task created:', result.task_id)

import time
time.sleep(2)
task = manager.get_task(result.task_id)
print('Task state:', task.state)
ctx = manager._contexts.get(result.task_id)
if ctx:
    print('Messages:')
    for i, msg in enumerate(ctx.messages):
        print(f'  {i}: role={msg.get("role")}, content={str(msg.get("content", ""))[:100]}')
        if msg.get('tool_calls'):
            print(f'  tool_calls: {msg.get("tool_calls")}')

time.sleep(10)
manager.close(wait=True)