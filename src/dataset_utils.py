import json

task_sets = {
    'training': {
        'challenges': 'arc-prize-2024/arc-agi_training_challenges.json',
        'solutions': 'arc-prize-2024/arc-agi_training_solutions.json',
    },
    'evaluation': {
        'challenges': 'arc-prize-2024/arc-agi_evaluation_challenges.json',
        'solutions': 'arc-prize-2024/arc-agi_evaluation_solutions.json',
    }
}

def load_tasks_from_file(task_set):
    with open(task_set['challenges'], "r") as tasks:
        challenges = json.load(tasks)

    with open(task_set['solutions'], "r") as tasks:
        solutions = json.load(tasks)

    return challenges, solutions

def json_task_to_string(challenge_tasks, task_id, test_input_index):
    json_task = challenge_tasks[task_id]
    final_output = "CHALLENGE\n"
    train_tasks = json_task['train']
    test_task = json_task['test']

    final_output += "Training Examples\n"
    for i, task in enumerate(train_tasks):
        final_output += f"Example {i + 1}: Input\n["
        for row in task['input']:
            final_output += f"\n{str(row)},"
        final_output += f"]\n\nExample {i + 1}: Output\n["
        for row in task['output']:
            final_output += f"\n{str(row)},"
        final_output += "]\n\n"

    final_output += "Test\n["
    for row in test_task[test_input_index]['input']:
        final_output += f"\n{str(row)}"
    final_output += "]"
    return final_output