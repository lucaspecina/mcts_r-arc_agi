from src.mcts_llm.mctsr import MCTSrGPT4o, print_tree, MCTSrLlama318B
from src.dataset_utils import load_tasks_from_file, task_sets, json_task_to_string
import argparse
import os

debug = False

pre_question = """You are an AI expert at pattern recognition and puzzle-solving. Analyze the following input-output pairs to identify the transformation rules:

1. Visualize each input and output as a grid, where numbers represent different colors.
2. Pay close attention to the shape, dimensions, and color patterns in both input and output grids.
3. Identify a consistent transformation rule that applies to all example pairs.
4. The rule should be generalizable to any input of similar structure.

Your task:
1. Identify the pattern(s) that map the input to the output.
2. Formulate a clear, concise TRANSFORMATION RULE as a string.
3. Ensure your rule can be applied to the examples provided to produce the correct corresponding outputs.

"""

post_question = """
Remember:
- Be precise in your observations.
- Consider both spatial and numerical transformations.
- Your rule should work for all provided examples.

NO code in your response. Only a string with ONE **SIMPLE** but non-trivial TRANSFORMATION RULE or interesting pattern or observation. 
One simple and clear fact (rule/pattern) that if applied to the input, the output will follow.
""" # AND ALSO the test output (for the corresponding test input) that you would expect to get if your rule is applied correctly.


def main(model, task_id, max_rollouts, max_tokens, debug=False):

    # Load task
    challenges, solutions = load_tasks_from_file(task_sets['training'])
    task_string = json_task_to_string(challenges, task_id, 0)
    print(f"Task string:\n{task_string}") if debug else None

    question = pre_question + "\n\n" + task_string + "\n\n" + post_question

    if model == 'gpt4o':
        mctsr = MCTSrGPT4o(
            problem=question,
            max_rollouts=max_rollouts,
            max_tokens=max_tokens,
            max_children=20,
        )
    elif model == 'llama3.1':
        mctsr = MCTSrLlama318B(
            problem=question,
            max_rollouts=max_rollouts,
            max_tokens=max_tokens,
            max_children=20,
        )
    else:
        raise ValueError(f"Model {model} not supported")
    
    mctsr.run()
    mctsr.print()
    print("\nFINAL ANSWER:")
    print(mctsr.get_best_answer())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--debug", action="store_true", help="Enable verbose output for debugging", default=False)
    parser.add_argument("-m", "--model", type=str, help="model_name", default="llama3.1")
    parser.add_argument("-t", "--task_id", type=str, help="Task ID", default="0520fde7")
    parser.add_argument("-r", "--max_rollouts", type=int, help="Number of rollouts", default=30)
    parser.add_argument("-k", "--max_tokens", type=int, help="Max tokens", default=500)
    args = parser.parse_args()

    print(f"Model: {args.model}")
    print(f"Task ID: {args.task_id}")
    print(f"max_rollouts: {args.max_rollouts}")
    print(f"max_tokens: {args.max_tokens}")

    main(
        args.model,
        args.task_id, 
        args.max_rollouts, 
        args.max_tokens, 
        args.debug
        )
