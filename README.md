# MCTS_r-ARC_AGI

## Project Description
MCTS_r-ARC_AGI is an AI project that combines Monte Carlo Tree Search (MCTS) with large language models to tackle ARC-AGI tasks. This project aims to enhance AI's problem-solving capabilities in abstract reasoning scenarios.

## Features
- Integration of MCTS with state-of-the-art language models (GPT-4o and LLaMA 3.1)
- Customizable task selection and model parameters
- Efficient handling of ARC tasks with adjustable rollout and token limits

## Usage

Create a `.env` file in the project root and add your API keys or other necessary configurations.

Run the main script with desired parameters:
```
python main.py --task_id <task_id> --model <model_name> --max_rollouts <num_rollouts> --max_tokens <max_tokens>
```

## License
This project is licensed under the MIT License. See the LICENSE file for more details.
