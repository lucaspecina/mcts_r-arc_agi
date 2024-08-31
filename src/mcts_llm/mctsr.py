from __future__ import annotations
import re

"""

Implements the MCTS + Self-Refine algorithm from
`Accessing GPT-4 level Mathematical Olympiad Solutions via Monte
Carlo Tree Self-refine with LLaMa-3 8B: A Technical Report`
by Zhang et. al.

The authors' [repo](https://github.com/trotsky1997/MathBlackBox) uses critiques,
refinements, and parent nodes' answers as conversation history.
I haven't tried it yet.

"""

import random
import math
from collections import deque
from enum import Enum
from .llm import openai_chat_completion
from pydantic import BaseModel
import tqdm
from .prompt_configs import llama3_1_8b_prompt_config, gpt_4o_prompt_config, RefineResponse
import numpy as np

ROOT_UCT_SCORE = 10_000


class MCTSNode(BaseModel):
    """
    Represents a node in the Monte Carlo Tree Search (MCTS) algorithm.

    Attributes:
        answer (str): The answer associated with the node.
        parent (MCTSNode | None): The parent node of the current node.
        children (list[MCTSNode]): The list of child nodes.
        visits (int): The number of times the node has been visited.
        Q (float): The Q value of the node.
        reward_samples (list[int]): The list of reward samples for the node.

    Methods:
        add_child(child_node: MCTSNode): Adds a child node to the current node.
        add_reward(reward: int): Adds a reward sample to the node and updates the Q value.
    """

    answer: str
    parent: MCTSNode | None = None
    children: list[MCTSNode] = []
    visits: int = 0
    Q: float = 0
    reward_samples: list[int] = []

    def add_child(self, child_node: MCTSNode):
        """
        Adds a child node to the current node.

        Args:
            child_node (MCTSNode): The child node to be added.
        """
        self.children.append(child_node)

    def __repr__(self):
        return f"MCTSNode(answer={self.answer}, Q={self.Q:.2f}, visits={self.visits})"

    def add_reward(self, reward: int):
        """
        Adds a reward sample to the node and updates the Q value.

        Args:
            reward (int): The reward to be added.
        """
        self.reward_samples.append(reward)
        avg_reward = np.mean(self.reward_samples)
        min_reward = np.min(self.reward_samples)

        # Average worst-case and average outcomes
        self.Q = (min_reward + avg_reward) / 2


class SelectionPolicy(Enum):
    """
    Enumerates the selection policies for choosing a node in the MCTS algorithm.

    Attributes:
        GREEDY (int): Greedy selection policy.
        IMPORTANCE_SAMPLING (int): Importance sampling selection policy.
        PAIRWISE_IMPORTANCE_SAMPLING (int): Pairwise importance sampling selection policy.
    """
    GREEDY = 1
    IMPORTANCE_SAMPLING = 2
    PAIRWISE_IMPORTANCE_SAMPLING = 3


class InitializeStrategy(Enum):
    """
    Enumerates the initialization strategies for the MCTS algorithm.

    Attributes:
        ZERO_SHOT (int): Zero-shot initialization strategy.
        DUMMY_ANSWER (int): Dummy answer initialization strategy.
    """
    ZERO_SHOT = 1
    DUMMY_ANSWER = 2


class MCTSr(BaseModel):
    """
    Implements the MCTS + Self-Refine algorithm.

    Attributes:
        problem (str): The problem to be solved.
        max_rollouts (int): The maximum number of rollouts in the MCTS algorithm.
        exploration_constant (float): The exploration constant for UCT calculation.
        max_children (int): The maximum number of children for each node.
        epsilon (float): A small value to avoid division by zero in UCT calculation.
        reward_limit (int): The reward limit for penalizing excessive rewards.
        excess_reward_penalty (int): The penalty for exceeding the reward limit.
        selection_policy (SelectionPolicy): The selection policy for choosing a node.
        initialize_strategy (InitializeStrategy): The initialization strategy for the MCTS algorithm.
        root (MCTSNode): The root node of the MCTS tree.
        critiques (list[str]): The list of critiques received during self-refinement.
        refinements (list[str]): The list of refined answers during self-refinement.
        rewards (list[float]): The list of rewards obtained during evaluation.
        selected_nodes (list[MCTSNode]): The list of selected nodes during the MCTS algorithm.

    Methods:
        self_refine(node: MCTSNode) -> MCTSNode: Performs self-refinement on a node and returns the refined node.
        _evaluate_answer(node: MCTSNode) -> int: Evaluates the quality of an answer and returns the reward.
        self_evaluate(node: MCTSNode): Evaluates the quality of an answer and updates the node's reward.
        backpropagate(node: MCTSNode): Backpropagates the rewards from a node to its ancestors.
        uct(node: MCTSNode): Calculates the UCT value for a node.
        is_fully_expanded(node: MCTSNode): Checks if a node is fully expanded.
        select_node(): Selects a non-fully expanded node with the highest UCT value.
        zero_shot() -> str: Generates a zero-shot answer.
        initialize(): Initializes the MCTS algorithm.
        run(): Runs the MCTS algorithm.
        get_best_answer(): Returns the best answer found by the MCTS algorithm.
        print(): Prints the MCTS tree.
    """

    problem: str
    max_rollouts: int
    max_tokens: int = 1000
    exploration_constant: float = 1.0
    max_children: int = 2
    epsilon: float = 1e-10
    reward_limit: int = 95
    excess_reward_penalty: int = 5
    selection_policy: SelectionPolicy = SelectionPolicy.IMPORTANCE_SAMPLING
    initialize_strategy: InitializeStrategy = InitializeStrategy.ZERO_SHOT

    root: MCTSNode = MCTSNode(answer="I don't know.")

    # Logs
    critiques: list[str] = []
    refinements: list[str] = []
    rewards: list[float] = []
    selected_nodes: list[MCTSNode] = []

    def self_refine(self, node: MCTSNode) -> MCTSNode:
        """
        Performs self-refinement on a node and returns the refined node.

        Args:
            node (MCTSNode): The node to be refined.

        Returns:
            MCTSNode: The refined node.
        """
        raise NotImplementedError()

    def _evaluate_answer(self, node: MCTSNode) -> int:
        """
        Evaluates the quality of an answer and returns the reward.

        Args:
            node (MCTSNode): The node containing the answer to be evaluated.

        Returns:
            int: The reward obtained for the answer.
        """
        raise NotImplementedError()

    def self_evaluate(self, node: MCTSNode):
        """
        Evaluates the quality of an answer and updates the node's reward.
        Sample `num_samples` times and average the results.

        Args:
            node (MCTSNode): The node to be evaluated.
        """
        reward = self._evaluate_answer(node)

        if reward > self.reward_limit:
            reward -= self.excess_reward_penalty

        node.add_reward(reward)

    def backpropagate(self, node: MCTSNode):
        """
        Backpropagates the rewards from a node to its ancestors.

        Args:
            node (MCTSNode): The node to start the backpropagation from.
        """
        parent = node.parent
        while parent:
            best_child_Q = max(child.Q for child in parent.children)
            parent.Q = (parent.Q + best_child_Q) / 2
            parent.visits += 1
            parent = parent.parent

    def uct(self, node: MCTSNode):
        """
        Calculates the UCT value for a node.

        Args:
            node (MCTSNode): The node to calculate the UCT value for.

        Returns:
            float: The UCT value of the node.
        """
        if not node.parent:
            # Using an arbitrarily high UCT score for the root node.
            # helps to prioritize breadth.
            return ROOT_UCT_SCORE

        return node.Q + self.exploration_constant * math.sqrt(
            math.log(node.parent.visits + 1) / (node.visits + self.epsilon)
        )

    def is_fully_expanded(self, node: MCTSNode):
        """
        Checks if a node is fully expanded.

        A node is fully expanded if either:
        1. It has reached the max number of children
        2. Any of its children have a Q value greater than its own

        Args:
            node (MCTSNode): The node to check.

        Returns:
            bool: True if the node is fully expanded, False otherwise.
        """
        return len(node.children) >= self.max_children or any(
            child.Q > node.Q for child in node.children
        )

    def select_node(self):
        """
        Selects a non-fully expanded node with the highest UCT value.

        A node is fully expanded if either:
        1. It has reached the max number of children
        2. Any of its children have a Q value greater than its own

        Returns:
            MCTSNode: The selected node.
        """
        candidates: list[MCTSNode] = []
        to_consider = deque([self.root])

        while to_consider:
            current_node = to_consider.popleft()
            if not self.is_fully_expanded(current_node):
                candidates.append(current_node)
            to_consider.extend(current_node.children)

        if not candidates:
            return self.root

        if self.selection_policy == SelectionPolicy.GREEDY:
            return max(candidates, key=self.uct)
        elif self.selection_policy == SelectionPolicy.IMPORTANCE_SAMPLING:
            # Sample, weighted by UCT score
            uct_scores = [self.uct(node) for node in candidates]
            selected_pair_idx = random.choices(
                range(len(candidates)), weights=uct_scores, k=1
            )[0]
            return candidates[selected_pair_idx]
        elif self.selection_policy == SelectionPolicy.PAIRWISE_IMPORTANCE_SAMPLING:
            # Sample, weighted by the difference in UCT scores between pairs
            uct_scores = [self.uct(node) for node in candidates]
            pairs = [
                (i, j) for i in range(len(candidates)) for j in range(len(candidates))
            ]
            pair_weights = [
                max(uct_scores[i], uct_scores[j]) - min(uct_scores[i], uct_scores[j])
                for i, j in pairs
            ]
            selected_pair_idx = random.choices(
                range(len(pairs)), weights=pair_weights, k=1
            )[0]
            selected_candidate_idx = max(
                pairs[selected_pair_idx], key=lambda x: uct_scores[x]
            )
            return candidates[selected_candidate_idx]
        else:
            raise ValueError(f"Invalid selection policy: {self.selection_policy}")

    def zero_shot(self) -> str:
        """
        Generates a zero-shot answer.

        Returns:
            str: The zero-shot answer.
        """
        raise NotImplementedError()

    def initialize(self):
        """
        Initializes the MCTS algorithm. Generate a zero-shot answer.
        """
        if self.initialize_strategy == InitializeStrategy.ZERO_SHOT:
            self.root = MCTSNode(answer=self.zero_shot())
        elif self.initialize_strategy == InitializeStrategy.DUMMY_ANSWER:
            self.root = MCTSNode(answer="I don't know.")
        else:
            raise ValueError(f"Invalid initialize strategy: {self.initialize_strategy}")

    def run(self):
        """
        Runs the MCTS algorithm.

        Returns:
            str: The best answer found by the MCTS algorithm.
        """
        self.initialize()

        for i in tqdm.tqdm(range(self.max_rollouts)):
            node = self.select_node()
            self.self_evaluate(node)
            child = self.self_refine(node)
            node.add_child(child)
            self.self_evaluate(child)
            self.backpropagate(child)
            print("="*30 + f"\nRollout {i} complete (Q={child.Q:.2f})\n{child.answer}")

        return self.get_best_answer()

    def get_best_answer(self):
        from collections import deque
        """
        Returns the best answer found by the MCTS algorithm.

        Returns:
            str: The best answer.
        """

        to_visit = deque([self.root])
        best_node = self.root

        while to_visit:
            current_node = to_visit.popleft()
            if current_node.Q > best_node.Q:
                best_node = current_node
            to_visit.extend(current_node.children)

        return best_node.answer

    def print(self):
        """
        Prints the MCTS tree.
        """
        print_tree(self.root)


class MCTSrLlama318B(MCTSr):
    """
    Implements the MCTS + Self-Refine algorithm using the LLaMa-3.1 8B model.

    Methods:
        zero_shot() -> str: Generates a zero-shot answer using the LLaMa-3.1 8B model.
        self_refine(node: MCTSNode) -> MCTSNode: Performs self-refinement using the LLaMa-3.1 8B model.
        _evaluate_answer(node: MCTSNode) -> int: Evaluates the quality of an answer using the LLaMa-3.1 8B model.
    """

    def zero_shot(self) -> str:
        """
        Generates a zero-shot answer using the LLaMa-3.1 8B model.

        Returns:
            str: The zero-shot answer.
        """
        response = openai_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": "The user will provide a problem. Solve the problem. Think step by step.",
                },
                {
                    "role": "user",
                    "content": f"<problem>\n{self.problem}\n</problem>",
                },
            ],
            model=llama3_1_8b_prompt_config.model,
            base_url=llama3_1_8b_prompt_config.base_url,
            max_tokens=self.max_tokens,
        )
        assert response.choices[0].message.content is not None
        return response.choices[0].message.content

    def self_refine(self, node: MCTSNode) -> MCTSNode:
        """
        Performs self-refinement on a node using the LLaMa-3.1 8B model and returns the refined node.

        Args:
            node (MCTSNode): The node to be refined.

        Returns:
            MCTSNode: The refined node.
        """
        critique_response = openai_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": llama3_1_8b_prompt_config.critic_system_prompt,
                },
                {
                    "role": "user",
                    "content": "\n\n".join(
                        [
                            f"<problem>\n{self.problem}\n</problem>",
                            f"<current_answer>\n{node.answer}\n</current_answer>",
                        ]
                    ),
                },
            ],
            model=llama3_1_8b_prompt_config.model,
            base_url=llama3_1_8b_prompt_config.base_url,
            max_tokens=self.max_tokens,
        )
        critique = critique_response.choices[0].message.content
        assert critique is not None
        self.critiques.append(critique)

        refined_answer_response = openai_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": llama3_1_8b_prompt_config.refine_system_prompt,
                },
                {
                    "role": "user",
                    "content": "\n\n".join(
                        [
                            f"<problem>\n{self.problem}\n</problem>",
                            f"<current_answer>\n{node.answer}\n</current_answer>",
                            f"<critique>\n{critique}\n</critique>",
                        ]
                    ),
                },
            ],
            model=llama3_1_8b_prompt_config.model,
            base_url=llama3_1_8b_prompt_config.base_url,
            max_tokens=self.max_tokens,
        )
        refined_answer = refined_answer_response.choices[0].message.content
        assert refined_answer is not None
        self.refinements.append(refined_answer)

        return MCTSNode(answer=refined_answer, parent=node)

    def _evaluate_answer(self, node: MCTSNode) -> int:
        """
        Evaluates the quality of an answer using the LLaMa-3.1 8B model and returns the reward.

        Args:
            node (MCTSNode): The node containing the answer to be evaluated.

        Returns:
            int: The reward obtained for the answer.
        """
        messages = [
            {
                "role": "system",
                "content": llama3_1_8b_prompt_config.evaluate_system_prompt,
            },
            {
                "role": "user",
                "content": "\n\n".join(
                    [
                        f"<problem>\n{self.problem}\n</problem>",
                        f"<answer>\n{node.answer}\n</answer>",
                    ]
                ),
            },
        ]
        for attempt in range(3):
            try:
                response = openai_chat_completion(
                    messages=messages,
                    model=llama3_1_8b_prompt_config.model,
                    base_url=llama3_1_8b_prompt_config.base_url,
                    max_tokens=self.max_tokens,
                )
                assert response.choices[0].message.content is not None
                content = response.choices[0].message.content.strip()

                # Try to parse the entire content as an integer
                try:
                    return int(content)
                except ValueError:
                    # If that fails, try to extract the first integer from the string
                    match = re.search(r'\d+', content)
                    if match:
                        return int(match.group())
                    else:
                        raise ValueError("No integer found in the response")

            except ValueError:
                messages.extend([
                    {"role": "assistant", "content": response.choices[0].message.content},
                    {"role": "user", "content": "Failed to parse reward as an integer. Please provide only an integer value."},
                    ])
                if attempt == 2:
                    raise


class MCTSrGPT4o(MCTSr):
    """
    Implements the MCTS + Self-Refine algorithm using the GPT-4o model.

    Methods:
        zero_shot() -> str: Generates a zero-shot answer using the GPT-4o model.
        self_refine(node: MCTSNode) -> MCTSNode: Performs self-refinement using the GPT-4o model.
        _evaluate_answer(node: MCTSNode) -> int: Evaluates the quality of an answer using the GPT-4o model.
    """

    def zero_shot(self) -> str:
        """
        Generates a zero-shot answer using the GPT-4o model.

        Returns:
            str: The zero-shot answer.
        """
        response = openai_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": "The user will provide a problem. Solve the problem. Think step by step.",
                },
                {
                    "role": "user",
                    "content": f"<problem>\n{self.problem}\n</problem>",
                },
            ],
            model=gpt_4o_prompt_config.model,
            max_tokens=self.max_tokens,
        )
        assert response.choices[0].message.content is not None
        return response.choices[0].message.content


    def self_refine(self, node: MCTSNode) -> MCTSNode:
        """
        Performs self-refinement on a node using the GPT-4o model and returns the refined node.

        Args:
            node (MCTSNode): The node to be refined.

        Returns:
            MCTSNode: The refined node.
        """
        critique_response = openai_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": gpt_4o_prompt_config.critic_system_prompt,
                },
                {
                    "role": "user",
                    "content": "\n\n".join(
                        [
                            f"<problem>\n{self.problem}\n</problem>",
                            f"<current_answer>\n{node.answer}\n</current_answer>",
                        ]
                    ),
                },
            ],
            model=gpt_4o_prompt_config.model,
            max_tokens=self.max_tokens,
        )
        critique = critique_response.choices[0].message.content
        assert critique is not None
        self.critiques.append(critique)

        refined_answer_response = openai_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": gpt_4o_prompt_config.refine_system_prompt,
                },
                {
                    "role": "user",
                    "content": "\n\n".join(
                        [
                            f"<problem>\n{self.problem}\n</problem>",
                            f"<current_answer>\n{node.answer}\n</current_answer>",
                            f"<critique>\n{critique}\n</critique>",
                        ]
                    ),
                },
            ],
            model=gpt_4o_prompt_config.model,
            max_tokens=self.max_tokens,
            response_format={"type": "json_object"},
        )
        refined_answer = RefineResponse.model_validate_json(
            refined_answer_response.choices[0].message.content
        )
        self.refinements.append(refined_answer)

        return MCTSNode(
            answer=f"# Thought {refined_answer.thought}\n\n# Answer\n{refined_answer.answer}",
            parent=node,
        )

    def _evaluate_answer(self, node: MCTSNode) -> int:
        """
        Evaluates the quality of an answer using the GPT-4o model and returns the reward.

        Args:
            node (MCTSNode): The node containing the answer to be evaluated.

        Returns:
            int: The reward obtained for the answer.
        """
        messages = [
            {
                "role": "system",
                "content": gpt_4o_prompt_config.evaluate_system_prompt,
            },
            {
                "role": "user",
                "content": "\n\n".join(
                    [
                        f"<problem>\n{self.problem}\n</problem>",
                        f"<answer>\n{node.answer}\n</answer>",
                    ]
                ),
            },
        ]
        for attempt in range(3):
            try:
                response = openai_chat_completion(
                    messages=messages,
                    model=gpt_4o_prompt_config.model,
                    max_tokens=self.max_tokens,
                )
                assert response.choices[0].message.content is not None
                return int(response.choices[0].message.content)
            except ValueError:
                messages.extend(
                    [
                        {
                            "role": "assistant",
                            "content": response.choices[0].message.content,
                        },
                        {
                            "role": "user",
                            "content": "Failed to parse reward as an integer.",
                        },
                    ]
                )
                if attempt == 2:
                    raise

def print_tree(node: MCTSNode | None, level: int = 0, prefix: str = ""):
    """
    Recursively prints the tree structure starting from the given node.

    Args:
        node (MCTSNode | None): The starting node of the tree.
        level (int): The current level of the tree (used for indentation).
        prefix (str): The prefix to use for the current line.
    """
    if node is None:
        return
    
    print(f"{prefix}{'└── ' if level > 0 else ''}Q: {node.Q:.2f}, visits: {node.visits}")
    
    for i, child in enumerate(node.children):
        is_last = (i == len(node.children) - 1)
        new_prefix = prefix + ('    ' if level == 0 or is_last else '│   ')
        print_tree(child, level + 1, new_prefix)