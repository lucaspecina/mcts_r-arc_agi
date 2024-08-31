from pydantic import BaseModel, Field


class PromptConfig(BaseModel):
    model: str
    critic_system_prompt: str
    refine_system_prompt: str
    evaluate_system_prompt: str
    base_url: str | None = None

class RefineResponse(BaseModel):
    thought: str = Field(..., description="The thought process behind the answer.")
    answer: str = Field(..., description="The answer to the problem.")


critic_system_prompt_base = """
Provide a detailed and constructive critique to improve the answer.
Highlight specific areas that need refinement or correction."""

refine_system_prompt_base = """
# Instruction
Refine the answer based on the critique. Your refined answer should be a direct and concise solution to the problem.

## Additional guidelines
- Your response should not refer to or discuss the criticisms.
- Do not repeat the problem statement.
"""

evaluate_system_prompt_base = """
Provide a reward score between -100 and 100 for the answer quality, using very strict standards.
Do not give a full score above 95. Make sure the reward score is an integer.
Return *ONLY* the score.
"""

llama3_1_8b_prompt_config = PromptConfig(
    base_url="http://localhost:11434/v1",
    model="llama3.1",
    critic_system_prompt=critic_system_prompt_base,
    refine_system_prompt=refine_system_prompt_base + "- Respond with only the answer.",
    evaluate_system_prompt=evaluate_system_prompt_base,
)

gpt_4o_prompt_config = PromptConfig(
    model="gpt-4o",
    critic_system_prompt=critic_system_prompt_base,
    refine_system_prompt=refine_system_prompt_base + """
# JSON Response format
{"thought": "The thought process behind the answer.", "answer": "A string representing the grid answer (test output) for the test case."}
""",
    evaluate_system_prompt=evaluate_system_prompt_base,
)
