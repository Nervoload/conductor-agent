import sys
from google import genai
from google.genai import types

client = genai.Client(api_key="AIzaSyDxagj6X9E5AyJRfwlQ7ib-kL5ph0ttyDg")
# Configure the Google GenAI API with your API key.
class Conductor:
    def __init__(self, base_context, max_agents=2, max_iterations=2):
        self.base_context = base_context      # Conductor's overarching instructions
        self.max_agents = max_agents          # Maximum number of agents allowed
        self.max_iterations = max_iterations  # Maximum re-prompt cycles for each agent
        self.agents = []                      # List to hold spawned agents
        self.problem = ""                     # The user-provided problem and context
        self.plan = ""                        # The generated plan of action
        self.final_outputs = []               # Storage for agent outputs

    def get_problem(self):
        print("\nConductor: Please provide the problem description and context:")
        self.problem = input("User: ")
        return self.problem

    def generate_plan(self):
        # Generate a detailed plan of action using the Google GenAI API.
        prompt = (
            f"System: {self.base_context}\n\n"
            f"User: I provide here the task context:\n\n{self.problem}\n\n"
            "Before you delegate the tasks to agents, you should lay out your plan on how you will solve the problem and delegate to me first, and then we'll continue. "
            "The plan should include:\n"
            "  - How many agents to spawn (maximum of 2).\n"
            "  - The specific task each agent will perform.\n"
            "  - A summarized context for each agent.\n"
            "Present the plan in a paragraph form. Be concise and accurate."
        )
        print("\nConductor: Generating plan...")
        response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[prompt],
                config=types.GenerateContentConfig(
                    max_output_tokens=500,
                    temperature=0.1
                )
        )
        # Assume the first candidate contains our plan.
        self.plan = response.text
        print("\nConductor (Plan):")
        print(self.plan)
        return self.plan

    def get_plan_approval(self):
        print("\nConductor: Do you approve this plan? (yes/no)")
        approval = input("User: ").strip().lower()
        return approval == "yes"

    def spawn_agents(self):
        print("\nConductor: Spawning agents as per the approved plan...")
        for i in range(1, self.max_agents + 1):
            # In a complete version, you might parse self.plan to extract agent tasks.
            task = f"Agent {i} Task: Execute part {i} of the plan based on the problem:\n{self.problem}"
            agent_context = (
                f"You are Agent {i}. You are a highly technical and skilled assistant, and you are being assigned a task from the conductor, who is an AI agent. "
                "Your goal is to complete the assigned task efficiently and accurately. "
                f"Task details: {task}"
            )
            agent = Agent(agent_id=i, task=task, context=agent_context)
            self.agents.append(agent)
            print(f"Conductor: Spawned Agent {i} with task:\n{task}\n")

    def evaluate_agent_output(self, agent_output):
        # Simple evaluation: flag outputs that are too brief or contain markers of incompleteness.
        if "incomplete" in agent_output.lower():
            return False, "The output appears to be incomplete."
        if len(agent_output.split()) < 10:
            return False, "The output is too brief."
        return True, ""

    def run_agents(self):
        for agent in self.agents:
            print(f"\nConductor: Starting conversation with Agent {agent.agent_id}")
            final_response = agent.run_conversation(max_iterations=self.max_iterations)
            self.final_outputs.append((agent.agent_id, final_response))

    def compile_final_output(self):
        compiled = "Compiled Final Output:\n"
        for agent_id, output in self.final_outputs:
            compiled += f"\n--- Agent {agent_id} Output ---\n{output}\n"
        print("\nConductor: Compiled final output:")
        print(compiled)
        with open("final_output.txt", "w") as f:
            f.write(compiled)
        print("\nConductor: Final output saved to 'final_output.txt'.")
        return compiled

    def orchestrate(self):
        # Step 1: Get the problem description and context from the user.
        self.get_problem()

        # Step 2: Generate a plan of action and show it to the user.
        self.generate_plan()

        # Step 3: Ask for user approval; if not approved, allow re-evaluation.
        while not self.get_plan_approval():
            print("\nConductor: Please provide additional guidance or adjustments to the plan:")
            adjustment = input("User: ")
            self.problem += f"\nAdditional guidance: {adjustment}"
            self.generate_plan()

        # Step 4: Spawn agent instances as per the approved plan.
        self.spawn_agents()

        # Step 5 & 6: Send tasks to agents and collect their outputs.
        self.run_agents()

        # Step 7: Compile the agents' outputs into a final output.
        self.compile_final_output()

        # Step 8: End the process.
        print("\nConductor: Task complete. Exiting.")


class Agent:
    def __init__(self, agent_id, task, context):
        self.agent_id = agent_id
        self.task = task       # The initial task prompt.
        self.context = context # The agent's system instructions.
        # Create a chat conversation without passing system_instruction; then send system instructions as first message.
        self.chat = client.chats.create(model="gemini-2.0-flash")
        _ = self.chat.send_message(self.context)

    def run_conversation(self, max_iterations=2):
        current_prompt = self.task
        iteration = 0
        last_response = ""
        while iteration <= max_iterations:
            print(f"\nConductor to Agent {self.agent_id}: {current_prompt}")
            response = self.chat.send_message(current_prompt)
            last_response = response.text
            print(f"Agent {self.agent_id}: {last_response}\n")
            follow_up = input(f"Conductor: Enter follow-up instruction for Agent {self.agent_id} (or press Enter to accept): ")
            if not follow_up.strip():
                break
            current_prompt = follow_up
            iteration += 1
        return last_response


def main():
    # Step 0: Set the conductor's initial context.
    base_context = (
        "You are an AI responsible for managing a multi-agent system. You are the Conductor. You, as the Conductor, will be given a task or problem, and you have to reason out a plan from the assigned task. You will not carry out the task directly, instead you will delegate to agents. "
        "You have the ability to spawn up to 2 agents to solve a given problem. You will give them more context on the problem, and assign agents a task. They can either do the same task, different parts of a task, or multiple subtasks to accomplish a goal. "
        "Your functions include generating a plan, spawning agents, evaluating their outputs, "
        "and compiling the final results. You must ensure that agents are re-prompted if their output is not "
        "technically sound, complete, and accurate. Your dialogue with the user and agents will be visible "
        "in the terminal."
    )
    conductor = Conductor(base_context)
    conductor.orchestrate()


if __name__ == "__main__":
    main()