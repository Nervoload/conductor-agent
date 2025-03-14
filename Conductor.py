import sys
from google import genai
from google.genai import types

# Read API key from file
with open('api.txt', 'r') as f:
    api_key = f.read().strip()

client = genai.Client(api_key=api_key)
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
        self.chat = client.chats.create(model="gemini-2.0-flash")

    def get_problem(self):
        print("\nConductor: Please provide the problem description and context:")
        self.problem = input("User: ")
        return self.problem

    def generate_plan(self):
        # Generate a detailed plan of action using the persistent chat.
        prompt = (
            f"System: {self.base_context}\n\n"
            f"User: I provide here the task context:\n\n{self.problem}\n\n"
            "Before you delegate the tasks to agents, lay out your plan on how you will solve the problem and delegate to me first. "
            "The plan should include:\n"
            "  - How many agents to spawn.\n"
            "  - The specific task each agent will perform.\n"
            "  - A summarized context for each agent.\n"
            "Present the plan in a paragraph form. Be concise and accurate."
        )
        print("\nConductor: Generating plan...")
        response = self.chat.send_message(prompt)
        self.plan = response.text
        print("\nConductor (Plan):")
        print(self.plan)
        return self.plan

    def get_plan_approval(self):
        print("\nConductor: Do you approve this plan? (yes/no)")
        approval = input("User: ").strip().lower()
        return approval == "yes"

    def get_num_agents(self):
        # Use the persistent chat to ask for the number of agents.
        prompt = (
            f"Based on the problem context:\n{self.problem}\n\n"
            f"And the plan:\n{self.plan}\n\n"
            "How many agents do you need to delegate the tasks effectively? Please provide an integer."
        )
        response = self.chat.send_message(prompt)
        try:
            num = int(response.text.strip())
        except ValueError:
            num = 2  # Fallback default.
        print(f"\nConductor: Decided to spawn {num} agent(s).")
        return num

    def get_agent_context(self, agent_number):
        prompt = (
            f"Based on the problem context:\n{self.problem}\n\n"
            f"And the plan:\n{self.plan}\n\n"
            f"Generate a concise context for Agent {agent_number} that clearly summarizes the task they are delegated. "
            "The context should specify what the agent should accomplish without repeating the entire problem."
        )
        response = self.chat.send_message(prompt)
        context = response.text.strip()
        print(f"\nConductor: Generated context for Agent {agent_number}:\n{context}\n")
        return context

    def spawn_agents(self):
        print("\nConductor: Determining agent delegation...")
        num_agents = self.get_num_agents()
        agent_contexts = {}
        for i in range(1, num_agents + 1):
            agent_contexts[i] = self.get_agent_context(i)
        # Store the agent contexts for future reference.
        with open("agent_contexts.txt", "w") as f:
            for key, context in agent_contexts.items():
                f.write(f"Agent {key} Context:\n{context}\n\n")
        # Spawn agents using the generated contexts.
        for i in range(1, num_agents + 1):
            task = f"Agent {i} Task: Execute your assigned delegation as defined in your context.\n{self.problem}"
            agent = Agent(agent_id=i, task=task, context=agent_contexts[i])
            self.agents.append(agent)
            print(f"Conductor: Spawned Agent {i} with task:\n{task}\n")

    def evaluate_agent_output(self, agent_output):
        # Simple evaluation: flag outputs that are too brief or contain markers of incompleteness.
        if "incomplete" in agent_output.lower():
            return False, "The output appears to be incomplete."
        if len(agent_output.split()) < 10:
            return False, "The output is too brief."
        return True, ""

    def generate_followup(self, agent, last_response, evaluation_reason):
        # Aggregate agent conversation history for context.
        history_text = "\n".join([f"{role}: {msg}" for role, msg in agent.history])
        prompt = (
            f"Based on the following conversation history with Agent {agent.agent_id}:\n{history_text}\n\n"
            f"Agent's last response: {last_response}\n"
            f"Evaluation feedback: {evaluation_reason}\n\n"
            "Generate a follow-up instruction to improve the agent's output. "
            "Be concise and technical."
        )
        followup_response = self.chat.send_message(prompt)
        followup_instruction = followup_response.text.strip()
        print(f"\nConductor: Generated follow-up instruction for Agent {agent.agent_id}:\n{followup_instruction}\n")
        return followup_instruction

    def auto_evaluate_context(self, agent, context_response):
        eval_prompt = (
            f"Evaluate the following context confirmation from Agent {agent.agent_id}:\n" 
            f"{context_response}\n" 
            "Based on the previously provided task description, rate the agent's understanding on a scale from 0 (not understood) to 1 (fully understood). Provide only the numeric value."
        )
        eval_response = self.chat.send_message(eval_prompt)
        try:
            score = float(eval_response.text.strip())
            return score >= 0.8
        except:
            return False

    def auto_evaluate_output(self, agent, output):
        eval_prompt = (
            f"Evaluate the following output from Agent {agent.agent_id}:\n" 
            f"{output}\n" 
            "Based on the task context and requirements provided earlier, rate how well this output aligns with the assigned task on a scale from 0 (unsatisfactory) to 1 (perfectly satisfactory). Provide only the numeric value."
        )
        eval_response = self.chat.send_message(eval_prompt)
        try:
            score = float(eval_response.text.strip())
            return score >= 0.8
        except:
            return False

    def run_agents(self):
        import os
        for agent in self.agents:
            print(f"\nConductor: Initiating context phase for Agent {agent.agent_id}")
            # Context Providing Phase
            context_phase_success = False
            attempts = 0
            while not context_phase_success and attempts < 3:
                context_response = agent.provide_context()
                print(f"\nConductor: Evaluating Agent {agent.agent_id}'s context confirmation...")
                if self.auto_evaluate_context(agent, context_response):
                    context_phase_success = True
                    print(f"Conductor: Agent {agent.agent_id} context evaluation passed.")
                else:
                    print(f"Conductor: Agent {agent.agent_id} context evaluation failed. Retrying...")
                    attempts += 1
            if not context_phase_success:
                print(f"Conductor: Agent {agent.agent_id} failed to confirm understanding after 3 attempts. Skipping this agent.")
                continue

            # Output Phase
            print(f"\nConductor: Initiating output phase for Agent {agent.agent_id}")
            output_phase_success = False
            attempts = 0
            final_response = ""
            while not output_phase_success and attempts < 3:
                final_response = agent.attempt_task()
                print(f"\nConductor: Evaluating output from Agent {agent.agent_id}...")
                if self.auto_evaluate_output(agent, final_response):
                    output_phase_success = True
                    print(f"Conductor: Agent {agent.agent_id} output evaluation passed.")
                else:
                    print(f"Conductor: Agent {agent.agent_id} output evaluation failed. Generating follow-up instruction...")
                    suggestion = self.generate_followup(agent, final_response, "Output does not sufficiently align with the assigned task.")
                    final_response = agent.attempt_task(initial_prompt=suggestion)
                    attempts += 1
            self.final_outputs.append((agent.agent_id, final_response))
            
            # Save conversation history to a file in the 'conversations' folder
            os.makedirs("conversations", exist_ok=True)
            filename = f"conversations/agent_{agent.agent_id}.txt"
            with open(filename, "w") as f:
                for role, msg in agent.history:
                    f.write(f"{role}: {msg}\n")

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

        # Step 5 & 6: Let the agents converse with the Conductor (without interrupting the process).
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
        self.history = []      # Store the conversation history as (role, message) tuples.
        # Initialize the conversation with the provided context.
        init_response = self.chat.send_message(self.context)
        self.history.append(("System (context)", self.context))
        self.history.append(("Agent", init_response.text))

    def provide_context(self):
        prompt = f"Agent {self.agent_id}, please confirm your understanding of your task without executing it. Restate your understanding of your assigned task."
        self.history.append(("Conductor", prompt))
        response = self.chat.send_message(prompt)
        self.history.append(("Agent", response.text))
     
        return response.text

    def attempt_task(self, initial_prompt=None):
        prompt = initial_prompt if initial_prompt is not None else self.task
        self.history.append(("Conductor", prompt))
        response = self.chat.send_message(prompt)
        self.history.append(("Agent", response.text))
        return response.text


def main():
    # Step 0: Set the conductor's initial context.
    base_context = (
        "You are an AI responsible for managing a multi-agent system. You are the Conductor. You will be given a task or problem, "
        "and you have to reason out a plan from the assigned task. You will not carry out the task directly; instead, you will delegate to agents. "
        "You have the ability to spawn up to 2 agents to solve a given problem. You will give them more context on the problem and assign a task. "
        "They can either do the same task, different parts of a task, or multiple subtasks to accomplish a goal. "
        "Your functions include generating a plan, spawning agents, evaluating their outputs, and compiling the final results. "
        "Ensure that agents are re-prompted if their output is not technically sound, complete, and accurate. "
        "Your dialogue with the user and agents will be visible in the terminal."
    )
    conductor = Conductor(base_context)
    conductor.orchestrate()


if __name__ == "__main__":
    main()