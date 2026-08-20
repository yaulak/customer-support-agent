from langchain_core.messages import HumanMessage
from langgraph.types import Command

from src.agent.graph import graph


if __name__ == "__main__":
    config = {"configurable": {"thread_id": "demo-thread-1"}}

    while True:
        user_message = input("You: ")

        if user_message.lower() == "quit":
            break

        new_message = HumanMessage(content=user_message)
        result = graph.invoke({"messages": [new_message]}, config=config)

        while result.get("__interrupt__"):
            approval_request = result["__interrupt__"][0].value
            print(f"Requested action: {approval_request['requested_action']}")
            print(f"Order ID: {approval_request['order_id']}")
            print(approval_request["question"])

            human_response = input("Approve? Type yes or no: ")
            approved = human_response.strip().lower() == "yes"
            result = graph.invoke(Command(resume=approved), config=config)

        print(f"Assistant: {result['messages'][-1].content}")
