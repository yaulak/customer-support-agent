from langchain_core.messages import HumanMessage

from src.agent.graph import graph


if __name__ == "__main__":
    config = {"configurable": {"thread_id": "demo-thread-1"}}

    while True:
        user_message = input("You: ")

        if user_message.lower() == "quit":
            break

        new_message = HumanMessage(content=user_message)
        result = graph.invoke({"messages": [new_message]}, config=config)

        print(f"Assistant: {result['messages'][-1].content}")
