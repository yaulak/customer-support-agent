from langchain_core.messages import HumanMessage

from src.agent.graph import graph
from src.tools.order_cancellation import create_pending_review_after_interrupt


if __name__ == "__main__":
    config = {"configurable": {"thread_id": "demo-thread-1"}}

    while True:
        user_message = input("You: ")

        if user_message.lower() == "quit":
            break

        new_message = HumanMessage(content=user_message)
        result = graph.invoke({"messages": [new_message]}, config=config)

        if result.get("__interrupt__"):
            review = result["__interrupt__"][0].value
            ticket, _ = create_pending_review_after_interrupt(
                review,
                "demo-thread-1",
            )
            print(
                "Assistant: Cancellation requires admin review. "
                f"Pending ticket: {ticket['ticket_id']}."
            )
            continue

        print(f"Assistant: {result['messages'][-1].content}")
