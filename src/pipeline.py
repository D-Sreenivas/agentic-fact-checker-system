from src.agent import FactCheckingAgent


def fact_check(claim):

    agent = FactCheckingAgent()

    return agent.run(claim)


if __name__ == "__main__":

    claim = input(
        "Enter a claim to fact-check: "
    )

    result = fact_check(claim)

    print("\n==============================")
    print("FACT CHECK RESULT")
    print("==============================")

    print(result.get("result", "No result generated."))

    print("\n==============================")
    print("EVIDENCE")
    print("==============================")

    evidence = result.get("evidence", [])

    if not evidence:

        print("No evidence found.")

    else:

        for score, document in evidence:

            print("\nTitle:")
            print(document.metadata.get("title", ""))

            print("\nURL:")
            print(document.metadata.get("url", ""))

            print("\nRelevance Score:")
            print(f"{score:.4f}")

            print("\nContent:")
            print(document.page_content)

            print("-" * 60)