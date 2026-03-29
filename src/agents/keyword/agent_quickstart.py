"""Test script for keyword-based wine agent."""

from src.agents.keyword.agent import create_keyword_agent

def main():
    """Run interactive test queries against the keyword-based wine agent."""
    print('Testing Keyword-Based Wine Agent')
    print('=' * 60)

    # Create keyword agent
    agent = create_keyword_agent(verbose=True)
    print(f'✓ Created keyword agent with {len(agent.get_available_tools())} tools')
    print()

    # # Generate and save the agent graph visualization
    # graph_png = agent.agent.get_graph(xray=True).draw_mermaid_png()
    # with open("graph.png", "wb") as f:
    #     f.write(graph_png)
    # print("✓ Agent graph saved to graph.png")
    # sys.exit()

    # Test queries
    test_queries = [
        ('What wines do I have in my cellar?', 'cellar'),
        ('What are my preferred white wines?', 'taste'),
        ('What Romanian wine pairs with steak?', 'pairing'),
        ('What is malolactic fermentation?', 'knowledge')
    ]

    for i, (query, expected_type) in enumerate(test_queries, 1):
        print(f'{i}. Query: {query}')
        print(f'   Expected type: {expected_type}')
        print('-' * 60)
        try:
            result = agent.invoke(query)
            print(f'   Detected type: {result["query_type"]}')
            print(f'   Match: {"✓" if result["query_type"] == expected_type else "✗"}')
            print(f'   Answer: {result["final_answer"]}')
            print()
        except Exception as e:
            print(f'   Error: {e}')
            import traceback
            traceback.print_exc()
            print()



if __name__ == '__main__':
    main()

