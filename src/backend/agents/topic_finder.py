import os
import json
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Annotated
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from operator import add
from IPython.display import Image
from langsmith import traceable

from src.backend.tools import extract_titles_from_rss
from src.backend.utils import load_yaml_config
from src.backend.prompt_builder import build_prompt_from_config
from src.backend.paths import APP_CONFIG_FPATH, PROMPT_CONFIG_FPATH, RSS_CONFIG_FPATH
from src.backend.logger import logger

load_dotenv()

class TopicState(BaseModel):
    topics: Annotated[list[str], add] = []
    messages: Annotated[list, add_messages]

class TopicFinder:
    def __init__(self, groq_api_key: str):
        # Load application configurations
        app_config = load_yaml_config(APP_CONFIG_FPATH)
        self.llm_model = app_config["llm"]

        # Load prompt configurations
        prompt_config = load_yaml_config(PROMPT_CONFIG_FPATH)
        self.topic_finder_prompt = prompt_config["topic_finder_agent_prompt"]

        rss_config = load_yaml_config(RSS_CONFIG_FPATH)
        rss_feeds = rss_config["rss_feeds"]
        # Extract RSS feed URLs
        self.rss_feed_urls = [feed["url"] for feed in rss_feeds.values()]

        self.llm = ChatGroq(api_key=groq_api_key, model_name=self.llm_model)

        logger.info("[TopicFinder] Agent initialized")

    def get_tools(self):
        """
        Create and return a list of tools that the agent can use.
        Currently, it includes an RSS feed extraction tool for finding trending topics in AI.
        
        Returns:
            list: A list of tools available to the agent.
        """
        return [
            # self.extract_titles_from_rss
            extract_titles_from_rss
        ]
    
    # The LLM node - where your agent thinks and decides
    def search_topics(self, state: TopicState) -> dict:
        """
        Search trending topics in AI.
        This function uses RSS title extraction tool to find trending topics in AI and updates the state with the results.

        Args:
            state (TopicState): The current state of the topic finder, which will be updated with the found topics.

        Returns:
            dict: A dictionary containing the updated state with the found topics.
        """

        # Use the LLM to find topics using the response from rss title extraction tool
        logger.info("[TopicFinder] Calling LLM...")

        response = self.llm.invoke(state.messages)

        # If tool instructions were included, send directly to tool node
        if getattr(response, "tool_calls", None):
            return {"messages": [response]}

        # Final JSON processing
        try:
            # Convert response to a list of Topic objects
            data = json.loads(response.content)
            state.topics = [list[str](item.values())[0] for item in data["topics"]]
        except json.JSONDecodeError as e:
            logger.error("[TopicFinder] Failed to parse JSON:", e)
            raise e

        return {"messages": [response], "topics": state.topics}
    
    # The tools node - where the agent takes action
    def tools_node(self, state: TopicState):
        """The agent's hands - executes the chosen tools."""
        tools = self.get_tools()
        tool_registry = {tool.name: tool for tool in tools}
        
        last_message = state.messages[-1]
        tool_messages = []

        # Execute each tool the agent requested
        for tool_call in last_message.tool_calls:
            tool = tool_registry[tool_call["name"]]
            result = tool.invoke(tool_call["args"])

            # Print the tool call
            logger.info(f"[TopicFinder] Executing tool: {tool.name}")
            
            # Send the result back to the agent
            tool_messages.append(ToolMessage(
                content=str(result),
                tool_call_id=tool_call["id"]
            ))
        
        return {"messages": tool_messages}
    
    # Decision function - should we use tools or finish?
    def should_continue(self, state: TopicState):
        """Decides whether to use tools or provide final answer."""
        last_message = state.messages[-1]
        
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"  # Agent wants to use tools
        return END  # Agent is ready to respond

    # Build graph
    @traceable(name="TopicFinder")
    def build_topic_finder_graph(self) -> CompiledStateGraph:
        logger.info("[TopicFinder] Building graph...")
        # Bind tools once
        tools = self.get_tools()
        self.llm = self.llm.bind_tools(tools)
        # tool_node = ToolNode(tools)

        workflow = StateGraph(TopicState)

        # Register nodes
        workflow.add_node("search_topics", self.search_topics)
        workflow.add_node("tools", self.tools_node)
        # workflow.add_node("tools", tool_node)
        
        # Set entry point
        workflow.set_entry_point("search_topics")

        # Add the flow logic
        workflow.add_conditional_edges("search_topics", self.should_continue, {"tools": "tools", END: END})
        workflow.add_edge("tools", "search_topics")  # After using tools, go back to thinking

        return workflow.compile()


def main():
    # Initialize the topic finder with Groq API key
    groq_api_key = os.getenv("GROQ_API_KEY")
    topic_finder = TopicFinder(groq_api_key)
    graph = topic_finder.build_topic_finder_graph()

    # Visualize the graph
    graph_png = Image(graph.get_graph().draw_mermaid_png())
    # Save the graph to a file
    with open("resources/topic_finder_graph.png", "wb") as f:
        f.write(graph_png.data)

    # Add RSS feed URLs to the system prompt
    rss_feed_urls = "\n\nRSS Feed URLs:\n"
    for url in topic_finder.rss_feed_urls:
        rss_feed_urls += f"- {url}\n"
    system_prompt = build_prompt_from_config(config=topic_finder.topic_finder_prompt, input_data=rss_feed_urls)

    initial_state = TopicState(messages=[SystemMessage(content=system_prompt)], topics=[])

    final_state = graph.invoke(initial_state, config={"recursion_limit": 100})

    print("\n === Trending Topics in AI ===")
    print(final_state["topics"])


if __name__ == "__main__":
    main()