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

from src.backend.tools import extract_news_from_rss
from src.backend.utils import load_yaml_config
from src.backend.prompt_builder import build_prompt_from_config
from src.backend.paths import APP_CONFIG_FPATH, PROMPT_CONFIG_FPATH, RSS_CONFIG_FPATH
from src.backend.logger import logger

load_dotenv()

class Article(BaseModel):
    title: str
    link: str
    summary: str
    content: str

class ResearchState(BaseModel):
    news_articles: Annotated[list[Article], add] = []
    topic: str
    messages: Annotated[list, add_messages]

class DeepResearcher:
    def __init__(self, groq_api_key: str):
        # Load application configurations
        app_config = load_yaml_config(APP_CONFIG_FPATH)
        self.llm_model = app_config["llm"]

        # Load prompt configurations
        prompt_config = load_yaml_config(PROMPT_CONFIG_FPATH)
        self.deep_researcher_prompt = prompt_config["deep_researcher_agent_prompt"]

        rss_config = load_yaml_config(RSS_CONFIG_FPATH)
        rss_feeds = rss_config["rss_feeds"]
        # Extract RSS feed URLs
        self.rss_feed_urls = [feed["url"] for feed in rss_feeds.values()]

        self.llm = ChatGroq(api_key=groq_api_key, model_name=self.llm_model)

        logger.info("[DeepResearcher] Agent initialized")

    def get_tools(self):
        """
        Create and return a list of tools that the agent can use.
        Currently, it includes an RSS feed extraction tool for finding trending topics in AI.
        
        Returns:
            list: A list of tools available to the agent.
        """
        return [
            # self.extract_titles_from_rss
            extract_news_from_rss
        ]
    
    # The LLM node - where your agent thinks and decides
    def search_news(self, state: ResearchState) -> dict:
        """
        Search news articles for trending topics.
        This function uses RSS news extraction tool to find news articles related to trending topics in AI and updates the state with the results.

        Args:
            state (ResearchState): The current state of the deep researcher, which will be updated with the found news articles.

        Returns:
            dict: A dictionary containing the updated state with the found news articles.
        """

        # Use the LLM to find news articles using the response from rss news extraction tool
        logger.info("[DeepResearcher] Calling LLM...")
        response = self.llm.invoke(state.messages)
        
        # If tool instructions were included, send directly to tool node
        if getattr(response, "tool_calls", None):
            return {"messages": [response]}
        
        # Return if the response is empty
        if not response.content or response.content.strip() == "":
            return {"messages": [response]}

        # Final JSON processing
        try:
            # Convert response to a list of Topic objects
            data = json.loads(response.content)
            state.news_articles = [list[Article](item.values())[0] for item in data["articles"]]
        except json.JSONDecodeError as e:
            logger.error("[DeepResearcher] Failed to parse JSON:", e)
            raise e

        return {"messages": [response], "news_articles": state.news_articles}
    
    # The tools node - where the agent takes action
    def tools_node(self, state: ResearchState):
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
            logger.info(f"[DeepResearcher] Executing tool: {tool.name} for topic: {state.topic}")
            
            # Send the result back to the agent
            tool_messages.append(ToolMessage(
                content=str(result),
                tool_call_id=tool_call["id"]
            ))
        
        return {"messages": tool_messages}
    
    # Decision function - should we use tools or finish?
    def should_continue(self, state: ResearchState):
        """Decides whether to use tools or provide final answer."""
        last_message = state.messages[-1]
        
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"  # Agent wants to use tools
        return END  # Agent is ready to respond

    # Build graph
    @traceable(name="DeepResearcher")
    def build_deep_researcher_graph(self) -> CompiledStateGraph:
        logger.info("[DeepResearcher] Building graph...")
        # Bind tools once
        tools = self.get_tools()
        self.llm = self.llm.bind_tools(tools)
        # tool_node = ToolNode(tools)

        workflow = StateGraph(ResearchState)

        # Register nodes
        workflow.add_node("search_news", self.search_news)
        workflow.add_node("tools", self.tools_node)
        # workflow.add_node("tools", tool_node)
        
        # Set entry point
        workflow.set_entry_point("search_news")

        # Add the flow logic
        workflow.add_conditional_edges("search_news", self.should_continue, {"tools": "tools", END: END})
        workflow.add_edge("tools", "search_news")  # After using tools, go back to thinking

        return workflow.compile()


def main():
    # Initialize the topic finder with Groq API key
    groq_api_key = os.getenv("GROQ_API_KEY")
    deep_researcher = DeepResearcher(groq_api_key)
    graph = deep_researcher.build_deep_researcher_graph()

    # Visualize the graph
    graph_png = Image(graph.get_graph().draw_mermaid_png())
    # Save the graph to a file
    with open("resources/deep_researcher_graph.png", "wb") as f:
        f.write(graph_png.data)

    # Add RSS feed URLs to the system prompt
    rss_feed_urls = "\n\nRSS Feed URLs:\n"
    for url in deep_researcher.rss_feed_urls:
        rss_feed_urls += f"- {url}\n"

    deep_researcher.topic = 'Quantum computers and their advancements'
    # Add topic to the system prompt    
    topic = f"\n\nTopic:\n{deep_researcher.topic}"

    system_prompt = build_prompt_from_config(config=deep_researcher.deep_researcher_prompt, input_data=rss_feed_urls + topic)

    initial_state = ResearchState(messages=[SystemMessage(content=system_prompt)], topic=deep_researcher.topic, news_articles=[])

    final_state = graph.invoke(initial_state, config={"recursion_limit": 100})

    print(f"\n === News Articles for the Topic: {deep_researcher.topic} ===")
    for idx, article in enumerate(final_state["news_articles"], 1):
        print(f"{idx}. {article}")


if __name__ == "__main__":
    main()