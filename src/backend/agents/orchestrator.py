import os
import json
import time
from pydantic import BaseModel
from typing import Annotated

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from operator import add
from IPython.display import Image
from langsmith import traceable

from src.backend.prompt_builder import build_prompt_from_config
from src.backend.agents.topic_finder import TopicFinder, TopicState
from src.backend.agents.deep_researcher import DeepResearcher, ResearchState, Article
from src.backend.agents.newsletter_writer import NewsletterWriter, NewsletterState, News
from src.backend.logger import logger

from src.backend.paths import DATA_DIR

load_dotenv()

class OrchestratorState(BaseModel):
    topics: list = []
    news_articles: Annotated[list[Article], add] = []
    news: Annotated[list[News], add] = []
    newsletter: str = "" # markdown

class Orchestrator:
    def __init__(self, groq_api_key: str):
        self.topic_finder = TopicFinder(groq_api_key)
        self.deep_researcher = DeepResearcher(groq_api_key)
        self.newsletter_writer = NewsletterWriter(groq_api_key)

        logger.info("[Orchestrator] Agent initialized")

    def call_topic_finder(self, state: OrchestratorState) -> dict:
        """
        Invoke TopicFinder graph and propagate results into orchestrator state
        """
        logger.info("[Orchestrator] Invoking TopicFinder graph...")
        tf_graph = self.topic_finder.build_topic_finder_graph()

        # Add RSS feed URLs to the system prompt
        rss_feed_urls = "\n\nRSS Feed URLs:\n"
        for url in self.topic_finder.rss_feed_urls:
            rss_feed_urls += f"- {url}\n"

        system_prompt = build_prompt_from_config(config=self.topic_finder.topic_finder_prompt, input_data=rss_feed_urls)
        logger.info(f"[Orchestrator] Topic Finder System prompt: {system_prompt}")

        initial_state = TopicState(messages=[SystemMessage(content=system_prompt)], topics=[])

        result_state = tf_graph.invoke(initial_state, config={"recursion_limit": 100})

        state.topics = result_state["topics"]
        return {"topics": state.topics}
    
    def call_deep_researcher(self, state: OrchestratorState) -> dict:
        """
        Invoke DeepResearcher graph and propagate results into orchestrator state
        """
        logger.info("[Orchestrator] Invoking DeepResearcher graph...")
        dr_graph = self.deep_researcher.build_deep_researcher_graph()

        # Add RSS feed URLs to the system prompt
        rss_feed_urls = "\n\nRSS Feed URLs:\n"
        for url in self.deep_researcher.rss_feed_urls:
            rss_feed_urls += f"- {url}\n"

        self.deep_researcher.topic = state.topics[0]
        # Add topic to the system prompt    
        topic = f"\n\nTopic:\n{self.deep_researcher.topic}"

        system_prompt = build_prompt_from_config(config=self.deep_researcher.deep_researcher_prompt, input_data=rss_feed_urls + topic)
        logger.info(f"[Orchestrator] Deep Researcher System prompt: {system_prompt}")

        initial_state = ResearchState(messages=[SystemMessage(content=system_prompt)], topic=self.deep_researcher.topic, news_articles=[])

        try:
            time.sleep(2)
            result_state = dr_graph.invoke(initial_state, config={"recursion_limit": 100})
        except Exception as e: # Rate Limit Error
            time.sleep(2)
            result_state = dr_graph.invoke(initial_state, config={"recursion_limit": 100})
        time.sleep(2)

        state.news_articles = result_state["news_articles"]

        # Remove the first topic from the list
        if len(state.topics) > 1:
            state.topics = state.topics[1:]
        elif len(state.topics) == 1:
            state.topics = []

        # Build news object
        state.news = [News(topic=self.deep_researcher.topic, news_articles=state.news_articles)]

        return {"topics": state.topics, "news_articles": state.news_articles, "news": state.news}
    
    def should_research_continue(self, state: OrchestratorState):
        """Decides whether to call deep researcher or provide final answer."""
        remaining_topics = len(state.topics)
 
        if remaining_topics > 0:
            return "yes"  # Agent wants to perform deep research if there are more topics
        return "no"  # Agent is ready to respond
    
    def call_newsletter_writer(self, state: OrchestratorState) -> dict:
        """
        Invoke NewsletterWriter graph and propagate results into orchestrator state
        """
        logger.info("[Orchestrator] Invoking NewsletterWriter graph...")
        nw_graph = self.newsletter_writer.build_newsletter_writer_graph()

        # Add news (topics and news articles) to the system prompt
        news = "\n\nNews:\n"
        for n in state.news:
            for topic, articles in n:
                news += f"Topic: {topic}\n"
                news += "Articles:\n"
                for article in articles:
                    news += str(article)

        system_prompt = build_prompt_from_config(config=self.newsletter_writer.newsletter_writer_prompt, input_data=news)
        logger.info(f"[Orchestrator] Newsletter Writer System prompt: {system_prompt}")

        initial_state = NewsletterState(messages=[SystemMessage(content=system_prompt)], newsletter="")

        result_state = nw_graph.invoke(initial_state, config={"recursion_limit": 100})

        state.newsletter = result_state["newsletter"]
        return {"newsletter": state.newsletter}

    @traceable(name="Orchestrator")
    def build_orchestrator_graph(self) -> CompiledStateGraph:
        """Builds the orchestrator graph."""
        logger.info("[Orchestrator] Building graph...")
        workflow = StateGraph(OrchestratorState)

        workflow.add_node("topic_finder_agent", self.call_topic_finder)
        workflow.add_node("deep_research_agent", self.call_deep_researcher)
        workflow.add_node("newsletter_writer_agent", self.call_newsletter_writer)

        workflow.set_entry_point("topic_finder_agent")
        workflow.add_edge("topic_finder_agent", "deep_research_agent")

        # Add the flow logic
        workflow.add_conditional_edges("deep_research_agent", self.should_research_continue, {"yes": "deep_research_agent", "no": "newsletter_writer_agent"})
        workflow.add_edge("newsletter_writer_agent", END)

        return workflow.compile()


def main():
    groq_api_key = os.getenv("GROQ_API_KEY")
    orchestrator = Orchestrator(groq_api_key)
    graph = orchestrator.build_orchestrator_graph()

    # Visualize the graph
    graph_png = Image(graph.get_graph().draw_mermaid_png())
    # Save the graph to a file
    with open("resources/orchestrator_graph.png", "wb") as f:
        f.write(graph_png.data)

    # Run the orchestrator multiple times to generate datasets
    n = 3
    for i in range(n):
        # Invoke the graph with an initial state
        initial_state = OrchestratorState()
        final_state = graph.invoke(initial_state)

        # Save the final state to a file
        # Create the output directory if it doesn't exist
        output_path = os.path.join(DATA_DIR, f"generated_dataset_{i}.json")
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({
                "generated_news": [n.dict() for n in final_state["news"]]
            }, f, indent=2)

        # Save the newsletter to a markdown file
        newsletter_path = os.path.join(DATA_DIR, f"generated_newsletter_{i}.md")
        with open(newsletter_path, "w", encoding="utf-8") as f:
            f.write(final_state["newsletter"])

        print("Topics:\n")
        for i, news in enumerate(final_state["news"]):
            print(news.topic)

        print("\nNews:\n")
        for i, news in enumerate(final_state["news"]):
            print(f"News {i + 1}:")
            for n in news:
                print(n)
        print("\nNewsletter:\n")
        print()
        print(final_state["newsletter"])


if __name__ == "__main__":
    main()
