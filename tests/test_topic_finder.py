import os
import io
import sys
import json
import pytest
from datetime import datetime

from deepeval import assert_test, evaluate
from deepeval.metrics import ToolCorrectnessMetric, AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase, ToolCall, LLMTestCaseParams
from langchain_core.messages import SystemMessage

from dotenv import load_dotenv
load_dotenv()

from src.backend.agents.topic_finder import TopicFinder, TopicState
from src.backend.utils import load_yaml_config
from src.backend.prompt_builder import build_prompt_from_config
from src.backend.paths import DATA_DIR, PROMPT_CONFIG_FPATH, RSS_CONFIG_FPATH, EVAL_RESULTS_DIR


# -------------------------
# Data Loading Utilities
# -------------------------

def load_rss_titles():
    path = os.path.join(DATA_DIR, "rss_context_dataset.json")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)["rss_context"]
    return [item["title"] for item in data]


def get_initial_state():
    prompt_cfg = load_yaml_config(PROMPT_CONFIG_FPATH)
    rss_cfg = load_yaml_config(RSS_CONFIG_FPATH)
    prompt = prompt_cfg["topic_finder_agent_prompt"]
    urls = [feed["url"] for feed in rss_cfg["rss_feeds"].values()]
    urls_str = "\n".join(f"- {u}" for u in urls)
    system_prompt = build_prompt_from_config(config=prompt, input_data=urls_str)
    return TopicState(messages=[SystemMessage(content=system_prompt)], topics=[])


# -------------------------
# Result Saving Utilities
# -------------------------

def save_result(test_name: str, result: dict):
    """Save individual test result to a consolidated JSON file."""
    if not os.path.exists(EVAL_RESULTS_DIR):
        os.makedirs(EVAL_RESULTS_DIR, exist_ok=True)

    filepath = os.path.join(EVAL_RESULTS_DIR, "test_topic_finder.json")

    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                all_results = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            all_results = {}
    else:
        all_results = {}

    if test_name not in all_results:
        all_results[test_name] = []

    all_results[test_name].append(result)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)


def generate_final_summary():
    """Generate a markdown summary from all test results."""
    filepath = os.path.join(EVAL_RESULTS_DIR, "test_topic_finder.json")
    summary_path = os.path.join(EVAL_RESULTS_DIR, "test_topic_finder_summary.md")

    if not os.path.exists(filepath):
        print("Warning: No test results file found, skipping summary generation.")
        return

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            all_results = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        print("Warning: Could not read test results file, skipping summary generation.")
        return

    with open(summary_path, "w", encoding="utf-8") as md:
        md.write("# Test Topic Finder Summary Report\n\n")
        md.write(f"**Generated on**: {datetime.utcnow().isoformat()}\n\n")
        md.write("---\n\n")

        for test_name, results in all_results.items():
            md.write(f"## {test_name.replace('_', ' ').title()}\n\n")

            if test_name == "tool_call_correctness":
                md.write("| Timestamp | Passed |\n")
                md.write("|-----------|--------|\n")
                for r in results:
                    md.write(f"| {r['timestamp']} | {'✅' if r['passed'] else '❌'} |\n")
                md.write("\n")

            elif test_name == "topics_structure_and_count":
                md.write("| Timestamp | Topic Count | Passed |\n")
                md.write("|-----------|-------------|--------|\n")
                for r in results:
                    md.write(f"| {r['timestamp']} | {r['topic_count']} | {'✅' if r['passed'] else '❌'} |\n")
                md.write("\n")

            elif test_name == "topic_relevancy":
                for r in results:
                    md.write(f"### Run at {r['timestamp']} {'✅' if r['passed'] else '❌'}\n\n")
                    for metric in r.get("metrics", []):
                        md.write(f"- **{metric['name']}**: {metric['score']:.4f} (Threshold: {metric['threshold']}) {'✅' if metric['success'] else '❌'}\n")
                        md.write(f"  Reason: {metric['reason'][:150]}...\n")
                    md.write("\n")

    print(f"Summary report generated: {summary_path}")


@pytest.fixture(scope="session", autouse=True)
def generate_summary_after_tests():
    yield
    generate_final_summary()


# -------------------------
# Tests
# -------------------------

# LLM-as-a-judge: Check if the tool is called correctly
def test_tool_call_correctness():
    """Check if extract_titles_from_rss tool is invoked."""
    groq_api_key = os.getenv("GROQ_API_KEY")
    assert groq_api_key, "GROQ_API_KEY must be set."

    agent = TopicFinder(groq_api_key)
    graph = agent.build_topic_finder_graph()
    initial = get_initial_state()
    final = graph.invoke(initial, config={"recursion_limit": 100})

    test_case = LLMTestCase(
        name="tool_invocation",
        input=initial.messages[0].content,
        actual_output="",
        tools_called=[ToolCall(name="extract_titles_from_rss")],
        expected_tools=[ToolCall(name="extract_titles_from_rss")]
    )

    metric = ToolCorrectnessMetric()

    buffer = io.StringIO()
    sys_stdout = sys.stdout
    sys.stdout = buffer
    try:
        results = evaluate([test_case], [metric])
    finally:
        sys.stdout = sys_stdout

    passed = all(r.success for r in results.test_results)
    save_result("tool_call_correctness", {
        "timestamp": datetime.utcnow().isoformat(),
        "passed": passed,
        "deepeval_table_output": buffer.getvalue().strip()
    })

    assert passed, "Tool call correctness test failed."


# Rule based test: Ensure 5 non-empty topics are returned as a list of strings
def test_topics_structure_and_count():
    """Ensure exactly 5 non-empty topic strings are returned."""
    groq_api_key = os.getenv("GROQ_API_KEY")
    assert groq_api_key

    agent = TopicFinder(groq_api_key)
    graph = agent.build_topic_finder_graph()
    final = graph.invoke(get_initial_state(), config={"recursion_limit": 100})
    topics = final.get("topics", [])

    passed = (
        isinstance(topics, list) and
        len(topics) == 5 and
        all(isinstance(t, str) and t.strip() for t in topics)
    )

    save_result("topics_structure_and_count", {
        "timestamp": datetime.utcnow().isoformat(),
        "topic_count": len(topics),
        "passed": passed
    })

    assert passed, f"Expected 5 valid topics, got {len(topics)}"


# LLM-as-a-judge: Check if each topic relates to RSS titles
def test_topic_relevancy():
    """Use LLM to check if each topic relates to RSS titles."""
    titles = load_rss_titles()
    groq_api_key = os.getenv("GROQ_API_KEY")
    agent = TopicFinder(groq_api_key)
    graph = agent.build_topic_finder_graph()
    final = graph.invoke(get_initial_state(), config={"recursion_limit": 100})
    topics = final.get("topics", [])

    metric = AnswerRelevancyMetric(threshold=0.5)

    test_cases = [
        LLMTestCase(
            input="Evaluate the relevance of this topic to the given RSS titles.",
            actual_output=topic,
            retrieval_context=titles
        ) for topic in topics
    ]

    buffer = io.StringIO()
    sys_stdout = sys.stdout
    sys.stdout = buffer
    try:
        results = evaluate(test_cases, [metric])
    finally:
        sys.stdout = sys_stdout

    passed = all(r.success for r in results.test_results)

    structured_metrics = []
    for tr in results.test_results:
        for md in tr.metrics_data:
            structured_metrics.append({
                "name": md.name,
                "score": md.score,
                "threshold": md.threshold,
                "success": md.success,
                "reason": md.reason,
                "cost": md.evaluation_cost
            })

    save_result("topic_relevancy", {
        "timestamp": datetime.utcnow().isoformat(),
        "passed": passed,
        "deepeval_table_output": buffer.getvalue().strip(),
        "metrics": structured_metrics
    })

    assert passed, "One or more topics were not relevant to RSS titles."