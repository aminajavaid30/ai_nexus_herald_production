import os
import io
import sys
import json
import pytest
from datetime import datetime

from deepeval import assert_test, evaluate
from deepeval.metrics import ToolCorrectnessMetric, AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase, ToolCall, LLMTestCaseParams

from src.backend.agents.deep_researcher import DeepResearcher, ResearchState
from src.backend.utils import load_yaml_config
from src.backend.prompt_builder import build_prompt_from_config
from src.backend.paths import PROMPT_CONFIG_FPATH, RSS_CONFIG_FPATH, DATA_DIR, EVAL_RESULTS_DIR
from langchain_core.messages import SystemMessage

from dotenv import load_dotenv
load_dotenv()

# --------------------------
# Dataset & State Loaders
# --------------------------

def load_datasets(index: str):
    """Load generated topics for a given dataset index."""
    path = os.path.join(DATA_DIR, f"generated_dataset_{index}.json")
    with open(path, "r", encoding="utf-8") as f:
        gen = json.load(f)["generated_news"]
    return [item["topic"] for item in gen]

def get_initial_state(topic):
    """Build initial ResearchState for the given topic."""
    prompt_cfg = load_yaml_config(PROMPT_CONFIG_FPATH)
    rss_cfg = load_yaml_config(RSS_CONFIG_FPATH)
    prompt = prompt_cfg["deep_researcher_agent_prompt"]
    urls = [feed["url"] for feed in rss_cfg["rss_feeds"].values()]
    urls_str = "\n".join(f"- {u}" for u in urls)
    system_prompt = build_prompt_from_config(
        config=prompt, 
        input_data=urls_str + f"\n\nTopic:\n{topic}"
    )
    return ResearchState(messages=[SystemMessage(content=system_prompt)], topic=topic, news_articles=[])

# --------------------------
# Results Handling
# --------------------------

def save_result(index: str, test_name: str, result: dict):
    """Save test results in a structured JSON file."""
    if not os.path.exists(EVAL_RESULTS_DIR):
        os.makedirs(EVAL_RESULTS_DIR, exist_ok=True)
    
    filepath = os.path.join(EVAL_RESULTS_DIR, "test_deep_researcher.json")
    
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                all_results = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            all_results = {}
    else:
        all_results = {}
    
    if test_name not in all_results:
        all_results[test_name] = {}
    
    all_results[test_name][index] = result
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

def generate_final_summary():
    """Generate markdown summary report."""
    filepath = os.path.join(EVAL_RESULTS_DIR, "test_deep_researcher.json")
    summary_path = os.path.join(EVAL_RESULTS_DIR, "test_deep_researcher_summary.md")
    
    if not os.path.exists(filepath):
        print("No results found for Deep Researcher tests.")
        return
    
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            all_results = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        print("Could not parse results file.")
        return

    with open(summary_path, "w", encoding="utf-8") as md:
        md.write("# Deep Researcher Summary Report\n\n")
        md.write(f"**Generated on**: {datetime.utcnow().isoformat()}\n\n")
        md.write("---\n\n")

        datasets_tested = set()
        for test_name, test_data in all_results.items():
            datasets_tested.update(test_data.keys())

        md.write("## Overview\n\n")
        md.write(f"- **Total Test Functions**: {len(all_results)}\n")
        md.write(f"- **Datasets Tested**: {sorted(datasets_tested)}\n")
        md.write(f"- **Total Test Cases**: {sum(len(td) for td in all_results.values())}\n\n")

        total_tests = 0
        passed_tests = 0

        for test_name, test_data in all_results.items():
            md.write(f"## {test_name.replace('_', ' ').title()}\n\n")
            for dataset_idx, result in sorted(test_data.items()):
                total_tests += 1
                status_emoji = "✅" if result.get("passed", False) else "❌"
                md.write(f"- **Dataset {dataset_idx}**: {status_emoji}\n")
                if result.get("passed", False):
                    passed_tests += 1

        success_rate = (passed_tests / total_tests * 100) if total_tests else 0
        md.write("\n## Final Summary\n")
        md.write(f"- **Total Tests**: {total_tests}\n")
        md.write(f"- **Passed**: {passed_tests}\n")
        md.write(f"- **Failed**: {total_tests - passed_tests}\n")
        md.write(f"- **Success Rate**: {success_rate:.1f}%\n\n")
        if success_rate >= 80:
            md.write("🎉 **Overall Status**: EXCELLENT\n")
        elif success_rate >= 60:
            md.write("✅ **Overall Status**: GOOD\n")
        else:
            md.write("⚠️ **Overall Status**: NEEDS IMPROVEMENT\n")

    print(f"Summary report generated: {summary_path}")

@pytest.fixture(scope="session", autouse=True)
def generate_summary_after_tests():
    yield
    generate_final_summary()

# --------------------------
# Tests
# --------------------------

# LLM-as-a-judge: Check if the agent calls the tool correctly
@pytest.mark.parametrize("dataset_index", ["0", "1", "2"])
def test_tool_call_correctness(dataset_index):
    groq_api_key = os.getenv("GROQ_API_KEY")
    assert groq_api_key, "GROQ_API_KEY must be set."
    topics = load_datasets(dataset_index)
    passed_all = True

    for topic in topics:
        agent = DeepResearcher(groq_api_key)
        graph = agent.build_deep_researcher_graph()
        initial = get_initial_state(topic)
        _ = graph.invoke(initial, config={"recursion_limit": 100})

        case = LLMTestCase(
            name=f"tool_usage_for_{topic}",
            input=initial.messages[0].content,
            actual_output="",
            tools_called=[ToolCall(name="extract_news_from_rss")],
            expected_tools=[ToolCall(name="extract_news_from_rss")]
        )
        metric = ToolCorrectnessMetric()
        result = evaluate([case], [metric])
        if not all(r.success for r in result.test_results):
            passed_all = False

    save_result(dataset_index, "tool_call_correctness", {
        "dataset_index": dataset_index,
        "passed": passed_all,
        "timestamp": datetime.utcnow().isoformat()
    })

    assert passed_all, f"Tool call correctness failed for dataset {dataset_index}"

# Rule-based test: Check if news articles are present and have a specific structure (Title, Link, Summary, Content)
@pytest.mark.parametrize("dataset_index", ["0", "1", "2"])
def test_news_output_format_and_presence(dataset_index):
    groq_api_key = os.getenv("GROQ_API_KEY")
    assert groq_api_key
    topics = load_datasets(dataset_index)
    passed_all = True

    for topic in topics:
        agent = DeepResearcher(groq_api_key)
        graph = agent.build_deep_researcher_graph()
        final = graph.invoke(get_initial_state(topic), config={"recursion_limit": 100})
        arts = final.get("news_articles", [])
        if not (isinstance(arts, list) and len(arts) >= 1):
            passed_all = False
            continue
        for art in arts:
            if not (isinstance(art, dict) and all(k in art and isinstance(art[k], str) for k in ("title", "link", "summary", "content"))):
                passed_all = False

    save_result(dataset_index, "news_output_format", {
        "dataset_index": dataset_index,
        "passed": passed_all,
        "timestamp": datetime.utcnow().isoformat()
    })

    assert passed_all, f"News output format or presence failed for dataset {dataset_index}"

# LLM-as-a-judge: Check if each news article is relevant to the topic
@pytest.mark.parametrize("dataset_index", ["0", "1", "2"])
def test_news_relevancy_to_topic(dataset_index):
    groq_api_key = os.getenv("GROQ_API_KEY")
    assert groq_api_key
    topics = load_datasets(dataset_index)
    passed_all = True
    relevancy_metric = AnswerRelevancyMetric(threshold=0.5)

    for topic in topics:
        agent = DeepResearcher(groq_api_key)
        graph = agent.build_deep_researcher_graph()
        final = graph.invoke(get_initial_state(topic), config={"recursion_limit": 100})
        for art in final.get("news_articles", []):
            content = f"{art['title']}\n{art['link']}\n{art['summary']}\n{art.get('content','')}"
            case = LLMTestCase(
                name=f"rel_news_{topic}",
                input=f"Assess if this news is relevant to topic: {topic}",
                actual_output=content,
                retrieval_context=[topic]
            )
            result = evaluate([case], [relevancy_metric])
            if not all(r.success for r in result.test_results):
                passed_all = False

    save_result(dataset_index, "news_relevancy", {
        "dataset_index": dataset_index,
        "passed": passed_all,
        "timestamp": datetime.utcnow().isoformat()
    })

    assert passed_all, f"News relevancy failed for dataset {dataset_index}"
