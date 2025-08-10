import os
import io
import sys
import json
import pytest
from datetime import datetime
from dotenv import load_dotenv

from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from src.backend.agents.newsletter_writer import NewsletterWriter, NewsletterState
from src.backend.utils import load_yaml_config
from src.backend.prompt_builder import build_prompt_from_config
from src.backend.paths import APP_CONFIG_FPATH, PROMPT_CONFIG_FPATH, DATA_DIR, EVAL_RESULTS_DIR
from langchain_core.messages import SystemMessage

load_dotenv()


# ------------------------
# Data Loading Utilities
# ------------------------
def load_generated_news(index: str):
    """Load generated news dataset by index."""
    fn = os.path.join(DATA_DIR, f"generated_dataset_{index}.json")
    with open(fn, "r", encoding="utf-8") as f:
        data = json.load(f)["generated_news"]
    return data


def compose_initial_state(news_list):
    """Prepare NewsletterState for the NewsletterWriter agent."""
    prompt_cfg = load_yaml_config(PROMPT_CONFIG_FPATH)
    prompt = prompt_cfg["newsletter_writer_agent_prompt"]

    # Build input data string from news list
    news_str = ""
    for entry in news_list:
        news_str += f"Topic: {entry['topic']}\n"
        news_str += "Articles:\n"
        for art in entry.get("news_articles", []):
            if isinstance(art, str):
                news_str += f"{art}\n"
            elif isinstance(art, dict):
                news_str += f"{art.get('title', '')}\n{art.get('summary', '')}\n{art.get('content', '')}\n"

    system_prompt = build_prompt_from_config(config=prompt, input_data=news_str)
    return NewsletterState(messages=[SystemMessage(content=system_prompt)], news=news_list, newsletter="")


# ------------------------
# Result Saving Utilities
# ------------------------
def save_result(index: str, test_name: str, result: dict):
    """Save test result in a consolidated JSON file."""
    if not os.path.exists(EVAL_RESULTS_DIR):
        os.makedirs(EVAL_RESULTS_DIR, exist_ok=True)

    filepath = os.path.join(EVAL_RESULTS_DIR, "test_newsletter_writer.json")

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
    """Generate markdown summary of newsletter writer tests."""
    filepath = os.path.join(EVAL_RESULTS_DIR, "test_newsletter_writer.json")
    summary_path = os.path.join(EVAL_RESULTS_DIR, "test_newsletter_writer_summary.md")

    if not os.path.exists(filepath):
        print("Warning: No test results file found, skipping summary generation.")
        return

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            all_results = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        print("Warning: Could not read test results file, skipping summary generation.")
        return

    with open(summary_path, "w", encoding="utf-8") as md_file:
        md_file.write("# Newsletter Writer Summary Report\n\n")
        md_file.write(f"**Generated on**: {datetime.utcnow().isoformat()}\n\n")
        md_file.write("---\n\n")

        datasets_tested = set()
        for test_name, test_data in all_results.items():
            datasets_tested.update(test_data.keys())

        md_file.write("## Overview\n\n")
        md_file.write(f"- **Total Test Functions**: {len(all_results)}\n")
        md_file.write(f"- **Datasets Tested**: {sorted(datasets_tested)}\n")
        md_file.write(f"- **Total Test Cases**: {sum(len(test_data) for test_data in all_results.values())}\n\n")

        for test_name, test_data in all_results.items():
            md_file.write(f"## {test_name.replace('_', ' ').title()}\n\n")

            for dataset_idx in sorted(test_data.keys()):
                result = test_data[dataset_idx]
                status_emoji = "✅" if result.get("passed", False) else "❌"
                md_file.write(f"### Dataset {dataset_idx} {status_emoji}\n\n")

                if "structured_results" in result:
                    for test_result in result.get("structured_results", []):
                        md_file.write(f"**Test**: {test_result.get('test_name', 'Unknown')}\n\n")
                        for metric in test_result.get("metrics", []):
                            metric_status = "✅" if metric.get("success", False) else "❌"
                            md_file.write(f"- **{metric.get('name')}** {metric_status}\n")
                            md_file.write(f"  - Score: {metric.get('score', 0):.4f} (Threshold: {metric.get('threshold', 0):.4f})\n")
                            md_file.write(f"  - Reason: {metric.get('reason', '')[:200]}...\n")
                        md_file.write("\n")

        # Summary stats
        total_tests = 0
        passed_tests = 0
        for test_data in all_results.values():
            for result in test_data.values():
                total_tests += 1
                if result.get("passed", False):
                    passed_tests += 1
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0

        md_file.write("## Final Summary\n\n")
        md_file.write(f"- **Total Tests**: {total_tests}\n")
        md_file.write(f"- **Passed**: {passed_tests}\n")
        md_file.write(f"- **Failed**: {total_tests - passed_tests}\n")
        md_file.write(f"- **Success Rate**: {success_rate:.1f}%\n\n")

        if success_rate >= 80:
            md_file.write("🎉 **Overall Status**: EXCELLENT\n")
        elif success_rate >= 60:
            md_file.write("✅ **Overall Status**: GOOD\n")
        else:
            md_file.write("⚠️ **Overall Status**: NEEDS IMPROVEMENT\n")

    print(f"Summary report generated: {summary_path}")


# ------------------------
# Pytest Fixture
# ------------------------
@pytest.fixture(scope="session", autouse=True)
def generate_summary_after_tests():
    """Generate summary report after all tests complete."""
    yield
    generate_final_summary()


# ------------------------
# Tests
# ------------------------
@pytest.mark.parametrize("dataset_index", ["0", "1", "2"])
def test_generation_and_structure(dataset_index):
    """Test newsletter generation produces non-empty markdown."""
    groq_api = os.getenv("GROQ_API_KEY")
    assert groq_api, "GROQ_API_KEY not set."

    news_list = load_generated_news(dataset_index)
    writer = NewsletterWriter(groq_api)
    graph = writer.build_newsletter_writer_graph()
    state = compose_initial_state(news_list)
    result = graph.invoke(state, config={"recursion_limit": 100})
    newsletter = result.get("newsletter", "")

    passed = isinstance(newsletter, str) and bool(newsletter.strip())

    save_result(dataset_index, "generation_and_structure", {
        "dataset_index": dataset_index,
        "newsletter_length": len(newsletter),
        "passed": passed,
        "timestamp": datetime.utcnow().isoformat()
    })

    assert passed, "Newsletter markdown is empty or not a string."


@pytest.mark.parametrize("dataset_index", ["0", "1", "2"])
def test_newsletter_relevancy_and_faithfulness(dataset_index):
    """Evaluate that newsletter is relevant and faithful to articles."""
    groq_api = os.getenv("GROQ_API_KEY")
    assert groq_api

    news_list = load_generated_news(dataset_index)
    compiled_context = []
    for entry in news_list:
        for art in entry.get("news_articles", []):
            if isinstance(art, str):
                compiled_context.append(art)
            elif isinstance(art, dict):
                compiled_context.append(
                    f"{art.get('title','')}\n{art.get('summary','')}\n{art.get('content','')}"
                )

    state = compose_initial_state(news_list)
    writer = NewsletterWriter(groq_api)
    graph = writer.build_newsletter_writer_graph()
    result = graph.invoke(state, config={"recursion_limit": 100})
    newsletter_text = result.get("newsletter", "")

    rel_metric = AnswerRelevancyMetric(threshold=0.5)
    faith_metric = FaithfulnessMetric(threshold=0.5)

    test_case = LLMTestCase(
        name=f"newsletter_eval_{dataset_index}",
        input="Evaluate if the newsletter reflects the extracted news correctly.",
        actual_output=newsletter_text,
        retrieval_context=compiled_context
    )

    buffer = io.StringIO()
    sys_stdout = sys.stdout
    sys.stdout = buffer
    try:
        results = evaluate([test_case], [rel_metric, faith_metric])
    finally:
        sys.stdout = sys_stdout

    table_output = buffer.getvalue()

    passed = all(r.success for r in results.test_results)
    result_data = {
        "dataset_index": dataset_index,
        "passed": passed,
        "timestamp": datetime.utcnow().isoformat(),
        "deepeval_table_output": table_output.strip(),
        "structured_results": [
            {
                "test_name": tr.name,
                "success": tr.success,
                "metrics": [
                    {
                        "name": md.name,
                        "score": md.score,
                        "threshold": md.threshold,
                        "success": md.success,
                        "reason": md.reason,
                        "cost": md.evaluation_cost
                    }
                    for md in tr.metrics_data
                ]
            }
            for tr in results.test_results
        ]
    }

    save_result(dataset_index, "newsletter_relevancy_and_faithfulness", result_data)

    assert passed, f"Newsletter relevancy/faithfulness failed for dataset {dataset_index}"
