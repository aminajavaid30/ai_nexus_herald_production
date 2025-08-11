import os
import io
import sys
import json
import pytest
from datetime import datetime

from deepeval import evaluate
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from src.backend.paths import DATA_DIR, EVAL_RESULTS_DIR

from dotenv import load_dotenv
load_dotenv()


def load_generated_news(index: str):
    generated_data_path = os.path.join(DATA_DIR, f"generated_dataset_{index}.json")
    if not os.path.exists(generated_data_path):
        pytest.skip(f"Generated news dataset {index} not found.")
    with open(generated_data_path, "r", encoding="utf-8") as f:
        return json.load(f)["generated_news"]


def load_newsletter(index: str):
    newsletter_path = os.path.join(DATA_DIR, f"generated_newsletter_{index}.md")
    if not os.path.exists(newsletter_path):
        pytest.skip(f"Newsletter file for dataset {index} not found.")
    with open(newsletter_path, "r", encoding="utf-8") as f:
        return f.read()


def save_result(index: str, test_name: str, result: dict):
    """Save result for each dataset and test."""
    if not os.path.exists(EVAL_RESULTS_DIR):
        os.makedirs(EVAL_RESULTS_DIR, exist_ok=True)

    filepath = os.path.join(EVAL_RESULTS_DIR, "test_newsletter.json")

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
    """Generate summary report in markdown."""
    filepath = os.path.join(EVAL_RESULTS_DIR, "test_newsletter.json")
    summary_path = os.path.join(EVAL_RESULTS_DIR, "test_newsletter_summary.md")

    if not os.path.exists(filepath):
        print("Warning: No test results found, skipping summary generation.")
        return

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            all_results = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        print("Warning: Could not read results, skipping summary generation.")
        return

    with open(summary_path, "w", encoding="utf-8") as md_file:
        md_file.write("# Test Newsletter Summary Report\n\n")
        md_file.write(f"**Generated on**: {datetime.utcnow().isoformat()}\n\n")
        md_file.write("---\n\n")

        datasets_tested = set()
        for test_name, test_data in all_results.items():
            datasets_tested.update(test_data.keys())

        md_file.write("## Overview\n\n")
        md_file.write(f"- **Total Test Functions**: {len(all_results)}\n")
        md_file.write(f"- **Datasets Tested**: {sorted(datasets_tested)}\n")
        md_file.write(f"- **Total Test Cases**: {sum(len(test_data) for test_data in all_results.values())}\n\n")

        total_tests = 0
        passed_tests = 0

        for test_name, test_data in all_results.items():
            md_file.write(f"## {test_name.replace('_', ' ').title()}\n\n")
            for dataset_idx, result in sorted(test_data.items()):
                passed = result.get("passed", False)
                status_emoji = "✅" if passed else "❌"
                total_tests += 1
                if passed:
                    passed_tests += 1

                md_file.write(f"### Dataset {dataset_idx} {status_emoji}\n\n")

                if "structured_results" in result:
                    for test_result in result["structured_results"]:
                        md_file.write(f"**Test**: {test_result.get('test_name', 'Unknown')}\n\n")
                        for metric in test_result.get("metrics", []):
                            metric_status = "✅" if metric.get("success", False) else "❌"
                            md_file.write(f"**{metric.get('name')}** {metric_status}\n")
                            md_file.write(f"- **Score**: {metric.get('score', 0):.4f}\n")
                            md_file.write(f"- **Threshold**: {metric.get('threshold', 0):.4f}\n")
                            md_file.write(f"- **Cost**: ${metric.get('cost', 0):.4f}\n")
                            reason = metric.get("reason", "No reason provided")
                            if len(reason) > 200:
                                reason = reason[:200] + "..."
                            md_file.write(f"- **Reason**: {reason}\n\n")

        success_rate = (passed_tests / total_tests * 100) if total_tests else 0
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


@pytest.fixture(scope="session", autouse=True)
def generate_summary_after_tests():
    yield
    generate_final_summary()


# Rule-based test: Check presence of newsletter and generated news
@pytest.mark.parametrize("dataset_index", ["0", "1", "2"])
def test_newsletter_and_generated_news_presence(dataset_index):
    generated = load_generated_news(dataset_index)
    newsletter = load_newsletter(dataset_index)

    result = {
        "dataset_index": dataset_index,
        "passed": bool(generated and isinstance(generated, list) and newsletter.strip()),
        "generated_news_count": len(generated) if generated else 0,
        "newsletter_length": len(newsletter.strip()) if newsletter else 0,
        "timestamp": datetime.utcnow().isoformat()
    }

    save_result(dataset_index, "newsletter_and_generated_news_presence", result)

    assert generated and isinstance(generated, list), f"Generated news dataset {dataset_index} is missing or invalid."
    assert newsletter.strip(), f"Newsletter content for dataset {dataset_index} is empty."


# LLM-as-a-Judge: Evaluate newsletter relevance to news articles
@pytest.mark.parametrize("dataset_index", ["0", "1", "2"])
def test_newsletter_relevance_to_news(dataset_index):
    generated = load_generated_news(dataset_index)
    newsletter = load_newsletter(dataset_index)

    all_news_texts = []
    for item in generated:
        for news in item.get("news_articles", []):
            text = f"{news['title']}\n{news.get('link', '')}\n{news.get('summary', '')}\n{news.get('content', '')}"
            all_news_texts.append(text)

    relevance_metric = GEval(
        name="Newsletter Relevance",
        criteria="Determine whether the newsletter reflects the main ideas and topics of the extracted news articles.",
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.RETRIEVAL_CONTEXT],
        threshold=0.5
    )

    test_case = LLMTestCase(
        name=f"newsletter_relevance_test_{dataset_index}",
        input="Generated newsletter based on curated AI news.",
        actual_output=newsletter,
        retrieval_context=all_news_texts
    )

    buffer = io.StringIO()
    sys_stdout = sys.stdout
    sys.stdout = buffer
    try:
        results = evaluate([test_case], [relevance_metric])
    finally:
        sys.stdout = sys_stdout

    table_output = buffer.getvalue()
    test_results = results.test_results
    passed = all(r.success for r in test_results)

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
            for tr in test_results
        ]
    }

    save_result(dataset_index, "newsletter_relevance_to_news", result_data)

    assert passed, f"Newsletter relevance test failed for dataset {dataset_index}"