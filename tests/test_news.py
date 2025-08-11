import os
import io
import sys
import json
import pytest
from datetime import datetime

from deepeval import assert_test, evaluate
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from src.backend.paths import DATA_DIR, EVAL_RESULTS_DIR

from dotenv import load_dotenv
load_dotenv()


def load_datasets(index: str):
    """Load RSS context and generated data for a given dataset index."""
    context_data_path = os.path.join(DATA_DIR, "rss_context_dataset.json")
    with open(context_data_path, "r", encoding="utf-8") as f:
        rss_data = json.load(f)["rss_context"]

    generated_data_path = os.path.join(DATA_DIR, f"generated_dataset_{index}.json")
    with open(generated_data_path, "r", encoding="utf-8") as f:
        generated = json.load(f)["generated_news"]

    return rss_data, generated


def save_result(index: str, test_name: str, result: dict):
    """Save individual test result to a consolidated JSON file."""
    if not os.path.exists(EVAL_RESULTS_DIR):
        os.makedirs(EVAL_RESULTS_DIR, exist_ok=True)

    filepath = os.path.join(EVAL_RESULTS_DIR, "test_news.json")

    # Load existing results if file exists
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                all_results = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            all_results = {}
    else:
        all_results = {}

    # Create nested structure: test_name -> dataset_index -> result
    if test_name not in all_results:
        all_results[test_name] = {}

    all_results[test_name][index] = result

    # Save updated results
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)


def generate_final_summary():
    """Generate a comprehensive markdown summary from all test results."""
    filepath = os.path.join(EVAL_RESULTS_DIR, "test_news.json")
    summary_path = os.path.join(EVAL_RESULTS_DIR, "test_news_summary.md")

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
        md_file.write("# Test News Summary Report\n\n")
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

            if test_name == "data_structure_news":
                md_file.write("| Dataset | Generated Count | Passed |\n")
                md_file.write("|---------|-----------------|--------|\n")
                for dataset_idx in sorted(test_data.keys()):
                    result = test_data[dataset_idx]
                    passed = result.get("passed", False)
                    md_file.write(f"| {dataset_idx} | {result.get('generated_count', 0)} | {'✅' if passed else '❌'} |\n")
                md_file.write("\n")

            elif test_name == "news_relevance_to_topic":
                for dataset_idx in sorted(test_data.keys()):
                    result = test_data[dataset_idx]
                    passed = result.get("passed", False)
                    md_file.write(f"### Dataset {dataset_idx} {'✅' if passed else '❌'}\n\n")
                    for test_result in result.get("structured_results", []):
                        md_file.write(f"**Test**: {test_result.get('test_name', 'Unknown')}\n\n")
                        for metric in test_result.get('metrics', []):
                            md_file.write(f"- **{metric.get('name')}** ({'✅' if metric.get('success') else '❌'})\n")
                            md_file.write(f"  - Score: {metric.get('score', 0):.4f} / Threshold: {metric.get('threshold', 0):.4f}\n")
                            md_file.write(f"  - Reason: {metric.get('reason', '')[:200]}...\n")
                    md_file.write("\n---\n\n")

        md_file.write("## Final Summary\n\n")
        total_tests = sum(len(test_data) for test_data in all_results.values())
        passed_tests = sum(
            1
            for test_name, test_data in all_results.items()
            for result in test_data.values()
            if result.get("passed", False)
        )
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
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


@pytest.fixture(scope="session", autouse=True)
def generate_summary_after_tests():
    """Generate summary report after all tests complete."""
    yield
    generate_final_summary()

# Rule-based test: Check if the data structure of generated news matches the expected format
@pytest.mark.parametrize("dataset_index", ["0", "1", "2"])
def test_data_structure_news(dataset_index):
    _, generated = load_datasets(dataset_index)
    if not generated:
        pytest.skip(f"Skipping dataset {dataset_index} due to empty data.")

    passed = True
    for idx, item in enumerate(generated):
        if not all(key in item for key in ["topic", "news_articles"]):
            passed = False
            break
        if not isinstance(item["news_articles"], list):
            passed = False
            break
        for news in item["news_articles"]:
            if not all(k in news for k in ["title", "link", "summary", "content"]):
                passed = False
                break

    save_result(dataset_index, "data_structure_news", {
        "dataset_index": dataset_index,
        "generated_count": len(generated),
        "passed": passed,
        "timestamp": datetime.utcnow().isoformat()
    })

    assert passed, f"Data structure test failed for dataset {dataset_index}"


# LLM-as-a-Judge: Evaluate if the generated news articles are relevant to their assigned topic
@pytest.mark.parametrize("dataset_index", ["0", "1", "2"])
def test_news_relevance_to_topic(dataset_index):
    _, generated = load_datasets(dataset_index)
    if not generated:
        pytest.skip(f"Skipping dataset {dataset_index} due to empty data.")

    relevance_metric = GEval(
        name="News Relevance to Topic",
        criteria="Determine if the news articles are semantically relevant to their assigned topic.",
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        threshold=0.7
    )

    structured_results = []
    all_passed = True

    for i, item in enumerate(generated):
        topic = item["topic"]
        news_articles = item.get("news_articles", [])
        for j, news in enumerate(news_articles):
            test_case = LLMTestCase(
                name=f"news_relevance_topic_{i}_{j}",
                input=topic,
                actual_output=f"{news['title']}\n{news.get('summary', '')}\n{news.get('content', '')}"
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
            for tr in test_results:
                structured_results.append({
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
                })
                if not tr.success:
                    all_passed = False

    save_result(dataset_index, "news_relevance_to_topic", {
        "dataset_index": dataset_index,
        "passed": all_passed,
        "timestamp": datetime.utcnow().isoformat(),
        "structured_results": structured_results
    })

    assert all_passed, f"News relevance to topic failed for dataset {dataset_index}"