import os
import io
import sys
import json
import pytest
from datetime import datetime

from deepeval import assert_test
from deepeval import evaluate
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
    
    filepath = os.path.join(EVAL_RESULTS_DIR, "test_topics.json")
    
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
    filepath = os.path.join(EVAL_RESULTS_DIR, "test_topics.json")
    summary_path = os.path.join(EVAL_RESULTS_DIR, "test_topics_summary.md")
    
    if not os.path.exists(filepath):
        print("Warning: No test results file found, skipping summary generation.")
        return
    
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            all_results = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        print("Warning: Could not read test results file, skipping summary generation.")
        return
    
    # Generate markdown summary
    with open(summary_path, "w", encoding="utf-8") as md_file:
        md_file.write("# Test Topics Summary Report\n\n")
        md_file.write(f"**Generated on**: {datetime.utcnow().isoformat()}\n\n")
        md_file.write("---\n\n")
        
        # Summary of all datasets tested
        datasets_tested = set()
        for test_name, test_data in all_results.items():
            datasets_tested.update(test_data.keys())
        
        md_file.write("## Overview\n\n")
        md_file.write(f"- **Total Test Functions**: {len(all_results)}\n")
        md_file.write(f"- **Datasets Tested**: {sorted(datasets_tested)}\n")
        md_file.write(f"- **Total Test Cases**: {sum(len(test_data) for test_data in all_results.values())}\n\n")
        
        # Process each test function
        for test_name, test_data in all_results.items():
            md_file.write(f"## {test_name.replace('_', ' ').title()}\n\n")
            
            if test_name == "basic_data_loading":
                md_file.write("### Data Loading Results\n\n")
                md_file.write("| Dataset | Gold Titles | Generated Topics | Status |\n")
                md_file.write("|---------|-------------|------------------|--------|\n")
                
                for dataset_idx in sorted(test_data.keys()):
                    result = test_data[dataset_idx]
                    gold_count = result.get('gold_titles_count', 0)
                    gen_count = result.get('generated_topics_count', 0)
                    status = "✅" if gold_count > 0 and gen_count > 0 else "❌"
                    md_file.write(f"| {dataset_idx} | {gold_count} | {gen_count} | {status} |\n")
                
                md_file.write("\n")
            
            elif test_name == "data_structure":
                md_file.write("### Data Structure Validation\n\n")
                md_file.write("| Dataset | RSS Type | RSS Count | Generated Type | Generated Count |\n")
                md_file.write("|---------|----------|-----------|----------------|------------------|\n")
                
                for dataset_idx in sorted(test_data.keys()):
                    result = test_data[dataset_idx]
                    md_file.write(f"| {dataset_idx} | {result.get('rss_type', 'N/A')} | {result.get('rss_count', 0)} | {result.get('generated_type', 'N/A')} | {result.get('generated_count', 0)} |\n")
                
                md_file.write("\n")
            
            elif test_name == "news_topic_relevance":
                md_file.write("### Topic Relevance Evaluation\n\n")
                
                for dataset_idx in sorted(test_data.keys()):
                    result = test_data[dataset_idx]
                    passed = result.get('passed', False)
                    status_emoji = "✅" if passed else "❌"
                    
                    md_file.write(f"#### Dataset {dataset_idx} {status_emoji}\n\n")
                    
                    # Extract metrics data
                    structured_results = result.get('structured_results', [])
                    if structured_results:
                        for test_result in structured_results:
                            md_file.write(f"**Test**: {test_result.get('test_name', 'Unknown')}\n\n")
                            
                            for metric in test_result.get('metrics', []):
                                metric_status = "✅" if metric.get('success', False) else "❌"
                                md_file.write(f"**{metric.get('name', 'Unknown Metric')}** {metric_status}\n")
                                md_file.write(f"- **Score**: {metric.get('score', 0):.4f}\n")
                                md_file.write(f"- **Threshold**: {metric.get('threshold', 0):.4f}\n")
                                md_file.write(f"- **Cost**: ${metric.get('cost', 0):.4f}\n")
                                
                                reason = metric.get('reason', 'No reason provided')
                                if len(reason) > 200:
                                    reason = reason[:200] + "..."
                                md_file.write(f"- **Reason**: {reason}\n\n")
                    
                    md_file.write("---\n\n")
        
        # Final summary statistics
        md_file.write("## Final Summary\n\n")
        
        # Count total passed/failed tests
        total_tests = 0
        passed_tests = 0
        
        for test_name, test_data in all_results.items():
            for dataset_idx, result in test_data.items():
                total_tests += 1
                if test_name == "news_topic_relevance":
                    if result.get('passed', False):
                        passed_tests += 1
                elif test_name in ["basic_data_loading", "data_structure"]:
                    # These are considered passed if they have data
                    if test_name == "basic_data_loading":
                        if result.get('gold_titles_count', 0) > 0 and result.get('generated_topics_count', 0) > 0:
                            passed_tests += 1
                    elif test_name == "data_structure":
                        if result.get('rss_count', 0) > 0 and result.get('generated_count', 0) > 0:
                            passed_tests += 1
        
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


# Add a session-scoped fixture to generate summary after all tests complete
@pytest.fixture(scope="session", autouse=True)
def generate_summary_after_tests():
    """Generate summary report after all tests complete."""
    # This runs before tests
    yield
    # This runs after all tests complete
    generate_final_summary()


# Rule-based test: Ensure datasets are not empty
@pytest.mark.parametrize("dataset_index", ["0", "1", "2"])
def test_basic_data_loading(dataset_index):
    rss_data, generated = load_datasets(dataset_index)
    if not rss_data or not generated:
        pytest.skip(f"Skipping dataset {dataset_index} due to empty data.")

    gold_titles = [item["title"] for item in rss_data]
    generated_topics = [news["topic"] for news in generated]

    result = {
        "dataset_index": dataset_index,
        "gold_titles_count": len(gold_titles),
        "generated_topics_count": len(generated_topics),
        "timestamp": datetime.utcnow().isoformat()
    }

    save_result(dataset_index, "basic_data_loading", result)

    assert len(gold_titles) > 0, "Gold titles dataset is empty."
    assert len(generated_topics) > 0, "Generated topics dataset is empty."


# Rule-based test: Ensure data structure is correct
@pytest.mark.parametrize("dataset_index", ["0", "1", "2"])
def test_data_structure(dataset_index):
    rss_data, generated = load_datasets(dataset_index)
    if not rss_data or not generated:
        pytest.skip(f"Skipping dataset {dataset_index} due to empty data.")

    save_result(dataset_index, "data_structure", {
        "dataset_index": dataset_index,
        "rss_count": len(rss_data),
        "rss_type": type(rss_data).__name__,
        "generated_count": len(generated),
        "generated_type": type(generated).__name__,
        "timestamp": datetime.utcnow().isoformat()
    })

    # Top-level checks
    assert isinstance(rss_data, list), "rss_data must be a list"
    assert isinstance(generated, list), "generated must be a list"

    # Element structure
    for idx, item in enumerate(rss_data):
        assert isinstance(item, dict), f"rss_data[{idx}] must be a dict"
        assert "title" in item, f"rss_data[{idx}] missing 'title' key"

    for idx, news in enumerate(generated):
        assert isinstance(news, dict), f"generated[{idx}] must be a dict"
        assert "topic" in news, f"generated[{idx}] missing 'topic' key"


# LLM-as-a-Judge test: Evaluate news topic relevance to RSS titles
@pytest.mark.parametrize("dataset_index", ["0", "1", "2"])
def test_news_topic_relevance(dataset_index):
    rss_data, generated = load_datasets(dataset_index)
    if not rss_data or not generated:
        pytest.skip(f"Skipping dataset {dataset_index} due to empty data.")

    gold_titles = [item["title"] for item in rss_data]
    topics = [news["topic"] for news in generated]

    relevance_metric = GEval(
        name="Topic Relevance",
        criteria="Determine if the generated news topics are relevant to the titles of the RSS feeds.",
        evaluation_params=[
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.RETRIEVAL_CONTEXT
        ],
        threshold=0.7
    )

    test_case = LLMTestCase(
        name=f"topic_relevance_test_{dataset_index}",
        input="Select the top 5 trending news topics based on the RSS feed titles.",
        actual_output="\n".join(topics),
        retrieval_context=gold_titles
    )

    # Capture DeepEval table output
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

    # Enhanced save with structured data
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

    save_result(dataset_index, "news_topic_relevance", result_data)

    # Print summary
    for test_result in results.test_results:
        for metric_data in test_result.metrics_data:
            status = "PASSED" if metric_data.success else "FAILED"
            print(f"\nSUMMARY: {metric_data.name} - {status}")
            print(f"Score: {metric_data.score:.4f} (Threshold: {metric_data.threshold})")
            print(f"Reason: {metric_data.reason[:100]}...")

    # Fail pytest if needed
    assert passed, f"Topic relevance test failed for dataset {dataset_index}"