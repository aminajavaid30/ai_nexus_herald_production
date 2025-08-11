import os
import io
import sys
import json
import pytest
from datetime import datetime

from deepeval import assert_test, evaluate
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from src.backend.agents.orchestrator import Orchestrator, OrchestratorState
from src.backend.paths import DATA_DIR, EVAL_RESULTS_DIR

from dotenv import load_dotenv
load_dotenv()


def save_result(test_name: str, result: dict):
    """Save orchestrator test result to consolidated JSON file."""
    if not os.path.exists(EVAL_RESULTS_DIR):
        os.makedirs(EVAL_RESULTS_DIR, exist_ok=True)
    
    filepath = os.path.join(EVAL_RESULTS_DIR, "test_orchestrator.json")
    
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
    """Generate a markdown summary from orchestrator test results."""
    filepath = os.path.join(EVAL_RESULTS_DIR, "test_orchestrator.json")
    summary_path = os.path.join(EVAL_RESULTS_DIR, "test_orchestrator_summary.md")
    
    if not os.path.exists(filepath):
        print("Warning: No orchestrator test results found.")
        return
    
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            all_results = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        print("Warning: Could not read orchestrator test results.")
        return
    
    with open(summary_path, "w", encoding="utf-8") as md_file:
        md_file.write("# Orchestrator Test Summary Report\n\n")
        md_file.write(f"**Generated on**: {datetime.utcnow().isoformat()}\n\n---\n\n")
        
        total_tests = sum(len(v) for v in all_results.values())
        passed_tests = sum(1 for v in all_results.values() for r in v if r.get("passed", False))
        
        md_file.write(f"- **Total Tests**: {total_tests}\n")
        md_file.write(f"- **Passed**: {passed_tests}\n")
        md_file.write(f"- **Failed**: {total_tests - passed_tests}\n")
        md_file.write(f"- **Success Rate**: {(passed_tests / total_tests * 100) if total_tests else 0:.1f}%\n\n")
        
        for test_name, results in all_results.items():
            md_file.write(f"## {test_name.replace('_', ' ').title()}\n\n")
            for res in results:
                status_emoji = "✅" if res.get("passed", False) else "❌"
                md_file.write(f"### Run at {res.get('timestamp', 'N/A')} {status_emoji}\n\n")
                if "metrics" in res:
                    for m in res["metrics"]:
                        metric_status = "✅" if m.get("success") else "❌"
                        md_file.write(f"- **{m.get('name')}** {metric_status} — Score: {m.get('score', 0):.4f} / Threshold: {m.get('threshold', 0):.4f}\n")
                        if m.get("reason"):
                            md_file.write(f"  - Reason: {m.get('reason')[:200]}...\n")
                md_file.write("\n")
    
    print(f"Summary report generated: {summary_path}")


@pytest.fixture(scope="session", autouse=True)
def generate_summary_after_tests():
    yield
    generate_final_summary()


# Rule-based test: Check for task completion
def test_full_workflow_outputs():
    """Run full orchestrator and validate structure."""
    groq_api = os.getenv("GROQ_API_KEY")
    assert groq_api, "GROQ_API_KEY must be set."

    orchestrator = Orchestrator(groq_api)
    graph = orchestrator.build_orchestrator_graph()

    final_state = graph.invoke(OrchestratorState(), config={"recursion_limit": 200})
    news = final_state["news"]
    newsletter = final_state["newsletter"]

    passed = (
        isinstance(news, list) and news and
        isinstance(newsletter, str) and newsletter.strip()
    )

    save_result("full_workflow_outputs", {
        "timestamp": datetime.utcnow().isoformat(),
        "passed": passed,
        "news_count": len(news) if isinstance(news, list) else 0,
        "newsletter_length": len(newsletter) if isinstance(newsletter, str) else 0
    })

    assert passed, "Orchestrator failed to produce valid outputs."


# LLM-as-a-Judge: Test semantic consistency between topics, articles, and newsletter
def test_orchestrator_semantic_consistency():
    """Check semantic consistency between topics, articles, and newsletter."""
    groq_api = os.getenv("GROQ_API_KEY")
    assert groq_api

    orchestrator = Orchestrator(groq_api)
    graph = orchestrator.build_orchestrator_graph()
    final = graph.invoke(OrchestratorState(), config={"recursion_limit": 200})

    news = final["news"]
    newsletter = final["newsletter"]

    topic_metric = AnswerRelevancyMetric(threshold=0.5)
    rel_metric = AnswerRelevancyMetric(threshold=0.5)
    faith_metric = FaithfulnessMetric(threshold=0.5)

    all_metrics_results = []

    # Articles vs Topic
    for entry in news:
        topic = entry.topic
        for art in entry.news_articles:
            content = f"{art.title}\n{art.summary}\n{art.content}"
            case = LLMTestCase(
                name=f"article_relevancy_for_{topic}",
                input=f"Check relevance of news to topic: {topic}",
                actual_output=content,
                retrieval_context=[topic]
            )
            buffer = io.StringIO()
            sys_stdout = sys.stdout
            sys.stdout = buffer
            try:
                results = evaluate([case], [topic_metric])
            finally:
                sys.stdout = sys_stdout
            
            for tr in results.test_results:
                for md in tr.metrics_data:
                    all_metrics_results.append({
                        "name": md.name,
                        "score": md.score,
                        "threshold": md.threshold,
                        "success": md.success,
                        "reason": md.reason,
                        "cost": md.evaluation_cost
                    })

    # Newsletter vs Aggregated News
    aggregated_text = []
    for entry in news:
        for art in entry.news_articles:
            aggregated_text.append(f"{art.title}\n{art.summary}\n{art.content}")

    case = LLMTestCase(
        name="newsletter_end_to_end_relevancy",
        input="Check newsletter alignment with all generated news",
        actual_output=newsletter,
        retrieval_context=aggregated_text
    )
    buffer = io.StringIO()
    sys_stdout = sys.stdout
    sys.stdout = buffer
    try:
        results = evaluate([case], [rel_metric, faith_metric])
    finally:
        sys.stdout = sys_stdout

    for tr in results.test_results:
        for md in tr.metrics_data:
            all_metrics_results.append({
                "name": md.name,
                "score": md.score,
                "threshold": md.threshold,
                "success": md.success,
                "reason": md.reason,
                "cost": md.evaluation_cost
            })

    passed = all(m["success"] for m in all_metrics_results)

    save_result("semantic_consistency", {
        "timestamp": datetime.utcnow().isoformat(),
        "passed": passed,
        "metrics": all_metrics_results
    })

    assert passed, "Semantic consistency test failed."