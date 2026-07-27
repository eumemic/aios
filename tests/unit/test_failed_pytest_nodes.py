from pathlib import Path

from scripts.failed_pytest_nodes import failed_nodes


def test_failed_nodes_converts_junit_classnames(tmp_path: Path) -> None:
    report = tmp_path / "report.xml"
    report.write_text(
        """<testsuites><testsuite>
        <testcase classname="tests.e2e.test_example.TestThing" name="test_bad[param]">
          <failure message="no" />
        </testcase>
        <testcase classname="tests.e2e.test_example" name="test_error">
          <error message="boom" />
        </testcase>
        <testcase classname="tests.e2e.test_example" name="test_ok" />
        </testsuite></testsuites>"""
    )

    assert failed_nodes(report) == [
        "tests/e2e/test_example.py::TestThing::test_bad[param]",
        "tests/e2e/test_example.py::test_error",
    ]
