"""タスク取得."""

from __future__ import annotations

from typing import Iterable


def get_target_tasks(task_names: Iterable[str]):
    """task_names を JMTEB のタスクオブジェクトのリストに展開.

    get_jmteb_tasks() は MTEBTasks ラッパーを返すため list() で個別タスクに展開.
    JMTEBV2Evaluator.run() は task.metadata.name を参照するので展開が必須.
    """
    from jmteb.v2.tasks import get_jmteb_tasks

    names = list(task_names)
    tasks = list(get_jmteb_tasks(task_names=names))

    # 取りこぼし防止 (typo safety)
    obtained_names = {t.metadata.name for t in tasks}
    missing = [n for n in names if n not in obtained_names]
    assert not missing, f"requested tasks not found in JMTEB: {missing}"

    # Retrieval 限定 (eval-jmteb の慣行に従う)
    bad = [t.metadata.name for t in tasks if str(t.metadata.type) != "Retrieval"]
    assert not bad, f"non-Retrieval tasks present: {bad}"

    return tasks
