"""与 README“Instrumentation”章节对应的计时工具。"""

from __future__ import annotations

from time import perf_counter


class Timer:
    """用于度量代码片段的简单上下文管理器。"""

    def __init__(self, label: str | None = None) -> None:
        self.label = label or "timer"
        self._start: float | None = None
        self.elapsed: float | None = None

    def __enter__(self):
        """启动计时器（详见 README “Instrumentation”）。"""

        self._start = perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        """根据 README 的指导停止计时并记录耗时。

        Args:
            exc_type: 若上下文中抛出异常，这里为异常类型。
            exc: 异常实例。
            tb: Traceback 对象。
        """

        if self._start is None:
            return False
        self.elapsed = perf_counter() - self._start
        return False
