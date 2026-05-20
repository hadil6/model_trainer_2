from abc import ABC, abstractmethod
from typing import Any


class BaseProfiler(ABC):
    @abstractmethod
    def profile(self, filename: str, content: bytes) -> dict[str, Any]:
        raise NotImplementedError
