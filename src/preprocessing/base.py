"""
Base classes for modular preprocessing pipeline.

Allows composing multiple preprocessing steps in a pipeline.
"""

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray


class PreprocessingStep(ABC):
    """Abstract base class for preprocessing steps."""

    @abstractmethod
    def process(self, image: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """
        Process an image.

        Args:
            image: Input image in BGR format (OpenCV convention)

        Returns:
            Processed image in BGR format
        """

        pass  # pylint: disable=unnecessary-pass

    @abstractmethod
    def get_name(self) -> str:
        """Get the name of this preprocessing step."""

        pass  # pylint: disable=unnecessary-pass


class PreprocessingPipeline:
    """Pipeline for composing multiple preprocessing steps."""

    def __init__(self, steps: list[PreprocessingStep] = None):
        """
        Initialize preprocessing pipeline.

        Args:
            steps: List of preprocessing steps to apply in order
        """
        self.steps = steps or []

    def add_step(self, step: PreprocessingStep) -> None:
        """
        Add a preprocessing step to the pipeline.

        Args:
            step: Preprocessing step to add
        """
        self.steps.append(step)

    def process(self, image: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """
        Apply all preprocessing steps to an image.

        Args:
            image: Input image

        Returns:
            Processed image
        """
        processed = image.copy()

        for step in self.steps:
            processed = step.process(processed)

        return processed

    def get_step_names(self) -> list[str]:
        """Get names of all steps in the pipeline."""
        return [step.get_name() for step in self.steps]

    def __len__(self) -> int:
        """Get number of steps in pipeline."""
        return len(self.steps)

    def __repr__(self) -> str:
        """String representation of pipeline."""
        step_names = " → ".join(self.get_step_names()) if self.steps else "Empty"
        return f"PreprocessingPipeline({step_names})"
