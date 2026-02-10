"""
Test test
"""

import pytest


class TestTest:
    """
    Test test
    """

    def test_test(self) -> None:
        """
        Test test
        """

        a = "test"
        c = "_"
        b = "test"

        assert a + c + b == "test_test"

        with pytest.raises(AssertionError):
            assert a + b == "test_test"
