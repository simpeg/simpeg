"""
Tests for BaseSurvey class.
"""

import pytest
import numpy as np

from simpeg.utils import Counter
from simpeg.survey import BaseSurvey, BaseRx, BaseSrc


class TestCounterValidation:

    @pytest.fixture
    def sample_source(self):
        locations = np.array([1.0, 2.0, 3.0])
        receiver = BaseRx(locations=locations)
        source = BaseSrc(receiver_list=[receiver])
        return source

    def test_valid_counter(self, sample_source):
        """No error should be raise after passing a valid Counter object to Survey."""
        counter = Counter()
        BaseSurvey(source_list=[sample_source], counter=counter)

    def test_invalid_counter(self, sample_source):
        """Test error upon invalid Counter."""

        class InvalidCounter:
            pass

        invalid_counter = InvalidCounter()
        with pytest.raises(TypeError):
            BaseSurvey(source_list=[sample_source], counter=invalid_counter)
