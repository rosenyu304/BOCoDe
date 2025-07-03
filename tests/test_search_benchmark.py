import pytest
import math

from bocode.base import BenchmarkProblem, DataType
from bocode.search_benchmarks import (
    filter_functions,
    categorized_classes,
    _has_valid_val,
)


class TestSearchBenchmarks:
    """Test suite for search_benchmarks.py functionality."""

    def test_categorized_classes_structure(self):
        """Test that categorized_classes has the expected structure."""
        # Test that all expected categories are present
        expected_categories = [
            "Synthetics",
            "LassoBench",
            "Engineering",
            "Engineering.Gym",
            "Engineering.BayesianCHT",
            "CEC.CEC2020_RW_Constrained",
            "BBOB",
            "BoTorch",
            "MODAct",
            "CEC.CEC2017",
            "WFG",
            "ZDT",
            "DTLZ",
            "CEC.CEC2007",
            "CEC.CEC2019",
            "NEORL",
        ]

        for category in expected_categories:
            assert category in categorized_classes, f"Category {category} not found"
            assert len(categorized_classes[category]) > 0, (
                f"Category {category} is empty"
            )

    def test_has_valid_val_with_int(self):
        """Test _has_valid_val with integer values."""

        def constraint(x):
            return x >= 5

        assert _has_valid_val(10, constraint) is True
        assert _has_valid_val(5, constraint) is True
        assert _has_valid_val(3, constraint) is False

    def test_has_valid_val_with_list(self):
        """Test _has_valid_val with list values."""

        def constraint(x):
            return x >= 5

        assert _has_valid_val([1, 2, 3], constraint) is False
        assert _has_valid_val([1, 2, 10], constraint) is True
        assert _has_valid_val([5, 6, 7], constraint) is True

    def test_has_valid_val_with_set(self):
        """Test _has_valid_val with set values."""

        def constraint(x):
            return x >= 5

        assert _has_valid_val({1, 2, 3}, constraint) is False
        assert _has_valid_val({1, 2, 10}, constraint) is True
        assert _has_valid_val({5, 6, 7}, constraint) is True

    def test_has_valid_val_with_tuple_closed_interval(self):
        """Test _has_valid_val with closed interval tuples."""

        def constraint(x):
            return x >= 5

        assert _has_valid_val((1, 4), constraint) is False
        assert _has_valid_val((1, 5), constraint) is True
        assert _has_valid_val((5, 10), constraint) is True
        assert _has_valid_val((6, 10), constraint) is True

    def test_has_valid_val_with_tuple_open_interval(self):
        """Test _has_valid_val with open interval tuples."""

        def constraint(x):
            return x >= 5

        # Test with None (open interval)
        assert _has_valid_val((1, None), constraint) is True
        assert _has_valid_val((10, None), constraint) is True

        # Test with infinity (open interval)
        assert _has_valid_val((1, math.inf), constraint) is True
        assert _has_valid_val((10, math.inf), constraint) is True

        # Test case where constraint is never satisfied within sample limit
        def impossible_constraint(x):
            return x > 1000

        assert _has_valid_val((1, None), impossible_constraint) is False

    def test_has_valid_val_invalid_type(self):
        """Test _has_valid_val with invalid types."""

        def constraint(x):
            return x >= 5

        with pytest.raises(ValueError, match="Unsupported val type"):
            _has_valid_val("invalid", constraint)

        with pytest.raises(ValueError, match="Unsupported val type"):
            _has_valid_val(3.14, constraint)

    def test_filter_functions_no_filters(self):
        """Test filter_functions with no filters (should return all functions)."""
        result = filter_functions()

        # Should return all categories
        assert len(result) > 0

        # Check that result contains expected structure
        for category, functions in result.items():
            assert isinstance(functions, list)
            assert len(functions) > 0

            # Check that function names have correct format
            for func in functions:
                assert issubclass(func, BenchmarkProblem)

    def test_filter_functions_dimension_filter(self):
        """Test filter_functions with dimension filtering."""
        # Filter for functions with dimension <= 10
        result = filter_functions(dimension_filter=lambda x: x <= 10)

        # Should have some results
        assert len(result) > 0

        # Filter for functions with dimension > 1000 (should be very few or none)
        result_high_dim = filter_functions(dimension_filter=lambda x: x > 1000)

        # Should have fewer results than unfiltered
        total_unfiltered = sum(len(funcs) for funcs in filter_functions().values())
        total_high_dim = sum(len(funcs) for funcs in result_high_dim.values())
        assert total_high_dim < total_unfiltered

    def test_filter_functions_input_type_filter(self):
        """Test filter_functions with input type filtering."""
        # Filter for continuous functions only
        result = filter_functions(input_type_filter=lambda x: x == DataType.CONTINUOUS)

        # Should have some results
        assert len(result) > 0

        # Filter for discrete functions only
        result_discrete = filter_functions(
            input_type_filter=lambda x: x == DataType.DISCRETE
        )

        # Results should be different
        continuous_total = sum(len(funcs) for funcs in result.values())
        discrete_total = sum(len(funcs) for funcs in result_discrete.values())

        # At least one should have results
        assert continuous_total > 0 or discrete_total > 0

    def test_filter_functions_objectives_filter(self):
        """Test filter_functions with objectives filtering."""
        # Filter for single-objective functions
        result_single = filter_functions(objectives_filter=lambda x: x == 1)

        # Filter for multi-objective functions
        result_multi = filter_functions(objectives_filter=lambda x: x > 1)

        # Should have some results
        assert len(result_single) > 0 or len(result_multi) > 0

        # Total should be less than or equal to unfiltered
        total_unfiltered = sum(len(funcs) for funcs in filter_functions().values())
        total_single = sum(len(funcs) for funcs in result_single.values())
        total_multi = sum(len(funcs) for funcs in result_multi.values())

        assert total_single <= total_unfiltered
        assert total_multi <= total_unfiltered

    def test_filter_functions_constraints_filter(self):
        """Test filter_functions with constraints filtering."""
        # Filter for unconstrained functions
        result_unconstrained = filter_functions(constraints_filter=lambda x: x == 0)

        # Filter for constrained functions
        result_constrained = filter_functions(constraints_filter=lambda x: x > 0)

        # Should have some results
        assert len(result_unconstrained) > 0 or len(result_constrained) > 0

        # Total should be less than or equal to unfiltered
        total_unfiltered = sum(len(funcs) for funcs in filter_functions().values())
        total_unconstrained = sum(len(funcs) for funcs in result_unconstrained.values())
        total_constrained = sum(len(funcs) for funcs in result_constrained.values())

        assert total_unconstrained <= total_unfiltered
        assert total_constrained <= total_unfiltered

    def test_filter_functions_category_filter(self):
        """Test filter_functions with category filtering."""
        # Filter for specific categories
        result_synthetics = filter_functions(
            category_filter=lambda x: x == "Synthetics"
        )

        # Should only contain Synthetics category
        assert "Synthetics" in result_synthetics
        assert len(result_synthetics) == 1

        # Filter for CEC categories
        result_cec = filter_functions(category_filter=lambda x: x.startswith("CEC"))

        # Should only contain CEC categories
        for category in result_cec.keys():
            assert category.startswith("CEC")

    def test_filter_functions_combined_filters(self):
        """Test filter_functions with multiple filters combined."""
        # Combine dimension and objectives filters
        result = filter_functions(
            dimension_filter=lambda x: x <= 20,
            objectives_filter=lambda x: x == 1,
            category_filter=lambda x: x in ["Synthetics", "BBOB"],
        )

        # Should have some results
        total_results = sum(len(funcs) for funcs in result.values())

        # Should only contain specified categories
        for category in result.keys():
            assert category in ["Synthetics", "BBOB"]

        # Should have fewer results than unfiltered
        total_unfiltered = sum(len(funcs) for funcs in filter_functions().values())
        assert total_results <= total_unfiltered

    def test_filter_functions_empty_results(self):
        """Test filter_functions with filters that return no results."""
        # Use impossible constraint
        result = filter_functions(dimension_filter=lambda x: x < 0)

        # Should return empty dict or dict with empty lists
        total_results = sum(len(funcs) for funcs in result.values())
        assert total_results == 0

    def test_filter_functions_return_type(self):
        """Test that filter_functions returns the correct type."""
        result = filter_functions()

        # Should return a dictionary
        assert isinstance(result, dict)

        # Values should be lists of strings
        for category, functions in result.items():
            assert isinstance(category, str)
            assert isinstance(functions, list)
            for func in functions:
                assert issubclass(func, BenchmarkProblem)

    def test_specific_benchmark_categories(self):
        """Test that specific benchmark categories contain expected functions."""
        # Test CEC2020 functions
        result_cec2020 = filter_functions(
            category_filter=lambda x: x == "CEC.CEC2020_RW_Constrained"
        )

        if "CEC.CEC2020_RW_Constrained" in result_cec2020:
            cec2020_funcs = result_cec2020["CEC.CEC2020_RW_Constrained"]
            # Should have 57 CEC2020 functions
            assert len(cec2020_funcs) == 57

            cec2020_func_names = [func.__name__ for func in cec2020_funcs]

            # Check that some expected functions are present
            expected_funcs = [
                "CEC2020_p1",
                "CEC2020_p57",
            ]
            for expected in expected_funcs:
                assert expected in cec2020_func_names

    def test_filter_functions_edge_cases(self):
        """Test filter_functions with edge cases."""
        # Test with lambda that always returns True
        result_all_true = filter_functions(
            dimension_filter=lambda x: True,
            input_type_filter=lambda x: True,
            objectives_filter=lambda x: True,
            constraints_filter=lambda x: True,
            category_filter=lambda x: True,
        )

        # Should be equivalent to no filters
        result_no_filters = filter_functions()

        # Should have same number of total functions
        total_true = sum(len(funcs) for funcs in result_all_true.values())
        total_no_filters = sum(len(funcs) for funcs in result_no_filters.values())
        assert total_true == total_no_filters

    def test_docstring_examples(self):
        """Test examples from the docstring."""
        # Test filtering for continuous functions with dimension <= 10
        result = filter_functions(
            dimension_filter=lambda x: x <= 10,
            input_type_filter=lambda x: x == DataType.CONTINUOUS,
        )

        # Should have some results
        assert len(result) > 0

        # Test filtering for multi-objective functions
        result_multi = filter_functions(objectives_filter=lambda x: x > 1)

        # Should have some results
        total_multi = sum(len(funcs) for funcs in result_multi.values())
        assert total_multi > 0

    def test_specific_data_types(self):
        """Test filtering with specific data types."""
        # Test continuous functions
        continuous_result = filter_functions(
            input_type_filter=lambda x: x == DataType.CONTINUOUS
        )
        continuous_total = sum(len(funcs) for funcs in continuous_result.values())

        # Test discrete functions
        discrete_result = filter_functions(
            input_type_filter=lambda x: x == DataType.DISCRETE
        )
        discrete_total = sum(len(funcs) for funcs in discrete_result.values())

        # Test mixed functions
        mixed_result = filter_functions(input_type_filter=lambda x: x == DataType.MIXED)
        mixed_total = sum(len(funcs) for funcs in mixed_result.values())

        # Should have some continuous functions (most benchmarks are continuous)
        assert continuous_total > 0

        # Total should not exceed unfiltered total
        unfiltered_total = sum(len(funcs) for funcs in filter_functions().values())
        assert continuous_total <= unfiltered_total
        assert discrete_total <= unfiltered_total
        assert mixed_total <= unfiltered_total

    def test_nested_category_filtering(self):
        """Test filtering with nested categories."""
        # Filter for all Engineering categories
        engineering_result = filter_functions(
            category_filter=lambda x: x.startswith("Engineering")
        )

        # Should contain multiple Engineering subcategories
        engineering_categories = [
            cat for cat in engineering_result.keys() if cat.startswith("Engineering")
        ]
        assert len(engineering_categories) > 1

        # Test specific nested category
        gym_result = filter_functions(category_filter=lambda x: x == "Engineering.Gym")

        assert len(gym_result) == 1  # Should only contain this category
        assert "Engineering.Gym" in gym_result

    def test_function_count_consistency(self):
        """Test that function counts are consistent across different filters."""
        # Get all functions
        all_functions = filter_functions()
        total_all = sum(len(funcs) for funcs in all_functions.values())

        # Test that filtering by always-true conditions gives same result
        same_result = filter_functions(
            dimension_filter=lambda x: x >= 0,  # All dimensions are >= 0
            objectives_filter=lambda x: x >= 1,  # All objectives are >= 1
            constraints_filter=lambda x: x >= 0,  # All constraints are >= 0
        )
        total_same = sum(len(funcs) for funcs in same_result.values())

        # Should be equal
        assert total_all == total_same

    def test_empty_filter_results(self):
        """Test filters that should return empty results."""
        # Impossible dimension constraint
        impossible_dim = filter_functions(dimension_filter=lambda x: x < 0)
        total_impossible = sum(len(funcs) for funcs in impossible_dim.values())
        assert total_impossible == 0

        # Impossible objectives constraint
        impossible_obj = filter_functions(objectives_filter=lambda x: x < 1)
        total_impossible_obj = sum(len(funcs) for funcs in impossible_obj.values())
        assert total_impossible_obj == 0

        # Impossible constraints constraint
        impossible_constr = filter_functions(constraints_filter=lambda x: x < 0)
        total_impossible_constr = sum(
            len(funcs) for funcs in impossible_constr.values()
        )
        assert total_impossible_constr == 0

    def test_performance_with_large_filters(self):
        """Test that filtering performs reasonably with complex conditions."""
        import time

        # Complex filter that should still execute quickly
        start_time = time.time()

        complex_result = filter_functions(
            dimension_filter=lambda x: x <= 100 and x >= 1,
            input_type_filter=lambda x: x in [DataType.CONTINUOUS, DataType.DISCRETE],
            objectives_filter=lambda x: x <= 10,
            constraints_filter=lambda x: x <= 50,
            category_filter=lambda x: len(x) > 3,  # Non-empty category names
        )

        end_time = time.time()
        execution_time = end_time - start_time

        # Should complete within reasonable time (1 second should be more than enough)
        assert execution_time < 1.0

        # Should still have some results
        total_complex = sum(len(funcs) for funcs in complex_result.values())
        assert (
            total_complex >= 0
        )  # At least 0 results (could be 0 if filters are too restrictive)

    def test_has_valid_val_edge_cases(self):
        """Test edge cases for _has_valid_val function."""

        # Test with empty collections
        def always_true(x):
            return True

        def always_false(x):
            return False

        # Empty list
        assert _has_valid_val([], always_false) is False
        assert _has_valid_val([], always_true) is False

        # Empty set
        assert _has_valid_val(set(), always_false) is False
        assert _has_valid_val(set(), always_true) is False

        # Single element collections
        assert _has_valid_val([5], lambda x: x == 5) is True
        assert _has_valid_val([5], lambda x: x != 5) is False
        assert _has_valid_val({5}, lambda x: x == 5) is True
        assert _has_valid_val({5}, lambda x: x != 5) is False

        # Zero interval tuple
        assert _has_valid_val((5, 5), lambda x: x == 5) is True
        assert _has_valid_val((5, 5), lambda x: x != 5) is False

        # Reversed interval (should still work)
        assert _has_valid_val((5, 3), lambda x: x == 4) is False  # No values in range

    def test_categorized_classes_content(self):
        """Test that categorized_classes contains expected content."""
        # Test that each category has the expected types
        for category, functions in categorized_classes.items():
            assert isinstance(category, str)
            assert isinstance(functions, list)
            assert len(functions) > 0

            # Each function should be a class/type
            for func in functions:
                assert hasattr(func, "__name__")  # Should have a name attribute
                # Should be callable (class/function)
                assert callable(func)

    def test_real_world_filtering_scenarios(self):
        """Test realistic filtering scenarios that users might need."""
        # Scenario 1: Find low-dimensional, single-objective, unconstrained functions
        easy_functions = filter_functions(
            dimension_filter=lambda x: x <= 5,
            objectives_filter=lambda x: x == 1,
            constraints_filter=lambda x: x == 0,
        )

        total_easy = sum(len(funcs) for funcs in easy_functions.values())
        assert total_easy > 0  # Should find some functions

        # Scenario 2: Find high-dimensional optimization problems
        high_dim_functions = filter_functions(dimension_filter=lambda x: x >= 50)

        total_high_dim = sum(len(funcs) for funcs in high_dim_functions.values())
        # May or may not have results, but shouldn't crash
        assert total_high_dim >= 0

        # Scenario 3: Find multi-objective, constrained problems
        complex_functions = filter_functions(
            objectives_filter=lambda x: x > 1, constraints_filter=lambda x: x > 0
        )

        total_complex = sum(len(funcs) for funcs in complex_functions.values())
        assert total_complex >= 0  # Should not crash

        # Scenario 4: Find specific benchmark suite
        cec_functions = filter_functions(category_filter=lambda x: "CEC" in x)

        # Should find CEC functions
        cec_categories = [cat for cat in cec_functions.keys() if "CEC" in cat]
        assert len(cec_categories) > 0
