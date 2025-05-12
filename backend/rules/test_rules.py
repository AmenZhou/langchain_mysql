from typing import List, Dict, Optional, Any
from enum import Enum
import pytest
from dataclasses import dataclass
import re

class TestRuleType(Enum):
    NAMING = "naming"
    COVERAGE = "coverage"
    ASYNC = "async"
    MOCK = "mock"
    ASSERTION = "assertion"

@dataclass
class TestValidationResult:
    is_valid: bool
    rule_type: TestRuleType
    message: str

class TestRule:
    def __init__(self, rule_type: TestRuleType, description: str):
        self.rule_type = rule_type
        self.description = description
        self.enabled = True

    def validate(self, test_item: Any) -> TestValidationResult:
        raise NotImplementedError("Each rule must implement validate method")

class TestNamingRule(TestRule):
    def __init__(self):
        super().__init__(
            TestRuleType.NAMING,
            "Enforces test naming conventions"
        )
        self.patterns = {
            "file": r"^test_.*\.py$",
            "class": r"^Test[A-Z][a-zA-Z0-9]*$",
            "function": r"^test_[a-z][a-z0-9_]*$"
        }

    def validate(self, test_item: str) -> TestValidationResult:
        item_type = self._determine_item_type(test_item)
        pattern = self.patterns.get(item_type)
        if not pattern:
            return TestValidationResult(False, self.rule_type, f"Unknown item type for {test_item}")
        
        matches = bool(re.match(pattern, test_item))
        message = f"{'Valid' if matches else 'Invalid'} {item_type} name: {test_item}"
        return TestValidationResult(matches, self.rule_type, message)

    def _determine_item_type(self, test_item: str) -> str:
        if test_item.endswith(".py"):
            return "file"
        if test_item[0].isupper():
            return "class"
        return "function"

class AsyncTestRule(TestRule):
    def __init__(self):
        super().__init__(
            TestRuleType.ASYNC,
            "Enforces proper async test configuration"
        )

    def validate(self, test_item: Any) -> TestValidationResult:
        # Check if it's an async function
        is_async = hasattr(test_item, "__code__") and test_item.__code__.co_flags & 0x80
        if is_async:
            # Check for pytest.mark.asyncio decorator
            has_asyncio_mark = hasattr(test_item, "pytestmark") and \
                             any(mark.name == "asyncio" for mark in test_item.pytestmark)
            if not has_asyncio_mark:
                return TestValidationResult(
                    False,
                    self.rule_type,
                    f"Async test {test_item.__name__} missing @pytest.mark.asyncio decorator"
                )
        return TestValidationResult(True, self.rule_type, "Valid async test configuration")

class MockRule(TestRule):
    def __init__(self):
        super().__init__(
            TestRuleType.MOCK,
            "Enforces proper mock usage in tests"
        )
        self.required_mocks = {
            "openai": ["AsyncOpenAI", "ChatOpenAI"],
            "database": ["SQLDatabase", "AsyncEngine"],
            "schema": ["SchemaVectorizer"]
        }

    def validate(self, test_module: Any) -> TestValidationResult:
        missing_mocks = []
        for category, mocks in self.required_mocks.items():
            for mock_name in mocks:
                if not self._has_mock(test_module, mock_name):
                    missing_mocks.append(f"{category}.{mock_name}")
        
        if missing_mocks:
            return TestValidationResult(
                False,
                self.rule_type,
                f"Missing required mocks: {', '.join(missing_mocks)}"
            )
        return TestValidationResult(True, self.rule_type, "All required mocks present")

    def _has_mock(self, test_module: Any, mock_name: str) -> bool:
        # Check if mock is defined in fixtures or patch decorators
        return hasattr(test_module, f"mock_{mock_name.lower()}") or \
               any(hasattr(attr, "__name__") and "patch" in attr.__name__ 
                   for attr in getattr(test_module, "__dict__", {}).values())

class AssertionRule(TestRule):
    def __init__(self, min_assertions: int = 1):
        super().__init__(
            TestRuleType.ASSERTION,
            "Enforces minimum number of assertions in tests"
        )
        self.min_assertions = min_assertions

    def validate(self, test_function: Any) -> TestValidationResult:
        assertion_count = self._count_assertions(test_function)
        is_valid = assertion_count >= self.min_assertions
        message = f"Found {assertion_count} assertions (minimum: {self.min_assertions})"
        return TestValidationResult(is_valid, self.rule_type, message)

    def _count_assertions(self, test_function: Any) -> int:
        # Count assert statements in function body
        if not hasattr(test_function, "__code__"):
            return 0
        
        code = test_function.__code__
        count = 0
        for const in code.co_consts:
            if isinstance(const, str) and const.startswith("assert"):
                count += 1
        return count

class TestRuleEngine:
    def __init__(self):
        self.rules: List[TestRule] = []
        self._initialize_default_rules()

    def _initialize_default_rules(self):
        self.rules.extend([
            TestNamingRule(),
            AsyncTestRule(),
            MockRule(),
            AssertionRule(min_assertions=1)
        ])

    def add_rule(self, rule: TestRule):
        self.rules.append(rule)

    def validate_test(self, test_item: Any) -> Dict[str, TestValidationResult]:
        results = {}
        for rule in self.rules:
            if rule.enabled:
                results[rule.rule_type.value] = rule.validate(test_item)
        return results

    def is_test_valid(self, test_item: Any) -> bool:
        return all(result.is_valid for result in self.validate_test(test_item).values())

    def get_validation_messages(self, test_item: Any) -> List[str]:
        return [result.message for result in self.validate_test(test_item).values()] 
