from typing import List, Dict, Optional
from enum import Enum

class SQLRuleType(Enum):
    PROHIBITED_OPERATIONS = "prohibited_operations"
    TABLE_ACCESS = "table_access"
    QUERY_COMPLEXITY = "query_complexity"
    DATA_SENSITIVITY = "data_sensitivity"

class SQLRule:
    def __init__(self, rule_type: SQLRuleType, description: str):
        self.rule_type = rule_type
        self.description = description
        self.enabled = True

    def validate(self, query: str) -> bool:
        raise NotImplementedError("Each rule must implement validate method")

class ProhibitedOperationsRule(SQLRule):
    def __init__(self):
        super().__init__(
            SQLRuleType.PROHIBITED_OPERATIONS,
            "Prevents dangerous SQL operations"
        )
        self.prohibited_operations = [
            "DROP", "TRUNCATE", "DELETE FROM", "UPDATE",
            "ALTER TABLE", "GRANT", "REVOKE"
        ]

    def validate(self, query: str) -> bool:
        query = query.upper()
        return not any(op in query for op in self.prohibited_operations)

class TableAccessRule(SQLRule):
    def __init__(self, allowed_tables: List[str]):
        super().__init__(
            SQLRuleType.TABLE_ACCESS,
            "Restricts access to specific tables"
        )
        self.allowed_tables = [table.lower() for table in allowed_tables]

    def validate(self, query: str) -> bool:
        # Simple validation - can be enhanced for more complex queries
        query = query.lower()
        tables_in_query = self._extract_tables(query)
        return all(table in self.allowed_tables for table in tables_in_query)

    def _extract_tables(self, query: str) -> List[str]:
        # Simple table extraction - can be enhanced
        words = query.split()
        tables = []
        for i, word in enumerate(words):
            if word == "from" and i + 1 < len(words):
                tables.append(words[i + 1].strip(";"))
            elif word == "join" and i + 1 < len(words):
                tables.append(words[i + 1].strip(";"))
        return tables

class QueryComplexityRule(SQLRule):
    def __init__(self, max_joins: int = 3, max_conditions: int = 5):
        super().__init__(
            SQLRuleType.QUERY_COMPLEXITY,
            "Limits query complexity"
        )
        self.max_joins = max_joins
        self.max_conditions = max_conditions

    def validate(self, query: str) -> bool:
        query = query.upper()
        join_count = query.count("JOIN")
        condition_count = query.count("AND") + query.count("OR")
        return join_count <= self.max_joins and condition_count <= self.max_conditions

class RuleEngine:
    def __init__(self):
        self.rules: List[SQLRule] = []

    def add_rule(self, rule: SQLRule):
        self.rules.append(rule)

    def validate_query(self, query: str) -> Dict[str, bool]:
        results = {}
        for rule in self.rules:
            if rule.enabled:
                results[rule.rule_type.value] = rule.validate(query)
        return results

    def is_query_valid(self, query: str) -> bool:
        return all(self.validate_query(query).values()) 
