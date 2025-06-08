# GitHub Actions CI/CD Setup

## 🚀 Overview

This project uses GitHub Actions for comprehensive testing on every pull request and push to main/develop branches. The CI/CD pipeline ensures code quality, security, and deployment readiness.

## 📋 Workflows

### 1. Test Suite (`.github/workflows/test.yml`)

**Triggers**: PR to main/develop, push to main/develop

**What it does**:
- ✅ Runs all 58 unit and integration tests
- ✅ Tests database connectivity and operations  
- ✅ Validates schema vectorization
- ✅ Checks import paths and dependencies
- ✅ Performs deployment smoke tests

**Services**:
- MySQL 8.0 with test database
- Docker with BuildKit support

### 2. PII Filter Test (`.github/workflows/pii-filter-test.yml`)

**Triggers**: Changes to PII-related files (`utils/`, `prompts.py`, etc.)

**What it does**:
- ✅ Tests PII filter functionality specifically
- ✅ Loads Sakila sample database with customer data
- ✅ Validates that emails, phones, and sensitive data are filtered
- ✅ Comments on PR with PII filter status

**Special Features**:
- Uses real customer data from Sakila database
- Tests both unit and integration levels
- Automatically comments on PRs

## 🔧 Configuration

### Required Secrets

Add these to your GitHub repository secrets:

```bash
# Optional: For real OpenAI API testing
OPENAI_API_KEY=your_openai_api_key_here
```

### Environment Variables

The workflows use these environment variables:

```yaml
DATABASE_URL: mysql+pymysql://test_user:testpassword@localhost:3306/test_db
OPENAI_API_KEY: test_key (or from secrets)
PYTHONPATH: /app
```

## 🏃‍♂️ Running Locally

To run the same tests locally that GitHub Actions runs:

```bash
# Run all tests (same as GitHub Actions)
./scripts/run_tests.sh

# Run with cleanup (same as deployment test)
./scripts/run_tests.sh --cleanup

# Test PII filter specifically
cd backend
python -c "
import asyncio
from utils.sql_utils import sanitize_query_data

async def test():
    data = [{'email': 'test@example.com', 'phone': '555-1234'}]
    result = await sanitize_query_data(data)
    print('Filtered result:', result)

asyncio.run(test())
"
```

## 📊 Test Coverage

The GitHub Actions test suite covers:

| Test Type | Count | Coverage |
|-----------|-------|----------|
| Unit Tests | 40+ | Individual functions |
| Integration Tests | 15+ | Component interactions |
| Deployment Tests | 3+ | Server startup & health |
| Security Tests | 5+ | PII filtering & safety |

## 🚨 Failure Scenarios

### Common Failures and Solutions

**Import Errors**:
- **Cause**: Module path issues between test and production
- **Solution**: Check `PYTHONPATH` and import statements

**Database Connection**:
- **Cause**: MySQL service not ready
- **Solution**: GitHub Actions includes health checks

**PII Filter Failures**:
- **Cause**: OpenAI API issues or filter logic
- **Solution**: Check OpenAI API key and filter implementation

**Docker Build Failures**:
- **Cause**: Dependency issues or Dockerfile problems
- **Solution**: Check requirements.txt and Dockerfile

## 🔄 Workflow Status

### Branch Protection Rules

Recommended branch protection settings:

```yaml
# For main/develop branches
require_status_checks: true
required_status_checks:
  - Test Suite
  - PII Filter Test
require_pull_request_reviews: true
dismiss_stale_reviews: true
require_code_owner_reviews: false
```

### Auto-merge Conditions

PRs can be auto-merged when:
- ✅ All GitHub Actions pass
- ✅ At least 1 approved review
- ✅ No merge conflicts
- ✅ Branch is up to date

## 🎯 Benefits

This CI/CD setup provides:

1. **Early Bug Detection**: Catch issues before they reach production
2. **Security Assurance**: Automatic PII filter validation
3. **Deployment Confidence**: Test actual deployment scenarios
4. **Code Quality**: Consistent testing standards
5. **Team Velocity**: Fast feedback on changes

## 🛠️ Customization

### Adding New Tests

To add a new test workflow:

1. Create `.github/workflows/your-test.yml`
2. Define triggers and jobs
3. Add to branch protection rules
4. Update this documentation

### Modifying Existing Workflows

When modifying workflows:

1. Test changes in a feature branch first
2. Update documentation
3. Verify all team members understand changes
4. Consider backward compatibility

---

**Need Help?** Check the [GitHub Actions documentation](https://docs.github.com/en/actions) or reach out to the team! 🚀 