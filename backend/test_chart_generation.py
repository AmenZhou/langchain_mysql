#!/usr/bin/env python3
"""
Test script for chart generation functionality.
Tests the complete chart generation pipeline with sample data.
"""

import asyncio
import json
import sys
from typing import List, Dict, Any
from charts import ChartOrchestrator, ChartEligibilityAnalyzer, ChartTypeDetector, ChartGenerator

def create_sample_datasets() -> Dict[str, List[Dict[str, Any]]]:
    """Create sample datasets for testing different chart types."""
    
    return {
        "sales_by_category": [
            {"category": "Action", "revenue": 12500, "count": 45},
            {"category": "Comedy", "revenue": 8900, "count": 32},
            {"category": "Drama", "revenue": 15600, "count": 56},
            {"category": "Horror", "revenue": 6700, "count": 24},
            {"category": "Sci-Fi", "revenue": 11200, "count": 38}
        ],
        
        "monthly_trends": [
            {"month": 1, "rentals": 1200, "revenue": 4500},
            {"month": 2, "rentals": 1350, "revenue": 4800},
            {"month": 3, "rentals": 1100, "revenue": 4200},
            {"month": 4, "rentals": 1450, "revenue": 5200},
            {"month": 5, "rentals": 1600, "revenue": 5800},
            {"month": 6, "rentals": 1750, "revenue": 6200}
        ],
        
        "rating_distribution": [
            {"rating": "G", "count": 178},
            {"rating": "PG", "count": 194},
            {"rating": "PG-13", "count": 223},
            {"rating": "R", "count": 195},
            {"rating": "NC-17", "count": 210}
        ],
        
        "pii_filtered_data": [
            {"customer_name": "[PRIVATE]", "email": "[PRIVATE]", "film_title": "ACADEMY DINOSAUR"},
            {"customer_name": "[PRIVATE]", "email": "[PRIVATE]", "film_title": "ACE GOLDFINGER"},
            {"customer_name": "[PRIVATE]", "email": "[PRIVATE]", "film_title": "ADAPTATION HOLES"}
        ],
        
        "insufficient_data": [
            {"item": "single_item", "value": 100}
        ],
        
        "non_chartable_data": [
            {"id": 1, "description": "Some text description"},
            {"id": 2, "description": "Another text description"},
            {"id": 3, "description": "More text content"}
        ]
    }

async def test_eligibility_analyzer():
    """Test the chart eligibility analyzer."""
    print("🔍 Testing Chart Eligibility Analyzer...")
    
    analyzer = ChartEligibilityAnalyzer()
    datasets = create_sample_datasets()
    
    test_cases = [
        ("sales_by_category", True, "Good numeric and categorical data"),
        ("monthly_trends", True, "Time series data"),
        ("rating_distribution", True, "Categorical distribution"),
        ("pii_filtered_data", False, "All meaningful data is private"),
        ("insufficient_data", False, "Insufficient data points"),
        ("non_chartable_data", False, "No chartable data")
    ]
    
    for dataset_name, expected_eligible, description in test_cases:
        data = datasets[dataset_name]
        is_eligible, reason = analyzer.is_eligible(data)
        
        status = "✅" if is_eligible == expected_eligible else "❌"
        print(f"  {status} {dataset_name}: {is_eligible} - {reason}")
        
        if is_eligible != expected_eligible:
            print(f"    Expected: {expected_eligible}, Got: {is_eligible}")
    
    print()

async def test_chart_type_detector():
    """Test the chart type detector."""
    print("📊 Testing Chart Type Detector...")
    
    detector = ChartTypeDetector()
    datasets = create_sample_datasets()
    
    # Test chart detection for eligible datasets
    eligible_datasets = ["sales_by_category", "monthly_trends", "rating_distribution"]
    
    for dataset_name in eligible_datasets:
        data = datasets[dataset_name]
        charts = detector.detect_best_charts(data)
        
        print(f"  📈 {dataset_name}:")
        if charts:
            for i, chart in enumerate(charts[:3], 1):
                confidence_emoji = "🔥" if chart.confidence_score >= 0.8 else "👍" if chart.confidence_score >= 0.6 else "🤔"
                print(f"    {i}. {chart.chart_type.value}: {chart.title} {confidence_emoji} ({chart.confidence_score:.1f})")
        else:
            print(f"    No charts detected")
    
    print()

async def test_chart_generator():
    """Test the chart generator."""
    print("🎨 Testing Chart Generator...")
    
    generator = ChartGenerator()
    detector = ChartTypeDetector()
    datasets = create_sample_datasets()
    
    # Test chart generation for sales data
    data = datasets["sales_by_category"]
    charts = detector.detect_best_charts(data)
    
    if charts:
        chart_config = charts[0]  # Use the highest confidence chart
        result = generator.generate_chart(data, chart_config)
        
        if "error" in result:
            print(f"  ❌ Chart generation failed: {result['error']}")
        else:
            print(f"  ✅ Generated {result['chart_type']} chart")
            print(f"    Title: {result['config']['title']}")
            print(f"    Data points: {len(data)}")
            if 'data_summary' in result:
                print(f"    Summary: {list(result['data_summary'].keys())}")
    else:
        print(f"  ❌ No charts to generate")
    
    print()

async def test_full_orchestrator():
    """Test the complete chart orchestrator."""
    print("🎭 Testing Chart Orchestrator...")
    
    orchestrator = ChartOrchestrator()
    datasets = create_sample_datasets()
    
    test_cases = [
        ("sales_by_category", "Sales data with categories"),
        ("monthly_trends", "Time series data"),
        ("rating_distribution", "Distribution data"),
        ("pii_filtered_data", "PII filtered data"),
        ("insufficient_data", "Insufficient data")
    ]
    
    for dataset_name, description in test_cases:
        data = datasets[dataset_name]
        result = await orchestrator.process_data_for_charts(data)
        
        status = "✅" if result['eligible'] else "❌"
        print(f"  {status} {description}:")
        print(f"    Eligible: {result['eligible']}")
        print(f"    Reason: {result['reason']}")
        if result['eligible']:
            print(f"    Charts generated: {len(result['charts'])}")
            for i, chart in enumerate(result['charts'], 1):
                print(f"      {i}. {chart['chart_type']} - {chart['config']['title']}")
        print()

def print_feature_summary():
    """Print a summary of chart generation features."""
    print("📋 Chart Generation Feature Summary")
    print("=" * 50)
    print()
    
    print("🎯 Eligibility Criteria:")
    print("  • Minimum 2 data points required")
    print("  • Maximum 1000 data points supported")
    print("  • Must contain non-PII data")
    print("  • Requires numeric or categorical data suitable for visualization")
    print()
    
    print("📊 Supported Chart Types:")
    print("  • Bar Charts: Categorical vs Numeric data")
    print("  • Line Charts: Sequential/trend data")
    print("  • Pie Charts: Categorical distributions (2-8 categories)")
    print("  • Scatter Plots: Numeric vs Numeric correlations")
    print("  • Histograms: Numeric value distributions")
    print("  • Time Series: Date/time based trends")
    print()
    
    print("🔒 PII Protection:")
    print("  • Automatically excludes columns marked as [PRIVATE]")
    print("  • Ensures chart generation respects data privacy")
    print("  • Prevents visualization of sensitive information")
    print()
    
    print("🎨 Chart Features:")
    print("  • Interactive Plotly visualizations")
    print("  • Responsive design")
    print("  • Confidence scoring for chart recommendations")
    print("  • Automatic chart type selection")
    print("  • Data summary statistics")
    print()

async def main():
    """Run all chart generation tests."""
    print("🚀 Chart Generation Test Suite")
    print("=" * 50)
    print()
    
    try:
        await test_eligibility_analyzer()
        await test_chart_type_detector()
        await test_chart_generator()
        await test_full_orchestrator()
        
        print("✅ All tests completed!")
        print()
        
        print_feature_summary()
        
    except Exception as e:
        print(f"❌ Test suite failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 