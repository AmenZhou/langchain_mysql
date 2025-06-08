#!/usr/bin/env python3
"""
Standalone Pie Chart Demo
Tests pie chart generation with real data from the database.
"""

import asyncio
import os
import sys
from typing import List, Dict, Any

# Add the backend directory to Python path
sys.path.insert(0, 'backend')

from charts.orchestrator import ChartOrchestrator
from charts.models import ChartSettings

# Sample data for testing (simulating film categories)
SAMPLE_DATA = [
    {"category": "Action", "film_count": 64},
    {"category": "Animation", "film_count": 66}, 
    {"category": "Children", "film_count": 60},
    {"category": "Classics", "film_count": 57},
    {"category": "Comedy", "film_count": 58},
    {"category": "Documentary", "film_count": 68},
    {"category": "Drama", "film_count": 62},
    {"category": "Family", "film_count": 69},
    {"category": "Foreign", "film_count": 73},
    {"category": "Games", "film_count": 61},
    {"category": "Horror", "film_count": 56},
    {"category": "Music", "film_count": 51},
    {"category": "New", "film_count": 63},
    {"category": "Sci-Fi", "film_count": 61},
    {"category": "Sports", "film_count": 74},
    {"category": "Travel", "film_count": 57}
]

# Sample data for pie chart testing (fewer categories for better pie chart)
PIE_CHART_DATA = [
    {"store_location": "California", "total_sales": 87543},
    {"store_location": "Texas", "total_sales": 65432},
    {"store_location": "New York", "total_sales": 98765},
    {"store_location": "Florida", "total_sales": 54321},
    {"store_location": "Illinois", "total_sales": 67890}
]

RATING_DATA = [
    {"rating": "G", "count": 178},
    {"rating": "PG", "count": 194}, 
    {"rating": "PG-13", "count": 223},
    {"rating": "R", "count": 195},
    {"rating": "NC-17", "count": 210}
]

async def test_pie_chart_generation():
    """Test pie chart generation with sample data."""
    print("🥧 Testing Pie Chart Generation")
    print("=" * 50)
    
    # Create chart orchestrator
    orchestrator = ChartOrchestrator()
    
    # Test 1: Store sales pie chart (perfect for pie chart)
    print("\n📊 Test 1: Store Sales Distribution")
    print("-" * 30)
    
    try:
        result = await orchestrator.process_data_for_charts(PIE_CHART_DATA)
        
        if result and result.get('eligible'):
            print(f"✅ Chart generation successful!")
            print(f"📈 Generated {result.get('recommendations', 0)} chart(s)")
            print(f"💡 Reason: {result.get('reason', 'N/A')}")
            
            # Display chart details
            charts = result.get('charts', [])
            for i, chart in enumerate(charts, 1):
                print(f"\n📊 Chart {i}: {chart.get('chart_type', 'Unknown').upper()}")
                print(f"   Title: {chart.get('config', {}).get('title', 'N/A')}")
                print(f"   Confidence: {chart.get('config', {}).get('confidence_score', 0):.1%}")
                
                # Show data summary
                summary = chart.get('data_summary', {})
                if 'total_categories' in summary:
                    print(f"   Categories: {summary['total_categories']}")
                    print(f"   Total Items: {summary.get('total_items', 'N/A')}")
                    print(f"   Largest: {summary.get('largest_category', 'N/A')} ({summary.get('largest_percentage', 0):.1f}%)")
                
                # Show category breakdown if available
                if 'category_breakdown' in summary:
                    print(f"   Category Details:")
                    for cat in summary['category_breakdown'][:5]:  # Show first 5
                        print(f"     • {cat['category']}: {cat['count']} ({cat['percentage']:.1f}%)")
        else:
            print(f"❌ Chart generation failed")
            print(f"💡 Reason: {result.get('reason', 'Unknown error') if result else 'No result returned'}")
    
    except Exception as e:
        print(f"❌ Error during chart generation: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Test 2: Film ratings pie chart
    print("\n📊 Test 2: Film Ratings Distribution")
    print("-" * 30)
    
    try:
        result = await orchestrator.process_data_for_charts(RATING_DATA)
        
        if result and result.get('eligible'):
            print(f"✅ Chart generation successful!")
            print(f"📈 Generated {result.get('recommendations', 0)} chart(s)")
            
            # Show pie chart specific data
            charts = result.get('charts', [])
            pie_charts = [c for c in charts if c.get('chart_type') == 'pie']
            
            if pie_charts:
                pie_chart = pie_charts[0]
                summary = pie_chart.get('data_summary', {})
                print(f"🥧 Pie Chart Details:")
                print(f"   Data Concentration: {summary.get('data_concentration', 'N/A')}")
                print(f"   Diversity Index: {summary.get('diversity_index', 0):.3f}")
                print(f"   Top 3 Share: {summary.get('top_3_categories_share', 0):.1f}%")
        else:
            print(f"❌ Chart generation failed: {result.get('reason', 'Unknown') if result else 'No result'}")
    
    except Exception as e:
        print(f"❌ Error: {str(e)}")
    
    # Test 3: Film categories (many categories - should still work but lower confidence)
    print("\n📊 Test 3: Film Categories (Many Categories)")
    print("-" * 30)
    
    try:
        result = await orchestrator.process_data_for_charts(SAMPLE_DATA)
        
        if result and result.get('eligible'):
            charts = result.get('charts', [])
            pie_charts = [c for c in charts if c.get('chart_type') == 'pie']
            
            if pie_charts:
                pie_chart = pie_charts[0]
                confidence = pie_chart.get('config', {}).get('confidence_score', 0)
                print(f"✅ Pie chart generated with {confidence:.1%} confidence")
                print(f"💡 Note: Lower confidence due to many categories ({len(SAMPLE_DATA)})")
            else:
                print(f"ℹ️  No pie chart generated (likely due to too many categories)")
                print(f"📊 Generated {len(charts)} other chart types instead")
        else:
            print(f"❌ Failed: {result.get('reason', 'Unknown') if result else 'No result'}")
    
    except Exception as e:
        print(f"❌ Error: {str(e)}")

def print_pie_chart_features():
    """Print information about pie chart capabilities."""
    print("\n🥧 Pie Chart Features Summary")
    print("=" * 50)
    print("✨ Enhanced Features:")
    print("  • 🎨 Beautiful color palettes")
    print("  • 📊 Interactive hover information")
    print("  • 📈 Detailed data summaries")
    print("  • 🎯 Smart confidence scoring")
    print("  • 📋 Category breakdown analysis")
    print("  • 🔍 Data concentration metrics")
    print("  • 📊 Simpson's diversity index")
    print("  • 🏆 Best/worst category highlighting")
    
    print("\n🎯 Optimal Use Cases:")
    print("  • 2-5 categories: 90% confidence (optimal)")
    print("  • 6-8 categories: 60% confidence (good)")
    print("  • Store sales by location")
    print("  • Product category distribution")
    print("  • Rating/grade breakdowns")
    print("  • Market share analysis")
    
    print("\n⚙️ Technical Features:")
    print("  • Plotly.js interactive charts")
    print("  • Responsive design")
    print("  • Custom color schemes")
    print("  • Percentage and count display")
    print("  • Hover tooltips with details")
    print("  • Legend positioning")

async def main():
    """Main function to run all tests."""
    print("🚀 Pie Chart Demo - LangChain MySQL Application")
    print("=" * 60)
    
    try:
        await test_pie_chart_generation()
        print_pie_chart_features()
        
        print("\n🎉 Demo Complete!")
        print("=" * 60)
        print("💡 To use pie charts in your application:")
        print("   1. Start the backend server: cd backend && python -m uvicorn server:app --port 8000")
        print("   2. Open the demo: frontend/chart_demo.html")
        print("   3. Try queries like:")
        print("      • 'Show store sales by location'")
        print("      • 'Film rating distribution'") 
        print("      • 'Category breakdown'")
        print("   4. Enable charts with the toggle")
        print("   5. Enjoy beautiful pie charts! 🥧")
        
    except Exception as e:
        print(f"❌ Demo failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main()) 