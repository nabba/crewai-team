#!/usr/bin/env python3
"""
Weather Forecast Data Retrieval and Reporting System
Executes the full pipeline: fetch -> validate -> merge -> report
"""

from datetime import datetime, date, timedelta
import sys
import os

# Add skills directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from weather_fetcher import WeatherFetcher, IncompleteDataError, SimulatedDataError
from weather_reporter import WeatherReporter

def main():
    print("=" * 60)
    print("Weather Forecast Data Retrieval System")
    print("=" * 60)
    print()
    
    # Configuration
    locations = ['london', 'manchester']
    forecast_days = 7
    
    # Calculate date range
    today = date.today()
    start_date = today
    end_date = today + timedelta(days=forecast_days - 1)
    
    print(f"Configuration:")
    print(f"  Locations: {', '.join([l.title() for l in locations])}")
    print(f"  Forecast period: {forecast_days} days")
    print(f"  Date range: {start_date} to {end_date}")
    print(f"  Current date: {today}")
    print()
    
    # Initialize components
    fetcher = WeatherFetcher()
    reporter = WeatherReporter()
    
    all_forecast_data = {}
    errors = []
    
    for location in locations:
        print(f"\n{'=' * 60}")
        print(f"Fetching weather for {location.title()}")
        print(f"{'=' * 60}")
        
        try:
            # Fetch from all sources with retry
            sources = fetcher.fetch_all_sources(location, start_date, end_date, retry_count=2)
            
            if not sources:
                raise IncompleteDataError(f"No weather data sources available for {location}")
            
            print(f"\nSuccessfully fetched from {len(sources)} source(s):")
            for source_name, data in sources.items():
                print(f"  - {source_name}: {len(data)} days")
            
            # Validate completeness for each source
            print("\nValidating data completeness...")
            for source_name, data in sources.items():
                is_complete = fetcher.validate_completeness(data, forecast_days)
                status = "✓ Complete" if is_complete else "✗ Incomplete"
                print(f"  {source_name}: {status}")
            
            # Merge sources
            print("\nMerging data from multiple sources...")
            merged_forecasts = fetcher.merge_sources(sources, location, start_date, end_date)
            print(f"✓ Merged into {len(merged_forecasts)} daily forecasts")
            
            # Validate merged data
            if not fetcher.validate_completeness(merged_forecasts, forecast_days):
                raise IncompleteDataError(f"Merged data for {location} is incomplete")
            
            all_forecast_data[location] = merged_forecasts
            print(f"✓ {location.title()} data validated successfully")
            
        except SimulatedDataError as e:
            error_msg = f"SIMULATED DATA DETECTED for {location}: {e}"
            print(f"\n❌ ERROR: {error_msg}")
            errors.append(error_msg)
            
        except IncompleteDataError as e:
            error_msg = f"INCOMPLETE DATA for {location}: {e}"
            print(f"\n❌ ERROR: {error_msg}")
            errors.append(error_msg)
            
        except Exception as e:
            error_msg = f"UNEXPECTED ERROR for {location}: {e}"
            print(f"\n❌ ERROR: {error_msg}")
            errors.append(error_msg)
    
    # Generate report if we have data
    print(f"\n{'=' * 60}")
    print("Generating Report")
    print(f"{'=' * 60}")
    
    if all_forecast_data:
        report_content = reporter.generate_markdown_report(all_forecast_data)
        
        # Save report to file
        output_dir = "output/responses"
        os.makedirs(output_dir, exist_ok=True)
        
        report_path = os.path.join(output_dir, "weather_forecast_report.md")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"\n✓ Report saved to: {report_path}")
        print(f"\n{len(all_forecast_data)} location(s) included in report")
        
        if errors:
            print(f"\n⚠ Warnings: {len(errors)} error(s) occurred during processing:")
            for err in errors:
                print(f"  - {err}")
        
        print("\n" + "=" * 60)
        print("EXECUTION COMPLETE")
        print("=" * 60)
        print("\nReport Preview:")
        print("-" * 60)
        print(report_content[:1000] + "..." if len(report_content) > 1000 else report_content)
        
        return 0
        
    else:
        print("\n❌ No forecast data available to generate report")
        print("\nErrors encountered:")
        for err in errors:
            print(f"  • {err}")
        
        # Create error report
        error_report = f"""# Weather Forecast Report - ERROR

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}

## Status: FAILED

Unable to retrieve complete weather data from any source.

## Errors Encountered:
"""
        for err in errors:
            error_report += f"- {err}\n"
        
        error_report += """
## Troubleshooting:
- Check network connectivity
- Verify weather service availability
- Try running the script again later
"""
        
        output_dir = "output/responses"
        os.makedirs(output_dir, exist_ok=True)
        report_path = os.path.join(output_dir, "weather_forecast_report.md")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(error_report)
        
        print(f"\nError report saved to: {report_path}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
