#!/usr/bin/env python3
"""
Simple Weather Fetcher - Fetches real weather data using requests only
No external dependencies beyond Python standard library and requests
"""

import requests
import json
from datetime import datetime, date, timedelta

def fetch_weather(location, days=7):
    """Fetch weather data from wttr.in API"""
    try:
        # wttr.in provides JSON weather data
        url = f"https://wttr.in/{location}?format=j1"
        
        headers = {
            'User-Agent': 'curl/7.68.0',
            'Accept': 'application/json'
        }
        
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        data = response.json()
        
        forecasts = []
        today = date.today()
        
        # wttr.in returns current + forecast days
        weather_days = data.get('weather', [])
        
        for i, day_data in enumerate(weather_days[:days]):
            forecast_date = today + timedelta(days=i)
            
            max_temp = float(day_data.get('maxtempC', 0))
            min_temp = float(day_data.get('mintempC', 0))
            avg_temp = float(day_data.get('avgtempC', 0))
            
            # Get hourly data for conditions
            hourly = day_data.get('hourly', [])
            if hourly:
                # Use midday conditions (around 12:00)
                midday = hourly[12] if len(hourly) > 12 else hourly[0]
                conditions = midday.get('weatherDesc', [{}])[0].get('value', 'Unknown')
                precip = float(midday.get('precipMM', 0))
            else:
                conditions = day_data.get('hourly', [{}])[0].get('weatherDesc', [{}])[0].get('value', 'Unknown')
                precip = float(day_data.get('totalprecipMM', 0))
            
            forecasts.append({
                'date': forecast_date.strftime('%Y-%m-%d'),
                'high_temp_c': max_temp,
                'low_temp_c': min_temp,
                'conditions': conditions,
                'precipitation_mm': precip,
                'source': 'wttr.in'
            })
        
        return forecasts, None
        
    except requests.exceptions.RequestException as e:
        return None, f"Network error: {e}"
    except json.JSONDecodeError as e:
        return None, f"Data parsing error: {e}"
    except Exception as e:
        return None, f"Unexpected error: {e}"

def generate_markdown_report(forecast_data_by_location):
    """Generate markdown report"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')
    
    report = []
    report.append("# Weather Forecast Report")
    report.append(f"\n**Generated:** {timestamp}")
    report.append("\n---\n")
    
    for location, forecasts in forecast_data_by_location.items():
        report.append(f"## 📍 {location.title()}")
        report.append("")
        
        if forecasts:
            avg_high = sum(f['high_temp_c'] for f in forecasts) / len(forecasts)
            avg_low = sum(f['low_temp_c'] for f in forecasts) / len(forecasts)
            
            report.append(f"**Period:** {forecasts[0]['date']} to {forecasts[-1]['date']} ({len(forecasts)} days)")
            report.append(f"**Average High:** {avg_high:.1f}°C")
            report.append(f"**Average Low:** {avg_low:.1f}°C")
            report.append("")
        
        report.append("| Date | High (°C) | Low (°C) | Conditions | Precipitation (mm) | Source |")
        report.append("|------|-----------|----------|------------|-------------------|--------|")
        
        for fc in forecasts:
            report.append(
                f"| {fc['date']} | {fc['high_temp_c']:.1f} | {fc['low_temp_c']:.1f} | "
                f"{fc['conditions']} | {fc['precipitation_mm']:.1f} | {fc['source']} |"
            )
        
        report.append("")
        report.append("---")
        report.append("")
    
    report.append("## Data Sources")
    report.append("- **wttr.in**: Open weather data service (https://wttr.in)")
    report.append("")
    report.append("*Temperatures in Celsius. Precipitation in millimeters.*")
    
    return "\n".join(report)

def main():
    print("=" * 60)
    print("Weather Forecast Data Retrieval System")
    print("=" * 60)
    print()
    
    locations = ['London', 'Manchester']
    forecast_days = 7
    
    today = date.today()
    print(f"Fetching {forecast_days}-day forecast for: {', '.join(locations)}")
    print(f"Current date: {today}")
    print()
    
    all_forecasts = {}
    errors = []
    
    for location in locations:
        print(f"Fetching weather for {location}...")
        forecasts, error = fetch_weather(location.lower(), forecast_days)
        
        if forecasts:
            all_forecasts[location] = forecasts
            print(f"  ✓ Retrieved {len(forecasts)} days of data")
            print(f"    Sample: {forecasts[0]['date']} - {forecasts[0]['conditions']}, "
                  f"{forecasts[0]['high_temp_c']}°C / {forecasts[0]['low_temp_c']}°C")
        else:
            errors.append(f"{location}: {error}")
            print(f"  ✗ Error: {error}")
    
    print()
    
    if all_forecasts:
        print("Generating report...")
        report = generate_markdown_report(all_forecasts)
        
        # Save report
        import os
        output_dir = "output/responses"
        os.makedirs(output_dir, exist_ok=True)
        
        report_path = os.path.join(output_dir, "weather_forecast_report.md")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"✓ Report saved to: {report_path}")
        print()
        print("=" * 60)
        print("EXECUTION COMPLETE")
        print("=" * 60)
        print()
        print(report)
        return 0
    else:
        print("❌ No weather data retrieved")
        print("\nErrors:")
        for err in errors:
            print(f"  • {err}")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())
