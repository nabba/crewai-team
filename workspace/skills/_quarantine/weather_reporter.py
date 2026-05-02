from datetime import datetime, date, timedelta
from typing import Dict, List
from weather_fetcher import DailyForecast

class WeatherReporter:
    def __init__(self):
        pass
    
    def generate_markdown_report(self, forecast_data_by_location: Dict[str, List[DailyForecast]]) -> str:
        """Generate a standardized markdown weather report"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')
        
        report = []
        report.append("# Weather Forecast Report")
        report.append(f"\n**Generated:** {timestamp}\n")
        report.append("---\n")
        
        for location, forecasts in forecast_data_by_location.items():
            report.append(f"## 📍 {location.title()}")
            report.append("")
            
            # Summary statistics
            if forecasts:
                avg_high = sum(f.high_temp_c for f in forecasts) / len(forecasts)
                avg_low = sum(f.low_temp_c for f in forecasts) / len(forecasts)
                max_precip = max(f.precipitation_mm for f in forecasts)
                
                report.append(f"**Period:** {forecasts[0].date} to {forecasts[-1].date} ({len(forecasts)} days)")
                report.append(f"**Average High:** {avg_high:.1f}°C")
                report.append(f"**Average Low:** {avg_low:.1f}°C")
                report.append(f"**Max Precipitation:** {max_precip:.1f}mm")
                report.append("")
            
            # Create markdown table
            report.append("| Date | High (°C) | Low (°C) | Conditions | Precipitation (mm) | Source |")
            report.append("|------|-----------|----------|------------|-------------------|--------|")
            
            for fc in forecasts:
                report.append(
                    f"| {fc.date} | {fc.high_temp_c:.1f} | {fc.low_temp_c:.1f} | "
                    f"{fc.conditions} | {fc.precipitation_mm:.1f} | {fc.source} |"
                )
            
            report.append("")
            report.append("---")
            report.append("")
        
        # Data sources note
        report.append("## Data Sources")
        report.append("This report is generated from live weather data sources:")
        report.append("- **BBC Weather**: UK-based meteorological service")
        report.append("- **wttr.in**: Open weather data service")
        report.append("- Cross-referenced and merged for accuracy")
        report.append("")
        report.append("*Note: All temperatures are in Celsius. Precipitation values are cumulative daily totals.*")
        
        return "\n".join(report)
