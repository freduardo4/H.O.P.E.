using System.Globalization;
using System.Windows.Data;
using System.Windows.Media;
using HOPE.Core.Models;

namespace HOPE.Desktop.Converters;

public class RpmAxisConverter : IValueConverter
{
    public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
    {
        int index = (int)value;
        int step = int.Parse(parameter.ToString() ?? "500");
        return (index * step + 800).ToString();
    }

    public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
    {
        return Binding.DoNothing;
    }
}

public class ValueToHeatmapConverter : IValueConverter
{
    public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
    {
        double val = (double)value;

        // Target AFR 10.0 (Rich/Blue) to 15.0 (Lean/Red)
        // For now, let's do a simple Green/Yellow/Red
        if (val < 13.0) return new SolidColorBrush(Color.FromRgb(46, 125, 50)); // Deep Green
        if (val < 14.0) return new SolidColorBrush(Color.FromRgb(102, 187, 106)); // Green
        if (val < 14.5) return new SolidColorBrush(Color.FromRgb(251, 192, 45)); // Yellow
        return new SolidColorBrush(Color.FromRgb(211, 47, 47)); // Red
    }

    public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
    {
        return Binding.DoNothing;
    }
}

/// <summary>
/// Converts a diff value (percentage change) to a color gradient.
/// Positive changes (increases) are shown in green, negative (decreases) in red.
/// </summary>
public class DiffToColorConverter : IValueConverter
{
    private static readonly Color PositiveColor = Color.FromRgb(46, 125, 50);   // Green - increase
    private static readonly Color NegativeColor = Color.FromRgb(211, 47, 47);   // Red - decrease
    private static readonly Color NeutralColor = Color.FromRgb(45, 45, 45);     // Dark gray - no change

    public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
    {
        if (value is not DiffCell cell)
            return new SolidColorBrush(NeutralColor);

        if (Math.Abs(cell.PercentChange) < 0.01)
            return new SolidColorBrush(NeutralColor);

        // Calculate intensity based on percentage change (cap at +/- 50%)
        double intensity = Math.Min(Math.Abs(cell.PercentChange) / 50.0, 1.0);

        Color baseColor = cell.PercentChange > 0 ? PositiveColor : NegativeColor;

        // Interpolate between neutral and base color based on intensity
        byte r = (byte)(NeutralColor.R + (baseColor.R - NeutralColor.R) * intensity);
        byte g = (byte)(NeutralColor.G + (baseColor.G - NeutralColor.G) * intensity);
        byte b = (byte)(NeutralColor.B + (baseColor.B - NeutralColor.B) * intensity);

        return new SolidColorBrush(Color.FromRgb(r, g, b));
    }

    public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
    {
        return Binding.DoNothing;
    }
}

/// <summary>
/// Represents a cell in the diff visualization with base, compare, and diff values.
/// </summary>


