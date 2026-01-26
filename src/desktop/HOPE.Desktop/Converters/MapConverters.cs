using System.Globalization;
using System.Windows.Data;
using System.Windows.Media;

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
