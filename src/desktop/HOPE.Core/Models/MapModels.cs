using System.Collections.ObjectModel;

namespace HOPE.Core.Models;

public class MapRow
{
    public int Index { get; set; }
    public ObservableCollection<double> Values { get; } = new();
}

public class DiffMapRow
{
    public int Index { get; set; }
    public ObservableCollection<DiffCell> Cells { get; } = new();
}

public class DiffCell
{
    public double BaseValue { get; set; }
    public double CompareValue { get; set; }

    public double Difference => CompareValue - BaseValue;
    public double PercentChange => BaseValue != 0 ? (Difference / BaseValue) * 100 : (CompareValue != 0 ? 100 : 0);
    public bool HasChanged => Math.Abs(Difference) > 0.001;

    public string DisplayText => $"Base: {BaseValue:F2}\nCompare: {CompareValue:F2}\nDiff: {Difference:F2} ({PercentChange:F1}%)";
}
