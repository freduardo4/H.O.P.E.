using HOPE.Desktop.Converters;
using HOPE.Core.Models;
using Xunit;

namespace HOPE.Desktop.Tests;

public class MapDiffTests
{
    [Fact]
    public void DiffCell_CalculatesDifference_Correctly()
    {
        var cell = new DiffCell
        {
            BaseValue = 14.0,
            CompareValue = 13.5
        };

        Assert.Equal(-0.5, cell.Difference, 3);
    }

    [Fact]
    public void DiffCell_CalculatesPercentChange_ForDecrease()
    {
        var cell = new DiffCell
        {
            BaseValue = 14.0,
            CompareValue = 13.0
        };

        // -1.0 / 14.0 * 100 = -7.14%
        Assert.Equal(-7.14, cell.PercentChange, 1);
    }

    [Fact]
    public void DiffCell_CalculatesPercentChange_ForIncrease()
    {
        var cell = new DiffCell
        {
            BaseValue = 12.0,
            CompareValue = 14.4
        };

        // 2.4 / 12.0 * 100 = 20%
        Assert.Equal(20.0, cell.PercentChange, 1);
    }

    [Fact]
    public void DiffCell_HasChanged_TrueWhenDifferent()
    {
        var cell = new DiffCell
        {
            BaseValue = 14.0,
            CompareValue = 13.9
        };

        Assert.True(cell.HasChanged);
    }

    [Fact]
    public void DiffCell_HasChanged_FalseWhenSame()
    {
        var cell = new DiffCell
        {
            BaseValue = 14.0,
            CompareValue = 14.0
        };

        Assert.False(cell.HasChanged);
    }

    [Fact]
    public void DiffCell_HasChanged_FalseForNegligibleDifference()
    {
        var cell = new DiffCell
        {
            BaseValue = 14.0,
            CompareValue = 14.0001
        };

        Assert.False(cell.HasChanged);
    }

    [Fact]
    public void DiffCell_DisplayText_ShowsDifferenceWhenChanged()
    {
        var cell = new DiffCell
        {
            BaseValue = 14.0,
            CompareValue = 13.5
        };

        // Base: 14.00
        // Compare: 13.50
        // Diff: -0.50 (-3.6%)
        Assert.Contains("Base: 14.00", cell.DisplayText);
        Assert.Contains("Compare: 13.50", cell.DisplayText);
        Assert.Contains("Diff: -0.50", cell.DisplayText);
    }

    [Fact]
    public void DiffCell_DisplayText_ShowsBaseValueWhenUnchanged()
    {
        var cell = new DiffCell
        {
            BaseValue = 14.0,
            CompareValue = 14.0
        };

        // Base: 14.00
        // Compare: 14.00
        // Diff: 0.00 (0.0%)
        Assert.Contains("Base: 14.00", cell.DisplayText);
        Assert.Contains("Compare: 14.00", cell.DisplayText);
        Assert.Contains("Diff: 0.00", cell.DisplayText);
    }

    [Fact]
    public void DiffCell_HandlesZeroBaseValue()
    {
        var cell = new DiffCell
        {
            BaseValue = 0,
            CompareValue = 5.0
        };

        // When base is 0, percent change should be 100% (or special case)
        Assert.Equal(100.0, cell.PercentChange);
    }

    [Fact]
    public void DiffCell_HandlesBothZeroValues()
    {
        var cell = new DiffCell
        {
            BaseValue = 0,
            CompareValue = 0
        };

        Assert.Equal(0.0, cell.PercentChange);
        Assert.False(cell.HasChanged);
    }
}
