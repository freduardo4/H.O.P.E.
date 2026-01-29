using System;
using HOPE.Core.Services.ECU;
using Xunit;

namespace HOPE.Desktop.Tests.Services.ECU;

public class CalibrationRepositoryTests
{
    [Fact]
    public void InterpolateMap_ResamplesCorrectly()
    {
        // Arrange
        var oldAxis = new double[] { 0, 10, 20 };
        var oldValues = new double[] { 0, 100, 200 }; // y = 10x
        
        var newAxis = new double[] { 0, 5, 10, 15, 20 };
        
        // Act
        var result = CalibrationRepository.InterpolateMap(oldAxis, oldValues, newAxis);
        
        // Assert
        Assert.Equal(5, result.Length);
        Assert.Equal(0, result[0]);   // x=0
        Assert.Equal(50, result[1]);  // x=5 (interp)
        Assert.Equal(100, result[2]); // x=10
        Assert.Equal(150, result[3]); // x=15 (interp)
        Assert.Equal(200, result[4]); // x=20
    }

    [Fact]
    public void InterpolateMap_ExtrapolatesClamping()
    {
        // Arrange
        var oldAxis = new double[] { 10, 20 };
        var oldValues = new double[] { 100, 200 };
        
        var newAxis = new double[] { 0, 30 }; // Outside range
        
        // Act
        var result = CalibrationRepository.InterpolateMap(oldAxis, oldValues, newAxis);
        
        // Assert
        Assert.Equal(100, result[0]); // Clamped to start
        Assert.Equal(200, result[1]); // Clamped to end
    }
}
