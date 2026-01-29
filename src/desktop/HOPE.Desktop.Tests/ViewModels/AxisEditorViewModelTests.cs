using Xunit;
using Moq;
using HOPE.Desktop.ViewModels;
using Microsoft.Extensions.Logging;
using System.Linq;

namespace HOPE.Desktop.Tests.ViewModels;

public class AxisEditorViewModelTests
{
    private readonly Mock<ILogger<AxisEditorViewModel>> _mockLogger;

    public AxisEditorViewModelTests()
    {
        _mockLogger = new Mock<ILogger<AxisEditorViewModel>>();
    }

    private AxisEditorViewModel CreateSUT()
    {
        return new AxisEditorViewModel(_mockLogger.Object);
    }

    [Fact]
    public void Apply_ShouldSortValues_MonotonicityEnforcement()
    {
        // Arrange
        var vm = CreateSUT();
        vm.LoadAxis(new double[] { 10, 20, 30 }); // Original size 3
        vm.NewAxisValuesInput = "30, 10, 20";

        double[]? appliedValues = null;
        vm.ApplyRequested += (vals, interp) => appliedValues = vals;

        // Act
        vm.ApplyCommand.Execute(null);

        // Assert
        Assert.NotNull(appliedValues);
        Assert.Equal(3, appliedValues.Length);
        Assert.Equal(10, appliedValues[0]);
        Assert.Equal(20, appliedValues[1]);
        Assert.Equal(30, appliedValues[2]);
    }

    [Fact]
    public void Apply_ShouldParseMultilineInput()
    {
        // Arrange
        var vm = CreateSUT();
        vm.LoadAxis(new double[] { 0, 0 });
        vm.NewAxisValuesInput = "10\r\n20";

        double[]? appliedValues = null;
        vm.ApplyRequested += (vals, interp) => appliedValues = vals;

        // Act
        vm.ApplyCommand.Execute(null);

        // Assert
        Assert.Equal(new double[] { 10, 20 }, appliedValues);
    }

    [Fact]
    public void Apply_ShouldLogAndNotFire_OnGarbageInput()
    {
        // Arrange
        var vm = CreateSUT();
        vm.LoadAxis(new double[] { 10 });
        vm.NewAxisValuesInput = "NotANumber";

        bool fired = false;
        vm.ApplyRequested += (vals, interp) => fired = true;

        // Act
        vm.ApplyCommand.Execute(null);

        // Assert
        Assert.False(fired);
        _mockLogger.Verify(
            x => x.Log(
                LogLevel.Error,
                It.IsAny<EventId>(),
                It.Is<It.IsAnyType>((v, t) => true),
                It.IsAny<Exception>(),
                It.Is<Func<It.IsAnyType, Exception?, string>>((v, t) => true)),
            Times.Once);
    }
}
