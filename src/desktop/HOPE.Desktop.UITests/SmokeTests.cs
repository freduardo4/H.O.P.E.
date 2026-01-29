using NUnit.Framework;

namespace HOPE.Desktop.UITests;

[TestFixture]
public class SmokeTests : AppSession
{
    [SetUp]
    public void TestSetup()
    {
        Setup(TestContext.CurrentContext);
    }

    [TearDown]
    public void TestTearDown()
    {
        TearDown();
    }

    [Test]
    public void AppLaunches_AndShowsMainWindow()
    {
        // Act
        var title = _session!.Title;

        // Assert
        Assert.That(title, Does.Contain("HOPE Desktop"), "Main window title should contain app name");
        
        // Optional: Verify specific elements if we knew their AutomationIDs
        // var loginBtn = _session.FindElementByAccessibilityId("LoginButton");
        // Assert.IsNotNull(loginBtn);
    }
}
