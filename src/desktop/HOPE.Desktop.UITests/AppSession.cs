using OpenQA.Selenium.Appium;
using OpenQA.Selenium.Appium.Windows;

namespace HOPE.Desktop.UITests;

public class AppSession
{
    private const string WindowsApplicationDriverUrl = "http://127.0.0.1:4723";
    
    // TODO: Update with actual build output path
    private const string AppPath = @"..\..\..\..\HOPE.Desktop\bin\Debug\net8.0-windows\HOPE.Desktop.exe";

    protected WindowsDriver? _session;

    public void Setup(TestContext context)
    {
        if (_session != null) return;

        var options = new AppiumOptions();
        options.App = Path.GetFullPath(AppPath);
        options.DeviceName = "WindowsPC";
        options.PlatformName = "Windows";

        _session = new WindowsDriver(new Uri(WindowsApplicationDriverUrl), options);
        
        Assert.That(_session, Is.Not.Null);
        _session.Manage().Timeouts().ImplicitWait = TimeSpan.FromSeconds(1.5);
    }

    public void TearDown()
    {
        if (_session != null)
        {
            _session.Quit();
            _session = null;
        }
    }
}
