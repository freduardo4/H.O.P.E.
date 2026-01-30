
namespace HOPE.Desktop;

public class Program
{
    [System.STAThreadAttribute()]
    public static void Main()
    {
        try
        {
            var app = new HOPE.Desktop.App();
            app.InitializeComponent();
            app.Run();
        }
        catch (System.Exception ex)
        {
            System.IO.File.WriteAllText(@"c:\Users\Test\Documents\H.O.P.E\desktop_fatal_error.txt", ex.ToString());
            System.Windows.MessageBox.Show(ex.ToString());
        }
    }
}
