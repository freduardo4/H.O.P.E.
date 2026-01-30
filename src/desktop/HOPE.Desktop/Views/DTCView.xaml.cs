using HOPE.Desktop.ViewModels;

namespace HOPE.Desktop.Views;

public partial class DTCView
{
    public DTCView(DTCViewModel viewModel)
    {
        InitializeComponent();
        DataContext = viewModel;
    }
}
