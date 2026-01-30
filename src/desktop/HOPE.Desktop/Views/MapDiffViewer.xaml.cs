using System.Windows.Controls;
using HOPE.Desktop.ViewModels;

namespace HOPE.Desktop.Views;

public partial class MapDiffViewer : UserControl
{
    public MapDiffViewer(MapDiffViewModel viewModel)
    {
        InitializeComponent();
        DataContext = viewModel;
    }
}
