using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using HOPE.Core.Services.ECU;
using System.Collections.ObjectModel;

namespace HOPE.Desktop.ViewModels;

public partial class MapVisualizationViewModel : ObservableObject
{
    private readonly IECUService _ecuService;

    [ObservableProperty]
    private string _statusMessage = "Ready to Read ECU";

    [ObservableProperty]
    private bool _isBusy;

    public ObservableCollection<MapRow> MapData { get; } = new();

    public MapVisualizationViewModel(IECUService ecuService)
    {
        _ecuService = ecuService;
    }

    [RelayCommand]
    private async Task ReadEcuMapAsync()
    {
        IsBusy = true;
        StatusMessage = "Reading ECU Map...";
        
        try
        {
            var data = await _ecuService.ReadMapAsync("FuelMap");
            
            MapData.Clear();
            for (int i = 0; i < data.GetLength(0); i++)
            {
                var row = new MapRow { Index = i };
                for (int j = 0; j < data.GetLength(1); j++)
                {
                    row.Values.Add(data[i, j]);
                }
                MapData.Add(row);
            }
            
            StatusMessage = "Map Loaded Successfully";
        }
        catch (Exception ex)
        {
            StatusMessage = $"Error: {ex.Message}";
        }
        finally
        {
            IsBusy = false;
        }
    }
}

public class MapRow
{
    public int Index { get; set; }
    public ObservableCollection<double> Values { get; } = new();
}
