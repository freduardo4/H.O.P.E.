using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using System.Collections.ObjectModel;
using System.Linq;

namespace HOPE.Desktop.ViewModels;

public partial class AxisEditorViewModel : ObservableObject
{
    [ObservableProperty]
    private ObservableCollection<string> _originalAxisValues = new();

    [ObservableProperty]
    private string _newAxisValuesInput = string.Empty;

    [ObservableProperty]
    private bool _shouldInterpolate = true;

    private double[] _originalDoubles = Array.Empty<double>();

    public event Action<double[], bool>? ApplyRequested;
    public event Action? CancelRequested;

    public void LoadAxis(double[] axisValues)
    {
        _originalDoubles = axisValues;
        OriginalAxisValues.Clear();
        foreach (var v in axisValues)
        {
            OriginalAxisValues.Add(v.ToString("F0"));
        }
        
        // Pre-fill input with current values
        NewAxisValuesInput = string.Join(Environment.NewLine, axisValues.Select(v => v.ToString("F0")));
    }

    [RelayCommand]
    private void Apply()
    {
        try
        {
            var lines = NewAxisValuesInput.Split(new[] { '\r', '\n', ',' }, StringSplitOptions.RemoveEmptyEntries);
            var newDoubles = lines.Select(double.Parse).OrderBy(x => x).ToArray(); // Sort to enforce monotonicity
            
            if (newDoubles.Length != _originalDoubles.Length)
            {
                // Warn size mismatch? For now allow it or just take what's given. The main VM might expect fixed size maps.
                // We'll pass it back and let VM handle validation.
            }

            ApplyRequested?.Invoke(newDoubles, ShouldInterpolate);
        }
        catch
        {
            // Handle parsing error
        }
    }

    [RelayCommand]
    private void Cancel()
    {
        CancelRequested?.Invoke();
    }
}
