using System;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using HOPE.Core.Models.Simulation;
using HOPE.Core.Services.Simulation;
using System.Collections.ObjectModel;

namespace HOPE.Desktop.ViewModels;

public partial class HiLSimulationViewModel : ObservableObject
{
    private readonly IHiLService _hilService;
    private readonly IBeamNgService _beamNg;

    [ObservableProperty]
    private bool _isSimulationConnected;

    public ObservableCollection<HiLFault> ActiveFaults { get; } = new();

    public HiLSimulationViewModel(IHiLService hilService, IBeamNgService beamNg)
    {
        _hilService = hilService;
        _beamNg = beamNg;
        UpdateState();
    }

    private void UpdateState()
    {
        IsSimulationConnected = _beamNg.IsConnected;
        ActiveFaults.Clear();
        foreach (var fault in _hilService.ActiveFaults)
        {
            ActiveFaults.Add(fault);
        }
    }

    [RelayCommand]
    private void InjectVoltageDrop()
    {
        _hilService.InjectFault(new HiLFault 
        { 
            Type = FaultType.VoltageDrop, 
            Intensity = 0.8, 
            DurationMs = 5000, 
            OccurredAt = DateTime.UtcNow 
        });
        UpdateState();
    }

    [RelayCommand]
    private void InjectRpmNoise()
    {
        _hilService.InjectFault(new HiLFault 
        { 
            Type = FaultType.SensorNoise, 
            TargetParameter = "Rpm",
            Intensity = 0.5, 
            DurationMs = 10000, 
            OccurredAt = DateTime.UtcNow 
        });
        UpdateState();
    }

    [RelayCommand]
    private void InjectHighTemp()
    {
        _hilService.InjectFault(new HiLFault 
        { 
            Type = FaultType.HighTemp,
            OccurredAt = DateTime.UtcNow 
        });
        UpdateState();
    }

    [RelayCommand]
    private void ClearAllFaults()
    {
        _hilService.ClearFaults();
        UpdateState();
    }
}
