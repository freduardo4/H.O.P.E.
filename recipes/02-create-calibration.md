# Recipe 02: Create Calibration File

## Overview
H.O.P.E. uses a JSON-based structure for ECU calibration files (maps). This ensures they are human-readable, version-controllable, and easy to parse.

## File Structure (`.json`)

```json
{
  "meta": {
    "ecuId": "0x12345678",
    "version": "1.0.0",
    "author": "TunerX",
    "baseMapId": "OEM_Generic_v1",
    "timestamp": "2024-01-27T12:00:00Z"
  },
  "tables": [
    {
      "name": "FuelReview (VE)",
      "id": "VE_TABLE_1",
      "type": "3D",
      "xAxis": {
        "label": "RPM",
        "unit": "rpm",
        "values": [800, 1200, 1600, 2000, 2400, 2800, 3200, 3600, 4000, 4400, 4800, 5200, 5600, 6000, 6400, 6800]
      },
      "yAxis": {
        "label": "Load",
        "unit": "kpa",
        "values": [20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200, 230, 250]
      },
      "data": [
        [45, 48, 55, 62, ...], 
        [50, 55, 60, 68, ...]
        // 16x16 matrix
      ]
    },
    {
      "name": "Ignition Timing",
      "id": "IGN_TABLE_1",
      "type": "3D",
      "data": [...]
    }
  ],
  "scalars": {
    "revLimit": 7200,
    "idleTarget": 850,
    "injectorSize": 1000
  },
  "checksum": "sha256:abcd..."
}
```

## Creating Programmatically

```csharp
var calibration = new CalibrationFile 
{
    Meta = new CalibrationMeta { Author = "Me", Version = "1.1" },
    Scalars = new Dictionary<string, double> { { "revLimit", 7500 } }
};

// Add VE Table
calibration.Tables.Add(new Table3D 
{
    Name = "VE",
    XAxis = Axis.RPM_Standard,
    YAxis = Axis.Load_Turbo,
    Data = Helper.GenerateBaseMap(14.7) 
});

// Save (automatically calculates checksum)
await calibrationRepo.SaveAsync(calibration, "my_tune_v1.json");
```
