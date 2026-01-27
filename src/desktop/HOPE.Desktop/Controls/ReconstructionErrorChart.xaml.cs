using System.Collections.ObjectModel;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Animation;
using System.Windows.Shapes;

namespace HOPE.Desktop.Controls;

/// <summary>
/// Interactive chart control for visualizing LSTM autoencoder reconstruction errors
/// and anomaly detection results with ghost curves and threshold zones.
/// </summary>
public partial class ReconstructionErrorChart : UserControl
{
    #region Dependency Properties

    public static readonly DependencyProperty TitleProperty =
        DependencyProperty.Register(nameof(Title), typeof(string), typeof(ReconstructionErrorChart),
            new PropertyMetadata("Reconstruction Error"));

    public static readonly DependencyProperty WarningThresholdProperty =
        DependencyProperty.Register(nameof(WarningThreshold), typeof(double), typeof(ReconstructionErrorChart),
            new PropertyMetadata(0.5, OnThresholdChanged));

    public static readonly DependencyProperty DangerThresholdProperty =
        DependencyProperty.Register(nameof(DangerThreshold), typeof(double), typeof(ReconstructionErrorChart),
            new PropertyMetadata(0.8, OnThresholdChanged));

    public static readonly DependencyProperty MaxYValueProperty =
        DependencyProperty.Register(nameof(MaxYValue), typeof(double), typeof(ReconstructionErrorChart),
            new PropertyMetadata(1.0, OnScaleChanged));

    public static readonly DependencyProperty TimeWindowSecondsProperty =
        DependencyProperty.Register(nameof(TimeWindowSeconds), typeof(int), typeof(ReconstructionErrorChart),
            new PropertyMetadata(60, OnScaleChanged));

    public static readonly DependencyProperty ShowExpectedBaselineProperty =
        DependencyProperty.Register(nameof(ShowExpectedBaseline), typeof(bool), typeof(ReconstructionErrorChart),
            new PropertyMetadata(true, OnVisualPropertyChanged));

    public static readonly DependencyProperty AnimateUpdatesProperty =
        DependencyProperty.Register(nameof(AnimateUpdates), typeof(bool), typeof(ReconstructionErrorChart),
            new PropertyMetadata(true));

    public string Title
    {
        get => (string)GetValue(TitleProperty);
        set => SetValue(TitleProperty, value);
    }

    public double WarningThreshold
    {
        get => (double)GetValue(WarningThresholdProperty);
        set => SetValue(WarningThresholdProperty, value);
    }

    public double DangerThreshold
    {
        get => (double)GetValue(DangerThresholdProperty);
        set => SetValue(DangerThresholdProperty, value);
    }

    public double MaxYValue
    {
        get => (double)GetValue(MaxYValueProperty);
        set => SetValue(MaxYValueProperty, value);
    }

    public int TimeWindowSeconds
    {
        get => (int)GetValue(TimeWindowSecondsProperty);
        set => SetValue(TimeWindowSecondsProperty, value);
    }

    public bool ShowExpectedBaseline
    {
        get => (bool)GetValue(ShowExpectedBaselineProperty);
        set => SetValue(ShowExpectedBaselineProperty, value);
    }

    public bool AnimateUpdates
    {
        get => (bool)GetValue(AnimateUpdatesProperty);
        set => SetValue(AnimateUpdatesProperty, value);
    }

    #endregion

    #region Fields

    private readonly ObservableCollection<ErrorDataPoint> _dataPoints = new();
    private readonly ObservableCollection<ErrorDataPoint> _expectedBaseline = new();
    private readonly List<AnomalyMarker> _anomalyMarkers = new();

    private double _chartWidth;
    private double _chartHeight;
    private DateTime _windowStart;
    private DateTime _windowEnd;

    // Statistics
    private double _currentScore;
    private double _meanScore;
    private double _maxScore;
    private double _stdDev;
    private int _anomalyCount;

    #endregion

    public ReconstructionErrorChart()
    {
        InitializeComponent();

        _windowEnd = DateTime.UtcNow;
        _windowStart = _windowEnd.AddSeconds(-TimeWindowSeconds);

        Loaded += OnLoaded;
        SizeChanged += OnSizeChanged;
        ChartCanvas.MouseMove += OnChartMouseMove;
        ChartCanvas.MouseLeave += OnChartMouseLeave;
    }

    #region Public Methods

    /// <summary>
    /// Add a new reconstruction error data point
    /// </summary>
    public void AddDataPoint(double errorValue, DateTime? timestamp = null)
    {
        var time = timestamp ?? DateTime.UtcNow;

        _dataPoints.Add(new ErrorDataPoint
        {
            Timestamp = time,
            Value = errorValue,
            IsAnomaly = errorValue >= DangerThreshold
        });

        // Remove old points outside the window
        while (_dataPoints.Count > 0 && _dataPoints[0].Timestamp < _windowStart)
        {
            _dataPoints.RemoveAt(0);
        }

        // Update window
        _windowEnd = time;
        _windowStart = _windowEnd.AddSeconds(-TimeWindowSeconds);

        // Update current score with animation
        UpdateCurrentScore(errorValue);

        // Check for anomaly
        if (errorValue >= DangerThreshold)
        {
            AddAnomalyMarker(time, errorValue);
        }

        // Update statistics
        UpdateStatistics();

        // Redraw
        Dispatcher.Invoke(RedrawChart);
    }

    /// <summary>
    /// Add expected baseline data for ghost curve
    /// </summary>
    public void SetExpectedBaseline(IEnumerable<ErrorDataPoint> baselinePoints)
    {
        _expectedBaseline.Clear();
        foreach (var point in baselinePoints)
        {
            _expectedBaseline.Add(point);
        }
        RedrawChart();
    }

    /// <summary>
    /// Clear all data
    /// </summary>
    public void Clear()
    {
        _dataPoints.Clear();
        _expectedBaseline.Clear();
        _anomalyMarkers.Clear();
        AnomalyMarkersCanvas.Children.Clear();

        _currentScore = 0;
        _meanScore = 0;
        _maxScore = 0;
        _stdDev = 0;
        _anomalyCount = 0;

        UpdateStatusDisplay();
        RedrawChart();
    }

    /// <summary>
    /// Auto-calibrate thresholds based on historical data
    /// </summary>
    public void AutoCalibrateThresholds(double warningPercentile = 90, double dangerPercentile = 99)
    {
        if (_dataPoints.Count < 10) return;

        var sortedValues = _dataPoints.Select(p => p.Value).OrderBy(v => v).ToList();
        var warningIndex = (int)(sortedValues.Count * warningPercentile / 100);
        var dangerIndex = (int)(sortedValues.Count * dangerPercentile / 100);

        WarningThreshold = sortedValues[Math.Min(warningIndex, sortedValues.Count - 1)];
        DangerThreshold = sortedValues[Math.Min(dangerIndex, sortedValues.Count - 1)];
    }

    #endregion

    #region Event Handlers

    private void OnLoaded(object sender, RoutedEventArgs e)
    {
        UpdateChartDimensions();
        RedrawChart();
    }

    private void OnSizeChanged(object sender, SizeChangedEventArgs e)
    {
        UpdateChartDimensions();
        RedrawChart();
    }

    private void OnChartMouseMove(object sender, MouseEventArgs e)
    {
        var position = e.GetPosition(ChartCanvas);
        ShowTooltip(position);
    }

    private void OnChartMouseLeave(object sender, MouseEventArgs e)
    {
        TooltipBorder.Visibility = Visibility.Collapsed;
    }

    private static void OnThresholdChanged(DependencyObject d, DependencyPropertyChangedEventArgs e)
    {
        if (d is ReconstructionErrorChart chart)
        {
            chart.RedrawChart();
        }
    }

    private static void OnScaleChanged(DependencyObject d, DependencyPropertyChangedEventArgs e)
    {
        if (d is ReconstructionErrorChart chart)
        {
            chart.RedrawChart();
        }
    }

    private static void OnVisualPropertyChanged(DependencyObject d, DependencyPropertyChangedEventArgs e)
    {
        if (d is ReconstructionErrorChart chart)
        {
            chart.RedrawChart();
        }
    }

    #endregion

    #region Drawing Methods

    private void UpdateChartDimensions()
    {
        _chartWidth = ChartCanvas.ActualWidth;
        _chartHeight = ChartCanvas.ActualHeight;
    }

    private void RedrawChart()
    {
        if (_chartWidth <= 0 || _chartHeight <= 0) return;

        DrawThresholdZones();
        DrawGridLines();
        DrawThresholdLines();
        DrawExpectedBaseline();
        DrawErrorLine();
        DrawAxisLabels();
        UpdateAnomalyMarkers();
    }

    private void DrawThresholdZones()
    {
        var warningY = ValueToY(WarningThreshold);
        var dangerY = ValueToY(DangerThreshold);

        // Normal zone (0 to warning)
        Canvas.SetLeft(NormalZone, 0);
        Canvas.SetTop(NormalZone, warningY);
        NormalZone.Width = _chartWidth;
        NormalZone.Height = _chartHeight - warningY;

        // Warning zone (warning to danger)
        Canvas.SetLeft(WarningZone, 0);
        Canvas.SetTop(WarningZone, dangerY);
        WarningZone.Width = _chartWidth;
        WarningZone.Height = warningY - dangerY;

        // Danger zone (danger to max)
        Canvas.SetLeft(DangerZone, 0);
        Canvas.SetTop(DangerZone, 0);
        DangerZone.Width = _chartWidth;
        DangerZone.Height = dangerY;
    }

    private void DrawGridLines()
    {
        GridCanvas.Children.Clear();

        // Horizontal grid lines
        var ySteps = 5;
        for (int i = 0; i <= ySteps; i++)
        {
            var y = _chartHeight * i / ySteps;
            var line = new Line
            {
                X1 = 0,
                Y1 = y,
                X2 = _chartWidth,
                Y2 = y,
                Stroke = new SolidColorBrush(Color.FromRgb(0x33, 0x33, 0x33)),
                StrokeThickness = 1,
                Opacity = 0.5
            };
            GridCanvas.Children.Add(line);
        }

        // Vertical grid lines (time)
        var xSteps = 6;
        for (int i = 0; i <= xSteps; i++)
        {
            var x = _chartWidth * i / xSteps;
            var line = new Line
            {
                X1 = x,
                Y1 = 0,
                X2 = x,
                Y2 = _chartHeight,
                Stroke = new SolidColorBrush(Color.FromRgb(0x33, 0x33, 0x33)),
                StrokeThickness = 1,
                Opacity = 0.3
            };
            GridCanvas.Children.Add(line);
        }
    }

    private void DrawThresholdLines()
    {
        var warningY = ValueToY(WarningThreshold);
        var dangerY = ValueToY(DangerThreshold);

        WarningThresholdLine.X1 = 0;
        WarningThresholdLine.Y1 = warningY;
        WarningThresholdLine.X2 = _chartWidth;
        WarningThresholdLine.Y2 = warningY;

        DangerThresholdLine.X1 = 0;
        DangerThresholdLine.Y1 = dangerY;
        DangerThresholdLine.X2 = _chartWidth;
        DangerThresholdLine.Y2 = dangerY;
    }

    private void DrawExpectedBaseline()
    {
        if (!ShowExpectedBaseline || _expectedBaseline.Count < 2)
        {
            ExpectedLinePath.Data = null;
            return;
        }

        var geometry = new PathGeometry();
        var figure = new PathFigure();
        var firstPoint = true;

        foreach (var point in _expectedBaseline.OrderBy(p => p.Timestamp))
        {
            var x = TimeToX(point.Timestamp);
            var y = ValueToY(point.Value);

            if (x < 0 || x > _chartWidth) continue;

            if (firstPoint)
            {
                figure.StartPoint = new Point(x, y);
                firstPoint = false;
            }
            else
            {
                figure.Segments.Add(new LineSegment(new Point(x, y), true));
            }
        }

        geometry.Figures.Add(figure);
        ExpectedLinePath.Data = geometry;
    }

    private void DrawErrorLine()
    {
        if (_dataPoints.Count < 2)
        {
            ErrorLinePath.Data = null;
            return;
        }

        var geometry = new PathGeometry();
        var figure = new PathFigure();
        var firstPoint = true;

        foreach (var point in _dataPoints.OrderBy(p => p.Timestamp))
        {
            var x = TimeToX(point.Timestamp);
            var y = ValueToY(point.Value);

            if (x < 0 || x > _chartWidth) continue;

            if (firstPoint)
            {
                figure.StartPoint = new Point(x, y);
                firstPoint = false;
            }
            else
            {
                figure.Segments.Add(new LineSegment(new Point(x, y), true));
            }
        }

        geometry.Figures.Add(figure);
        ErrorLinePath.Data = geometry;

        // Color the line based on current status
        var currentValue = _dataPoints.LastOrDefault()?.Value ?? 0;
        if (currentValue >= DangerThreshold)
        {
            ErrorLinePath.Stroke = new SolidColorBrush(Color.FromRgb(0xFF, 0x44, 0x44));
        }
        else if (currentValue >= WarningThreshold)
        {
            ErrorLinePath.Stroke = new SolidColorBrush(Color.FromRgb(0xFF, 0xFF, 0x00));
        }
        else
        {
            ErrorLinePath.Stroke = new SolidColorBrush(Color.FromRgb(0x00, 0xBF, 0xFF));
        }
    }

    private void DrawAxisLabels()
    {
        YAxisCanvas.Children.Clear();
        XAxisCanvas.Children.Clear();

        // Y-axis labels
        var ySteps = 5;
        for (int i = 0; i <= ySteps; i++)
        {
            var value = MaxYValue * (ySteps - i) / ySteps;
            var y = _chartHeight * i / ySteps - 6;

            var label = new TextBlock
            {
                Text = value.ToString("F2"),
                Foreground = new SolidColorBrush(Color.FromRgb(0xAA, 0xAA, 0xAA)),
                FontSize = 9
            };

            Canvas.SetRight(label, 5);
            Canvas.SetTop(label, y);
            YAxisCanvas.Children.Add(label);
        }

        // X-axis labels (time)
        var xSteps = 6;
        for (int i = 0; i <= xSteps; i++)
        {
            var time = _windowStart.AddSeconds(TimeWindowSeconds * i / xSteps);
            var x = _chartWidth * i / xSteps - 15;

            var label = new TextBlock
            {
                Text = time.ToString("mm:ss"),
                Foreground = new SolidColorBrush(Color.FromRgb(0xAA, 0xAA, 0xAA)),
                FontSize = 9
            };

            Canvas.SetLeft(label, x);
            Canvas.SetTop(label, 5);
            XAxisCanvas.Children.Add(label);
        }
    }

    private void AddAnomalyMarker(DateTime timestamp, double value)
    {
        var marker = new AnomalyMarker
        {
            Timestamp = timestamp,
            Value = value
        };

        _anomalyMarkers.Add(marker);
        _anomalyCount++;

        // Remove old markers
        var cutoff = _windowStart;
        _anomalyMarkers.RemoveAll(m => m.Timestamp < cutoff);
    }

    private void UpdateAnomalyMarkers()
    {
        AnomalyMarkersCanvas.Children.Clear();

        foreach (var marker in _anomalyMarkers)
        {
            var x = TimeToX(marker.Timestamp);
            var y = ValueToY(marker.Value);

            if (x < 0 || x > _chartWidth) continue;

            var ellipse = new Ellipse
            {
                Width = 10,
                Height = 10,
                Fill = new SolidColorBrush(Color.FromArgb(0x66, 0xFF, 0x44, 0x44)),
                Stroke = new SolidColorBrush(Color.FromRgb(0xFF, 0x44, 0x44)),
                StrokeThickness = 2
            };

            Canvas.SetLeft(ellipse, x - 5);
            Canvas.SetTop(ellipse, y - 5);

            // Add pulsing animation for recent anomalies
            if ((DateTime.UtcNow - marker.Timestamp).TotalSeconds < 5)
            {
                var animation = new DoubleAnimation
                {
                    From = 1.0,
                    To = 0.5,
                    Duration = TimeSpan.FromMilliseconds(500),
                    AutoReverse = true,
                    RepeatBehavior = RepeatBehavior.Forever
                };
                ellipse.BeginAnimation(OpacityProperty, animation);
            }

            AnomalyMarkersCanvas.Children.Add(ellipse);
        }
    }

    #endregion

    #region Helper Methods

    private double TimeToX(DateTime time)
    {
        var totalSeconds = TimeWindowSeconds;
        var elapsed = (time - _windowStart).TotalSeconds;
        return elapsed / totalSeconds * _chartWidth;
    }

    private double ValueToY(double value)
    {
        var normalized = Math.Clamp(value / MaxYValue, 0, 1);
        return _chartHeight * (1 - normalized);
    }

    private DateTime XToTime(double x)
    {
        var fraction = x / _chartWidth;
        return _windowStart.AddSeconds(TimeWindowSeconds * fraction);
    }

    private double YToValue(double y)
    {
        var fraction = 1 - (y / _chartHeight);
        return fraction * MaxYValue;
    }

    private void UpdateCurrentScore(double value)
    {
        _currentScore = value;

        if (AnimateUpdates)
        {
            var animation = new DoubleAnimation
            {
                To = value,
                Duration = TimeSpan.FromMilliseconds(200),
                EasingFunction = new QuadraticEase { EasingMode = EasingMode.EaseOut }
            };
            // Note: For actual animation, we'd need a custom dependency property
        }

        CurrentScoreText.Text = value.ToString("F3");
        UpdateStatusDisplay();
    }

    private void UpdateStatistics()
    {
        if (_dataPoints.Count == 0) return;

        var values = _dataPoints.Select(p => p.Value).ToList();

        _meanScore = values.Average();
        _maxScore = values.Max();

        var variance = values.Average(v => Math.Pow(v - _meanScore, 2));
        _stdDev = Math.Sqrt(variance);

        MeanText.Text = _meanScore.ToString("F3");
        MaxText.Text = _maxScore.ToString("F3");
        StdDevText.Text = _stdDev.ToString("F3");
        AnomalyCountText.Text = _anomalyCount.ToString();
    }

    private void UpdateStatusDisplay()
    {
        if (_currentScore >= DangerThreshold)
        {
            StatusIndicator.Background = new SolidColorBrush(Color.FromArgb(0x33, 0xFF, 0x44, 0x44));
            StatusText.Text = "ANOMALY";
            StatusText.Foreground = new SolidColorBrush(Color.FromRgb(0xFF, 0x44, 0x44));
        }
        else if (_currentScore >= WarningThreshold)
        {
            StatusIndicator.Background = new SolidColorBrush(Color.FromArgb(0x33, 0xFF, 0xFF, 0x00));
            StatusText.Text = "WARNING";
            StatusText.Foreground = new SolidColorBrush(Color.FromRgb(0xFF, 0xFF, 0x00));
        }
        else
        {
            StatusIndicator.Background = new SolidColorBrush(Color.FromArgb(0x33, 0x00, 0xFF, 0x00));
            StatusText.Text = "NORMAL";
            StatusText.Foreground = new SolidColorBrush(Color.FromRgb(0x00, 0xFF, 0x00));
        }
    }

    private void ShowTooltip(Point position)
    {
        if (_dataPoints.Count == 0)
        {
            TooltipBorder.Visibility = Visibility.Collapsed;
            return;
        }

        var time = XToTime(position.X);
        var closestPoint = _dataPoints
            .OrderBy(p => Math.Abs((p.Timestamp - time).TotalMilliseconds))
            .FirstOrDefault();

        if (closestPoint == null)
        {
            TooltipBorder.Visibility = Visibility.Collapsed;
            return;
        }

        TooltipTime.Text = closestPoint.Timestamp.ToString("HH:mm:ss.fff");
        TooltipValue.Text = $"Error: {closestPoint.Value:F4}";

        if (closestPoint.Value >= DangerThreshold)
        {
            TooltipStatus.Text = "ANOMALY DETECTED";
            TooltipStatus.Foreground = new SolidColorBrush(Color.FromRgb(0xFF, 0x44, 0x44));
        }
        else if (closestPoint.Value >= WarningThreshold)
        {
            TooltipStatus.Text = "Elevated";
            TooltipStatus.Foreground = new SolidColorBrush(Color.FromRgb(0xFF, 0xFF, 0x00));
        }
        else
        {
            TooltipStatus.Text = "Normal";
            TooltipStatus.Foreground = new SolidColorBrush(Color.FromRgb(0x00, 0xFF, 0x00));
        }

        // Position tooltip
        var tooltipX = Math.Min(position.X + 10, _chartWidth - 100);
        var tooltipY = Math.Min(position.Y - 50, _chartHeight - 60);

        Canvas.SetLeft(TooltipBorder, tooltipX);
        Canvas.SetTop(TooltipBorder, Math.Max(0, tooltipY));
        TooltipBorder.Visibility = Visibility.Visible;
    }

    #endregion
}

#region Data Models

public class ErrorDataPoint
{
    public DateTime Timestamp { get; set; }
    public double Value { get; set; }
    public bool IsAnomaly { get; set; }
}

public class AnomalyMarker
{
    public DateTime Timestamp { get; set; }
    public double Value { get; set; }
    public string? Description { get; set; }
}

#endregion
