using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using System.Windows.Media.Animation;
using System.Windows.Shapes;

namespace HOPE.Desktop.Controls;

/// <summary>
/// Professional radial gauge control for displaying vehicle telemetry data.
/// Features smooth needle animation, warning/danger zones, and digital readout.
/// </summary>
public partial class GaugeControl : UserControl
{
    private const double START_ANGLE = -135; // Start angle in degrees
    private const double END_ANGLE = 135;    // End angle in degrees
    private const double SWEEP_RANGE = END_ANGLE - START_ANGLE; // 270 degrees total

    #region Dependency Properties

    public static readonly DependencyProperty ValueProperty =
        DependencyProperty.Register(nameof(Value), typeof(double), typeof(GaugeControl),
            new PropertyMetadata(0.0, OnValueChanged));

    public static readonly DependencyProperty MinimumProperty =
        DependencyProperty.Register(nameof(Minimum), typeof(double), typeof(GaugeControl),
            new PropertyMetadata(0.0, OnRangeChanged));

    public static readonly DependencyProperty MaximumProperty =
        DependencyProperty.Register(nameof(Maximum), typeof(double), typeof(GaugeControl),
            new PropertyMetadata(100.0, OnRangeChanged));

    public static readonly DependencyProperty LabelProperty =
        DependencyProperty.Register(nameof(Label), typeof(string), typeof(GaugeControl),
            new PropertyMetadata("Value"));

    public static readonly DependencyProperty UnitProperty =
        DependencyProperty.Register(nameof(Unit), typeof(string), typeof(GaugeControl),
            new PropertyMetadata(string.Empty));

    public static readonly DependencyProperty GaugeColorProperty =
        DependencyProperty.Register(nameof(GaugeColor), typeof(Color), typeof(GaugeControl),
            new PropertyMetadata(Color.FromRgb(0, 230, 118), OnColorChanged));

    public static readonly DependencyProperty NeedleColorProperty =
        DependencyProperty.Register(nameof(NeedleColor), typeof(Brush), typeof(GaugeControl),
            new PropertyMetadata(new SolidColorBrush(Color.FromRgb(255, 82, 82))));

    public static readonly DependencyProperty WarningThresholdProperty =
        DependencyProperty.Register(nameof(WarningThreshold), typeof(double), typeof(GaugeControl),
            new PropertyMetadata(double.NaN, OnThresholdChanged));

    public static readonly DependencyProperty DangerThresholdProperty =
        DependencyProperty.Register(nameof(DangerThreshold), typeof(double), typeof(GaugeControl),
            new PropertyMetadata(double.NaN, OnThresholdChanged));

    public static readonly DependencyProperty WarningZoneColorProperty =
        DependencyProperty.Register(nameof(WarningZoneColor), typeof(Brush), typeof(GaugeControl),
            new PropertyMetadata(new SolidColorBrush(Color.FromRgb(255, 193, 7))));

    public static readonly DependencyProperty DangerZoneColorProperty =
        DependencyProperty.Register(nameof(DangerZoneColor), typeof(Brush), typeof(GaugeControl),
            new PropertyMetadata(new SolidColorBrush(Color.FromRgb(244, 67, 54))));

    public static readonly DependencyProperty AnimationDurationProperty =
        DependencyProperty.Register(nameof(AnimationDuration), typeof(TimeSpan), typeof(GaugeControl),
            new PropertyMetadata(TimeSpan.FromMilliseconds(150)));

    public static readonly DependencyProperty ShowTicksProperty =
        DependencyProperty.Register(nameof(ShowTicks), typeof(bool), typeof(GaugeControl),
            new PropertyMetadata(true, OnTicksChanged));

    public static readonly DependencyProperty MajorTickCountProperty =
        DependencyProperty.Register(nameof(MajorTickCount), typeof(int), typeof(GaugeControl),
            new PropertyMetadata(10, OnTicksChanged));

    public static readonly DependencyProperty MinorTickCountProperty =
        DependencyProperty.Register(nameof(MinorTickCount), typeof(int), typeof(GaugeControl),
            new PropertyMetadata(5, OnTicksChanged));

    public static readonly DependencyProperty ValueFormatProperty =
        DependencyProperty.Register(nameof(ValueFormat), typeof(string), typeof(GaugeControl),
            new PropertyMetadata("{0:F1}"));

    #endregion

    #region Properties

    /// <summary>Current value displayed on the gauge</summary>
    public double Value
    {
        get => (double)GetValue(ValueProperty);
        set => SetValue(ValueProperty, value);
    }

    /// <summary>Minimum value of the gauge scale</summary>
    public double Minimum
    {
        get => (double)GetValue(MinimumProperty);
        set => SetValue(MinimumProperty, value);
    }

    /// <summary>Maximum value of the gauge scale</summary>
    public double Maximum
    {
        get => (double)GetValue(MaximumProperty);
        set => SetValue(MaximumProperty, value);
    }

    /// <summary>Label text displayed at the top of the gauge</summary>
    public string Label
    {
        get => (string)GetValue(LabelProperty);
        set => SetValue(LabelProperty, value);
    }

    /// <summary>Unit text displayed below the value</summary>
    public string Unit
    {
        get => (string)GetValue(UnitProperty);
        set => SetValue(UnitProperty, value);
    }

    /// <summary>Primary gauge color for the value arc and digital readout</summary>
    public Color GaugeColor
    {
        get => (Color)GetValue(GaugeColorProperty);
        set => SetValue(GaugeColorProperty, value);
    }

    /// <summary>Needle color</summary>
    public Brush NeedleColor
    {
        get => (Brush)GetValue(NeedleColorProperty);
        set => SetValue(NeedleColorProperty, value);
    }

    /// <summary>Value at which warning zone begins (NaN to disable)</summary>
    public double WarningThreshold
    {
        get => (double)GetValue(WarningThresholdProperty);
        set => SetValue(WarningThresholdProperty, value);
    }

    /// <summary>Value at which danger zone begins (NaN to disable)</summary>
    public double DangerThreshold
    {
        get => (double)GetValue(DangerThresholdProperty);
        set => SetValue(DangerThresholdProperty, value);
    }

    /// <summary>Color for the warning zone arc</summary>
    public Brush WarningZoneColor
    {
        get => (Brush)GetValue(WarningZoneColorProperty);
        set => SetValue(WarningZoneColorProperty, value);
    }

    /// <summary>Color for the danger zone arc</summary>
    public Brush DangerZoneColor
    {
        get => (Brush)GetValue(DangerZoneColorProperty);
        set => SetValue(DangerZoneColorProperty, value);
    }

    /// <summary>Duration of needle animation</summary>
    public TimeSpan AnimationDuration
    {
        get => (TimeSpan)GetValue(AnimationDurationProperty);
        set => SetValue(AnimationDurationProperty, value);
    }

    /// <summary>Whether to display tick marks</summary>
    public bool ShowTicks
    {
        get => (bool)GetValue(ShowTicksProperty);
        set => SetValue(ShowTicksProperty, value);
    }

    /// <summary>Number of major tick marks</summary>
    public int MajorTickCount
    {
        get => (int)GetValue(MajorTickCountProperty);
        set => SetValue(MajorTickCountProperty, value);
    }

    /// <summary>Number of minor ticks between major ticks</summary>
    public int MinorTickCount
    {
        get => (int)GetValue(MinorTickCountProperty);
        set => SetValue(MinorTickCountProperty, value);
    }

    /// <summary>Format string for the digital value display</summary>
    public string ValueFormat
    {
        get => (string)GetValue(ValueFormatProperty);
        set => SetValue(ValueFormatProperty, value);
    }

    #endregion

    public GaugeControl()
    {
        InitializeComponent();
        Loaded += OnLoaded;
    }

    private void OnLoaded(object sender, RoutedEventArgs e)
    {
        DrawTicks();
        DrawZoneArcs();
        UpdateNeedle(false);
        UpdateValueArc();
    }

    private static void OnValueChanged(DependencyObject d, DependencyPropertyChangedEventArgs e)
    {
        if (d is GaugeControl gauge && gauge.IsLoaded)
        {
            gauge.UpdateNeedle(true);
            gauge.UpdateValueArc();
            gauge.UpdateValueColor();
        }
    }

    private static void OnRangeChanged(DependencyObject d, DependencyPropertyChangedEventArgs e)
    {
        if (d is GaugeControl gauge && gauge.IsLoaded)
        {
            gauge.DrawTicks();
            gauge.DrawZoneArcs();
            gauge.UpdateNeedle(false);
            gauge.UpdateValueArc();
        }
    }

    private static void OnColorChanged(DependencyObject d, DependencyPropertyChangedEventArgs e)
    {
        if (d is GaugeControl gauge && gauge.IsLoaded)
        {
            gauge.UpdateValueArc();
        }
    }

    private static void OnThresholdChanged(DependencyObject d, DependencyPropertyChangedEventArgs e)
    {
        if (d is GaugeControl gauge && gauge.IsLoaded)
        {
            gauge.DrawZoneArcs();
            gauge.UpdateValueColor();
        }
    }

    private static void OnTicksChanged(DependencyObject d, DependencyPropertyChangedEventArgs e)
    {
        if (d is GaugeControl gauge && gauge.IsLoaded)
        {
            gauge.DrawTicks();
        }
    }

    private void UpdateNeedle(bool animate)
    {
        double normalizedValue = Math.Clamp((Value - Minimum) / (Maximum - Minimum), 0, 1);
        double targetAngle = START_ANGLE + (normalizedValue * SWEEP_RANGE);

        if (animate && AnimationDuration > TimeSpan.Zero)
        {
            var animation = new DoubleAnimation
            {
                To = targetAngle,
                Duration = AnimationDuration,
                EasingFunction = new QuadraticEase { EasingMode = EasingMode.EaseOut }
            };
            NeedleRotation.BeginAnimation(RotateTransform.AngleProperty, animation);
        }
        else
        {
            NeedleRotation.Angle = targetAngle;
        }
    }

    private void UpdateValueArc()
    {
        double normalizedValue = Math.Clamp((Value - Minimum) / (Maximum - Minimum), 0, 1);

        const double radius = 80;
        const double centerX = 100;
        const double centerY = 100;

        double startAngleRad = START_ANGLE * Math.PI / 180;
        double endAngleRad = (START_ANGLE + (normalizedValue * SWEEP_RANGE)) * Math.PI / 180;

        var geometry = CreateArcGeometry(centerX, centerY, radius, startAngleRad, endAngleRad);
        ValueArc.Data = geometry;
    }

    private void UpdateValueColor()
    {
        // Change gauge color based on thresholds
        Color effectiveColor = GaugeColor;

        if (!double.IsNaN(DangerThreshold) && Value >= DangerThreshold)
        {
            if (DangerZoneColor is SolidColorBrush dangerBrush)
                effectiveColor = dangerBrush.Color;
        }
        else if (!double.IsNaN(WarningThreshold) && Value >= WarningThreshold)
        {
            if (WarningZoneColor is SolidColorBrush warningBrush)
                effectiveColor = warningBrush.Color;
        }

        ValueArc.Stroke = new SolidColorBrush(effectiveColor);
        ValueText.Foreground = new SolidColorBrush(effectiveColor);
    }

    private void DrawZoneArcs()
    {
        const double radius = 80;
        const double centerX = 100;
        const double centerY = 100;

        // Warning zone
        if (!double.IsNaN(WarningThreshold) && !double.IsNaN(DangerThreshold))
        {
            double warningStart = Math.Clamp((WarningThreshold - Minimum) / (Maximum - Minimum), 0, 1);
            double warningEnd = Math.Clamp((DangerThreshold - Minimum) / (Maximum - Minimum), 0, 1);

            double startAngleRad = (START_ANGLE + (warningStart * SWEEP_RANGE)) * Math.PI / 180;
            double endAngleRad = (START_ANGLE + (warningEnd * SWEEP_RANGE)) * Math.PI / 180;

            WarningArc.Data = CreateArcGeometry(centerX, centerY, radius, startAngleRad, endAngleRad);
        }
        else if (!double.IsNaN(WarningThreshold))
        {
            double warningStart = Math.Clamp((WarningThreshold - Minimum) / (Maximum - Minimum), 0, 1);

            double startAngleRad = (START_ANGLE + (warningStart * SWEEP_RANGE)) * Math.PI / 180;
            double endAngleRad = END_ANGLE * Math.PI / 180;

            WarningArc.Data = CreateArcGeometry(centerX, centerY, radius, startAngleRad, endAngleRad);
        }

        // Danger zone
        if (!double.IsNaN(DangerThreshold))
        {
            double dangerStart = Math.Clamp((DangerThreshold - Minimum) / (Maximum - Minimum), 0, 1);

            double startAngleRad = (START_ANGLE + (dangerStart * SWEEP_RANGE)) * Math.PI / 180;
            double endAngleRad = END_ANGLE * Math.PI / 180;

            DangerArc.Data = CreateArcGeometry(centerX, centerY, radius, startAngleRad, endAngleRad);
        }
    }

    private static PathGeometry CreateArcGeometry(double centerX, double centerY, double radius,
        double startAngleRad, double endAngleRad)
    {
        double startX = centerX + radius * Math.Cos(startAngleRad);
        double startY = centerY + radius * Math.Sin(startAngleRad);
        double endX = centerX + radius * Math.Cos(endAngleRad);
        double endY = centerY + radius * Math.Sin(endAngleRad);

        double angleDiff = endAngleRad - startAngleRad;
        bool isLargeArc = Math.Abs(angleDiff) > Math.PI;

        var figure = new PathFigure
        {
            StartPoint = new Point(startX, startY),
            IsClosed = false
        };

        figure.Segments.Add(new ArcSegment
        {
            Point = new Point(endX, endY),
            Size = new Size(radius, radius),
            IsLargeArc = isLargeArc,
            SweepDirection = SweepDirection.Clockwise
        });

        var geometry = new PathGeometry();
        geometry.Figures.Add(figure);
        return geometry;
    }

    private void DrawTicks()
    {
        TickCanvas.Children.Clear();

        if (!ShowTicks) return;

        const double outerRadius = 88;
        const double majorInnerRadius = 75;
        const double minorInnerRadius = 80;
        const double centerX = 100;
        const double centerY = 100;

        // Draw major ticks
        for (int i = 0; i <= MajorTickCount; i++)
        {
            double normalizedPos = (double)i / MajorTickCount;
            double angle = (START_ANGLE + (normalizedPos * SWEEP_RANGE)) * Math.PI / 180;

            double outerX = centerX + outerRadius * Math.Cos(angle);
            double outerY = centerY + outerRadius * Math.Sin(angle);
            double innerX = centerX + majorInnerRadius * Math.Cos(angle);
            double innerY = centerY + majorInnerRadius * Math.Sin(angle);

            var line = new Line
            {
                X1 = innerX,
                Y1 = innerY,
                X2 = outerX,
                Y2 = outerY,
                Stroke = new SolidColorBrush(Color.FromRgb(170, 170, 170)),
                StrokeThickness = 2
            };
            TickCanvas.Children.Add(line);

            // Draw minor ticks (except after last major tick)
            if (i < MajorTickCount && MinorTickCount > 0)
            {
                for (int j = 1; j <= MinorTickCount; j++)
                {
                    double minorNormalizedPos = normalizedPos + ((double)j / (MajorTickCount * (MinorTickCount + 1)));
                    double minorAngle = (START_ANGLE + (minorNormalizedPos * SWEEP_RANGE)) * Math.PI / 180;

                    double minorOuterX = centerX + outerRadius * Math.Cos(minorAngle);
                    double minorOuterY = centerY + outerRadius * Math.Sin(minorAngle);
                    double minorInnerX = centerX + minorInnerRadius * Math.Cos(minorAngle);
                    double minorInnerY = centerY + minorInnerRadius * Math.Sin(minorAngle);

                    var minorLine = new Line
                    {
                        X1 = minorInnerX,
                        Y1 = minorInnerY,
                        X2 = minorOuterX,
                        Y2 = minorOuterY,
                        Stroke = new SolidColorBrush(Color.FromRgb(102, 102, 102)),
                        StrokeThickness = 1
                    };
                    TickCanvas.Children.Add(minorLine);
                }
            }
        }
    }
}
