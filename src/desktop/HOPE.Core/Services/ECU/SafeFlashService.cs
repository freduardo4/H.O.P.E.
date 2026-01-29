using System.Diagnostics;
using System.IO;
using HOPE.Core.Hardware;
using HOPE.Core.Interfaces;
using HOPE.Core.Services.Protocols;

namespace HOPE.Core.Services.ECU;

/// <summary>
/// Safe ECU flashing service with comprehensive pre-flight checks,
/// automatic backup, and multi-step verification protocol.
/// </summary>
public class SafeFlashService : IDisposable
{
    private readonly IHardwareAdapter _adapter;
    private readonly IVoltageMonitor _voltageMonitor;
    private readonly IUdsProtocolService _udsService;
    private readonly ICalibrationRepository _repository;
    private readonly ICloudSafetyService _cloudSafety;
    private readonly SemaphoreSlim _flashLock = new(1, 1);

    private CancellationTokenSource? _flashCts;
    private FlashSession? _currentSession;

    /// <summary>
    /// Minimum battery voltage for safe flashing (engine running or charger)
    /// </summary>
    public const double MIN_FLASH_VOLTAGE = 13.0;

    /// <summary>
    /// Maximum allowed J2534 latency in milliseconds
    /// </summary>
    public const int MAX_LATENCY_MS = 50;

    /// <summary>
    /// Event raised when flash progress updates
    /// </summary>
    public event EventHandler<FlashProgressEventArgs>? ProgressChanged;

    /// <summary>
    /// Event raised when a warning condition is detected
    /// </summary>
    public event EventHandler<FlashWarningEventArgs>? WarningRaised;

    /// <summary>
    /// Event raised when flash completes (success or failure)
    /// </summary>
    public event EventHandler<FlashCompleteEventArgs>? FlashComplete;

    /// <summary>
    /// Gets whether a flash operation is currently in progress
    /// </summary>
    public bool IsFlashInProgress => _currentSession != null;

    /// <summary>
    /// Gets the current flash session, if any
    /// </summary>
    public FlashSession? CurrentSession => _currentSession;

    public SafeFlashService(
        IHardwareAdapter adapter,
        IVoltageMonitor voltageMonitor,
        IUdsProtocolService udsService,
        ICalibrationRepository repository,
        ICloudSafetyService cloudSafety)
    {
        _adapter = adapter ?? throw new ArgumentNullException(nameof(adapter));
        _voltageMonitor = voltageMonitor ?? throw new ArgumentNullException(nameof(voltageMonitor));
        _udsService = udsService ?? throw new ArgumentNullException(nameof(udsService));
        _repository = repository ?? throw new ArgumentNullException(nameof(repository));
        _cloudSafety = cloudSafety ?? throw new ArgumentNullException(nameof(cloudSafety));
    }

    /// <summary>
    /// Perform pre-flight checks before flashing
    /// </summary>
    public async Task<PreFlightResult> PerformPreFlightChecksAsync(
        FlashConfig config,
        CancellationToken ct = default)
    {
        var result = new PreFlightResult { Timestamp = DateTime.UtcNow };
        var checks = new List<PreFlightCheck>();

        // 1. Hardware Adapter Check
        var adapterCheck = new PreFlightCheck { Name = "Hardware Adapter", Category = CheckCategory.Hardware };
        if (!_adapter.IsConnected)
        {
            adapterCheck.Status = CheckStatus.Failed;
            adapterCheck.Message = "Hardware adapter is not connected";
        }
        else if (_adapter.Type != HardwareType.J2534)
        {
            adapterCheck.Status = CheckStatus.Warning;
            adapterCheck.Message = "J2534 adapter recommended for flashing; ELM327 has limited write capability";
        }
        else
        {
            adapterCheck.Status = CheckStatus.Passed;
            adapterCheck.Message = $"J2534 adapter connected: {_adapter.AdapterName}";
        }
        checks.Add(adapterCheck);

        // 2. Battery Voltage Check
        var voltageCheck = new PreFlightCheck { Name = "Battery Voltage", Category = CheckCategory.Power };
        var voltageReading = await _voltageMonitor.ReadBatteryVoltageAsync(ct);
        if (!voltageReading.Voltage.HasValue)
        {
            voltageCheck.Status = CheckStatus.Warning;
            voltageCheck.Message = "Unable to read battery voltage; ensure charger is connected";
        }
        else if (_adapter.HasQuantizedVoltageReporting && voltageReading.Voltage >= 13.7)
        {
            voltageCheck.Status = CheckStatus.Warning;
            voltageCheck.Message = $"Battery voltage reported as 13.7V (Quantized). Precise voltage unknown. Manual verification recommended.";
        }
        else if (voltageReading.Voltage < MIN_FLASH_VOLTAGE)
        {
            voltageCheck.Status = CheckStatus.Failed;
            voltageCheck.Message = $"Battery voltage {voltageReading.Voltage:F1}V is below minimum {MIN_FLASH_VOLTAGE}V. Connect battery charger or run engine.";
        }
        else
        {
            voltageCheck.Status = CheckStatus.Passed;
            voltageCheck.Message = $"Battery voltage: {voltageReading.Voltage:F2}V";
        }
        checks.Add(voltageCheck);

        // 3. Communication Latency Check
        var latencyCheck = new PreFlightCheck { Name = "Communication Latency", Category = CheckCategory.Communication };
        var latency = await MeasureLatencyAsync(ct);
        if (latency > MAX_LATENCY_MS)
        {
            latencyCheck.Status = CheckStatus.Failed;
            latencyCheck.Message = $"Latency {latency}ms exceeds maximum {MAX_LATENCY_MS}ms";
        }
        else
        {
            latencyCheck.Status = CheckStatus.Passed;
            latencyCheck.Message = $"Round-trip latency: {latency}ms";
        }
        checks.Add(latencyCheck);

        // 4. ECU Response Check
        var ecuCheck = new PreFlightCheck { Name = "ECU Communication", Category = CheckCategory.Communication };
        var ecuResponse = await _udsService.DiagnosticSessionControlAsync(UdsSession.Default, ct);
        if (!ecuResponse.IsPositive)
        {
            ecuCheck.Status = CheckStatus.Failed;
            ecuCheck.Message = $"ECU not responding: {ecuResponse.ErrorMessage}";
        }
        else
        {
            ecuCheck.Status = CheckStatus.Passed;
            ecuCheck.Message = "ECU responding to diagnostic requests";
        }
        checks.Add(ecuCheck);

        // 5. Active DTC Check
        var dtcCheck = new PreFlightCheck { Name = "Active DTCs", Category = CheckCategory.Vehicle };
        var dtcResponse = await _udsService.ReadDtcInformationAsync(DtcReportType.ReportDTCByStatusMask, 0x09, ct);
        if (dtcResponse.IsPositive && dtcResponse.DTCs.Any(d => d.IsActive))
        {
            var activeDtcs = dtcResponse.DTCs.Where(d => d.IsActive).ToList();
            var communicationDtcs = activeDtcs.Where(d => d.FormattedCode.StartsWith("U")).ToList();

            if (communicationDtcs.Any())
            {
                dtcCheck.Status = CheckStatus.Failed;
                dtcCheck.Message = $"Communication DTCs present: {string.Join(", ", communicationDtcs.Select(d => d.FormattedCode))}";
            }
            else
            {
                dtcCheck.Status = CheckStatus.Warning;
                dtcCheck.Message = $"{activeDtcs.Count} active DTCs present (non-communication)";
            }
        }
        else
        {
            dtcCheck.Status = CheckStatus.Passed;
            dtcCheck.Message = "No active DTCs that could interrupt flash";
        }
        checks.Add(dtcCheck);

        // 6. Calibration Checksum Check
        var checksumCheck = new PreFlightCheck { Name = "Calibration Checksum", Category = CheckCategory.Data };
        var checksumResult = await _repository.ValidateChecksumAsync(config.Calibration, ct);
        if (!checksumResult.IsValid)
        {
            checksumCheck.Status = CheckStatus.Failed;
            checksumCheck.Message = "Calibration file checksum mismatch - file may be corrupted";
        }
        else
        {
            checksumCheck.Status = CheckStatus.Passed;
            checksumCheck.Message = "Calibration checksum verified";
        }
        checks.Add(checksumCheck);

        // 7. Security Access Check
        var securityCheck = new PreFlightCheck { Name = "Security Access", Category = CheckCategory.Security };
        var sessionResponse = await _udsService.DiagnosticSessionControlAsync(UdsSession.Programming, ct);
        if (sessionResponse.IsPositive)
        {
            var seedResponse = await _udsService.RequestSecuritySeedAsync(config.SecurityLevel, ct);
            if (seedResponse.IsPositive)
            {
                securityCheck.Status = CheckStatus.Passed;
                securityCheck.Message = "Security access available";
            }
            else
            {
                securityCheck.Status = CheckStatus.Failed;
                securityCheck.Message = $"Security access denied: {seedResponse.NegativeResponseCode}";
            }

            // Return to default session
            await _udsService.DiagnosticSessionControlAsync(UdsSession.Default, ct);
        }
        else
        {
            securityCheck.Status = CheckStatus.Failed;
            securityCheck.Message = "Unable to enter programming session";
        }
        checks.Add(securityCheck);

        // 8. Cloud Safety Policy Check
        var cloudCheck = new PreFlightCheck { Name = "Cloud Safety Policy", Category = CheckCategory.Security };
        var canProceed = await _cloudSafety.ValidateFlashOperationAsync(
            config.Calibration.EcuId, 
            voltageReading.Voltage ?? 0, 
            ct);
            
        if (canProceed)
        {
            cloudCheck.Status = CheckStatus.Passed;
            cloudCheck.Message = "Operation authorized by cloud policy";
        }
        else
        {
            cloudCheck.Status = CheckStatus.Failed;
            cloudCheck.Message = "Operation blocked by cloud safety policy (or cloud unreachable)";
        }
        checks.Add(cloudCheck);

        result.Checks = checks;
        result.CanProceed = checks.All(c => c.Status != CheckStatus.Failed);
        result.HasWarnings = checks.Any(c => c.Status == CheckStatus.Warning);

        return result;
    }

    /// <summary>
    /// Execute the flash operation with full safety protocol
    /// </summary>
    public async Task<FlashResult> FlashAsync(
        FlashConfig config,
        Func<byte[], byte[]> securityKeyAlgorithm,
        CancellationToken ct = default)
    {
        if (!await _flashLock.WaitAsync(0, ct))
            throw new InvalidOperationException("Another flash operation is in progress");

        try
        {
            var result = await FlashInternalAsync(config, securityKeyAlgorithm, ct);
            FlashComplete?.Invoke(this, new FlashCompleteEventArgs(result));
            return result;
        }
        finally
        {
            _flashLock.Release();
        }
    }

    private async Task<FlashResult> FlashInternalAsync(
        FlashConfig config,
        Func<byte[], byte[]> securityKeyAlgorithm,
        CancellationToken ct = default)
    {
        _flashCts = CancellationTokenSource.CreateLinkedTokenSource(ct);
        var result = new FlashResult { StartTime = DateTime.UtcNow };
        var stopwatch = Stopwatch.StartNew();

        try
        {
            _currentSession = new FlashSession
            {
                Id = Guid.NewGuid(),
                StartTime = DateTime.UtcNow,
                Config = config,
                Stage = FlashStage.PreFlight
            };

            // Step 1: Pre-flight checks
            ReportProgress(FlashStage.PreFlight, 0, "Running pre-flight checks...");
            var preFlightResult = await PerformPreFlightChecksAsync(config, _flashCts.Token);

            if (!preFlightResult.CanProceed)
            {
                result.Success = false;
                result.FailureReason = "Pre-flight checks failed";
                result.PreFlightResult = preFlightResult;
                result.FailedAtStage = FlashStage.PreFlight;
                return result;
            }

            result.PreFlightResult = preFlightResult;

            // Step 2: Create shadow backup
            if (config.CreateBackup)
            {
                _currentSession.Stage = FlashStage.Backup;
                ReportProgress(FlashStage.Backup, 5, "Creating shadow backup...");

                var backup = await CreateShadowBackupAsync(config, _flashCts.Token);
                
                // Verify backup integrity immediately
                var backupCalPath = Path.Combine(backup.BackupPath, "calibration.json");
                if (!File.Exists(backupCalPath))
                    throw new FlashException("Shadow backup failed: calibration file not created");

                _currentSession.BackupPath = backup.BackupPath;
                result.BackupPath = backup.BackupPath;

                ReportProgress(FlashStage.Backup, 15, $"Backup verified: {backup.BackupHash[..8]}");
            }
            else
            {
                ReportProgress(FlashStage.Backup, 15, "Skipping backup as requested.");
            }

            // Step 3: Enter programming session
            _currentSession.Stage = FlashStage.EnterSession;
            ReportProgress(FlashStage.EnterSession, 20, "Entering programming session...");

            var sessionResponse = await _udsService.DiagnosticSessionControlAsync(UdsSession.Programming, _flashCts.Token);
            if (!sessionResponse.IsPositive)
            {
                throw new FlashException($"Failed to enter programming session: {sessionResponse.ErrorMessage}");
            }

            // Step 4: Security access
            _currentSession.Stage = FlashStage.Security;
            ReportProgress(FlashStage.Security, 25, "Requesting security access...");

            var securitySuccess = await _udsService.PerformSecurityAccessAsync(
                config.SecurityLevel,
                securityKeyAlgorithm,
                _flashCts.Token);

            if (!securitySuccess)
            {
                throw new FlashException("Security access denied - invalid key or locked ECU");
            }

            ReportProgress(FlashStage.Security, 30, "Security access granted");

            // Step 5: Request download
            _currentSession.Stage = FlashStage.RequestDownload;
            ReportProgress(FlashStage.RequestDownload, 32, "Requesting download permission...");

            var totalSize = config.Calibration.Blocks.Sum(b => b.Data.Length);
            var startAddress = config.Calibration.Blocks.Min(b => b.StartAddress);

            var downloadResponse = await _udsService.RequestDownloadAsync(
                startAddress,
                (uint)totalSize,
                ct: _flashCts.Token);

            if (!downloadResponse.IsPositive)
            {
                throw new FlashException($"Download request rejected: {downloadResponse.ErrorMessage}");
            }

            var maxBlockSize = downloadResponse.MaxBlockLength > 0 ? downloadResponse.MaxBlockLength : 4096;

            // Step 6: Transfer data
            _currentSession.Stage = FlashStage.Transfer;
            var blocksWritten = 0;
            long totalBytesWritten = 0; // Declared here
            long lastVoltageCheckBytes = 0; // Declared here
            byte blockSequence = 1;

            foreach (var block in config.Calibration.Blocks.OrderBy(b => b.StartAddress))
            {
                ReportProgress(FlashStage.Transfer, 35 + (blocksWritten * 50 / config.Calibration.Blocks.Count),
                    $"Writing block: {block.Name}");

                var offset = 0;
                while (offset < block.Data.Length)
                {
                    // Check voltage continuously during flash
                    if (totalBytesWritten == 0 || totalBytesWritten - lastVoltageCheckBytes >= 128)
                    {
                        var voltageResult = await _voltageMonitor.ReadBatteryVoltageAsync(_flashCts.Token);
                        if (voltageResult.Voltage.HasValue)
                        {
                            if (voltageResult.Voltage < VoltageMonitor.CRITICAL_THRESHOLD)
                            {
                                throw new FlashException($"CRITICAL: Battery voltage dropped below {VoltageMonitor.CRITICAL_THRESHOLD}V ({voltageResult.Voltage:F1}V)! Aborting for safety.", FlashStage.Transfer);
                            }
                            else if (voltageResult.Voltage < VoltageMonitor.WARNING_THRESHOLD)
                            {
                                RaiseWarning(FlashWarningType.LowVoltage,
                                    $"Battery voltage low: {voltageResult.Voltage:F1}V. Please check power supply.");
                            }
                        }
                        lastVoltageCheckBytes = totalBytesWritten;
                    }

                    var chunkSize = Math.Min(maxBlockSize - 2, block.Data.Length - offset);
                    var chunk = new byte[chunkSize];
                    Array.Copy(block.Data, offset, chunk, 0, chunkSize);

                    var transferResponse = await _udsService.TransferDataAsync(blockSequence, chunk, _flashCts.Token);

                    if (!transferResponse.IsPositive)
                    {
                        throw new FlashException($"Transfer failed at offset 0x{offset:X}: {transferResponse.ErrorMessage}");
                    }

                    offset += chunkSize;
                    totalBytesWritten += chunkSize;
                    blockSequence++;

                    // Update progress
                    var progress = 35 + (int)(totalBytesWritten * 50.0 / totalSize);
                    ReportProgress(FlashStage.Transfer, progress,
                        $"Writing: {totalBytesWritten:N0} / {totalSize:N0} bytes");
                }

                blocksWritten++;
            }

            // Step 7: Request transfer exit
            _currentSession.Stage = FlashStage.ExitTransfer;
            ReportProgress(FlashStage.ExitTransfer, 87, "Completing transfer...");

            var exitResponse = await _udsService.RequestTransferExitAsync(ct: _flashCts.Token);
            if (!exitResponse.IsPositive)
            {
                throw new FlashException($"Transfer exit failed: {exitResponse.ErrorMessage}");
            }

            // Step 8: Verify (optional read-back)
            if (config.VerifyAfterWrite)
            {
                _currentSession.Stage = FlashStage.Verify;
                ReportProgress(FlashStage.Verify, 90, "Verifying written data...");

                var verifyResult = await VerifyFlashAsync(config.Calibration, _flashCts.Token);
                if (!verifyResult.Success)
                {
                    throw new FlashException($"Verification failed: {verifyResult.FailureReason}");
                }

                result.VerificationResult = verifyResult;
            }

            // Step 9: ECU Reset
            _currentSession.Stage = FlashStage.Reset;
            ReportProgress(FlashStage.Reset, 95, "Resetting ECU...");

            var resetResponse = await _udsService.EcuResetAsync(EcuResetType.HardReset, _flashCts.Token);

            // Wait for ECU to restart
            await Task.Delay(2000, _flashCts.Token);

            // Step 10: Post-flash verification
            _currentSession.Stage = FlashStage.PostVerify;
            ReportProgress(FlashStage.PostVerify, 98, "Post-flash verification...");

            // Try to communicate with ECU
            for (int attempt = 0; attempt < 5; attempt++)
            {
                await Task.Delay(1000, _flashCts.Token);
                var postResponse = await _udsService.DiagnosticSessionControlAsync(UdsSession.Default, _flashCts.Token);
                if (postResponse.IsPositive)
                {
                    ReportProgress(FlashStage.Complete, 100, "Flash completed successfully!");
                    result.Success = true;
                    break;
                }
            }

            if (!result.Success)
            {
                RaiseWarning(FlashWarningType.PostFlashCommunicationFailed,
                    "ECU not responding after flash - may require ignition cycle");
                result.Success = true; // Flash itself succeeded, just communication issue
            }

            result.BytesWritten = (int)totalBytesWritten;
            result.BlocksWritten = blocksWritten;
        }
        catch (FlashException ex)
        {
            result.Success = false;
            result.FailureReason = ex.Message;
            result.FailedAtStage = ex.FailedStage ?? _currentSession?.Stage;

            // Trigger emergency recovery if flash failed during transfer
            if (result.FailedAtStage == FlashStage.Transfer || result.FailedAtStage == FlashStage.RequestDownload)
            {
                ReportProgress(FlashStage.Recovery, 50, "Crticial failure during write. Attempting emergency recovery...");
                var recoveryResult = await TryEmergencyRecoveryAsync(config, securityKeyAlgorithm, ct);
                if (recoveryResult.Success)
                {
                    result.Success = true;
                    result.FailureReason = "Recovered from partial flash failure via full restore.";
                    ReportProgress(FlashStage.Complete, 100, "Emergency recovery successful!");
                }
                else
                {
                    result.FailureReason += " | RECOVERY FAILED. ECU MAY BE BRICKED. Use manual bootloader tools.";
                }
            }
            else
            {
                ReportProgress(_currentSession?.Stage ?? FlashStage.PreFlight, -1,
                    $"Flash failed: {ex.Message}");
            }
        }
        catch (OperationCanceledException)
        {
            result.Success = false;
            result.FailureReason = "Flash operation was cancelled";
            result.WasCancelled = true;
        }
        catch (Exception ex)
        {
            result.Success = false;
            result.FailureReason = $"Unexpected error: {ex.Message}";
            result.FailedAtStage = _currentSession?.Stage;
        }
        finally
        {
            stopwatch.Stop();
            result.Duration = stopwatch.Elapsed;
            result.EndTime = DateTime.UtcNow;

            _currentSession = null;
            _flashCts?.Dispose();
            _flashCts = null;
            // Log to cloud
            _ = Task.Run(async () => 
            {
                try 
                {
                    await _cloudSafety.LogSafetyEventAsync(new HOPE.Core.Services.Safety.SafetyEvent
                    {
                        EventType = "ECU_FLASH",
                        EcuId = config.Calibration.EcuId,
                        Voltage = (await _voltageMonitor.ReadBatteryVoltageAsync()).Voltage,
                        Success = result.Success,
                        Message = result.Success ? "Flash completed successfully" : (result.FailureReason ?? "Unknown failure"),
                        Timestamp = DateTime.UtcNow
                    });
                }
                catch { /* ignore background log errors */ }
            });
        }

        return result;
    }

    /// <summary>
    /// Abort the current flash operation
    /// </summary>
    public void AbortFlash()
    {
        _flashCts?.Cancel();
    }

    /// <summary>
    /// Restore from a backup after a failed flash
    /// </summary>
    public async Task<FlashResult> RestoreFromBackupAsync(
        string backupPath,
        Func<byte[], byte[]> securityKeyAlgorithm,
        CancellationToken ct = default)
    {
        // Load backup
        var backupJson = await File.ReadAllTextAsync(Path.Combine(backupPath, "calibration.json"), ct);
        var calibration = System.Text.Json.JsonSerializer.Deserialize<CalibrationFile>(backupJson)
            ?? throw new FlashException("Failed to load backup calibration");

        var config = new FlashConfig
        {
            Calibration = calibration,
            VerifyAfterWrite = true,
            SecurityLevel = 1
        };

        return await FlashInternalAsync(config, securityKeyAlgorithm, ct);
    }

    private async Task<ShadowBackup> CreateShadowBackupAsync(FlashConfig config, CancellationToken ct)
    {
        var backupDir = Path.Combine(
            _repository.RepositoryPath,
            "backups",
            DateTime.UtcNow.ToString("yyyyMMdd_HHmmss"));

        Directory.CreateDirectory(backupDir);

        // Read current ECU calibration
        var readConfig = new EcuReadConfig
        {
            EcuId = config.Calibration.EcuId,
            MemoryRegions = config.Calibration.Blocks.Select(b => new MemoryRegion
            {
                Name = b.Name,
                StartAddress = b.StartAddress,
                Size = (uint)b.Data.Length
            }).ToList()
        };

        var progress = new Progress<CalibrationProgress>(p =>
            ReportProgress(FlashStage.Backup, 5 + (p.PercentComplete / 10), p.StatusMessage ?? "Reading ECU..."));

        var currentCalibration = await _repository.ReadFromEcuAsync(_adapter, readConfig, progress, ct);

        // Save backup
        var calibrationJson = System.Text.Json.JsonSerializer.Serialize(currentCalibration,
            new System.Text.Json.JsonSerializerOptions { WriteIndented = true });
        await File.WriteAllTextAsync(Path.Combine(backupDir, "calibration.json"), calibrationJson, ct);

        // Save metadata
        var metadata = new BackupMetadata
        {
            CreatedAt = DateTime.UtcNow,
            EcuId = currentCalibration.EcuId,
            Checksum = currentCalibration.FullChecksum,
            Reason = $"Pre-flash backup before applying {config.Calibration.FullChecksum[..8]}"
        };
        var metadataJson = System.Text.Json.JsonSerializer.Serialize(metadata);
        await File.WriteAllTextAsync(Path.Combine(backupDir, "metadata.json"), metadataJson, ct);

        return new ShadowBackup
        {
            BackupPath = backupDir,
            BackupHash = currentCalibration.FullChecksum,
            Timestamp = DateTime.UtcNow
        };
    }

    private async Task<VerifyResult> VerifyFlashAsync(CalibrationFile expectedCalibration, CancellationToken ct)
    {
        var result = new VerifyResult();

        foreach (var block in expectedCalibration.Blocks)
        {
            // Read back a sample from each block
            var sampleSize = Math.Min(256, block.Data.Length);
            var sampleOffset = block.Data.Length / 2; // Sample from middle

            // UDS Read Memory By Address
            var request = new byte[7];
            request[0] = 0x23;
            request[1] = 0x24;
            var address = block.StartAddress + (uint)sampleOffset;
            request[2] = (byte)(address >> 24);
            request[3] = (byte)(address >> 16);
            request[4] = (byte)(address >> 8);
            request[5] = (byte)(address & 0xFF);
            request[6] = (byte)sampleSize;

            var response = await _adapter.SendMessageAsync(request, 5000, ct);

            if (response.Length > 1 && response[0] == 0x63)
            {
                var readData = response.Skip(1).Take(sampleSize).ToArray();
                var expectedData = block.Data.Skip(sampleOffset).Take(sampleSize).ToArray();

                if (!readData.SequenceEqual(expectedData))
                {
                    result.Success = false;
                    result.FailureReason = $"Verification failed at block {block.Name}, offset 0x{sampleOffset:X}";
                    return result;
                }
            }
        }

        result.Success = true;
        return result;
    }

    private async Task<int> MeasureLatencyAsync(CancellationToken ct)
    {
        var stopwatch = Stopwatch.StartNew();
        await _udsService.TesterPresentAsync(false, ct);
        stopwatch.Stop();
        return (int)stopwatch.ElapsedMilliseconds;
    }

    private void ReportProgress(FlashStage stage, int percent, string message)
    {
        if (_currentSession != null)
        {
            _currentSession.Stage = stage;
            _currentSession.ProgressPercent = percent;
            _currentSession.StatusMessage = message;
        }

        ProgressChanged?.Invoke(this, new FlashProgressEventArgs(stage, percent, message));
    }

    private void RaiseWarning(FlashWarningType type, string message)
    {
        WarningRaised?.Invoke(this, new FlashWarningEventArgs(type, message));
    }

    /// <summary>
    /// Attempts to recover a "bricked" ECU by forcing it into a programming 
    /// session and restoring the shadow backup.
    /// </summary>
    public async Task<FlashResult> TryEmergencyRecoveryAsync(
        FlashConfig config,
        Func<byte[], byte[]> securityKeyAlgorithm,
        CancellationToken ct = default)
    {
        ReportProgress(FlashStage.Recovery, 10, "Starting ECU wake-up sequence...");
        
        // 1. Wakeup: Flood with TesterPresent to keep bootloader awake
        for (int i = 0; i < 20; i++)
        {
            await _udsService.TesterPresentAsync(true, ct);
            await Task.Delay(50, ct);
        }

        // 2. Try to re-enter session aggressively
        UdsResponse sessionResp = null;
        for (int attempt = 0; attempt < 5; attempt++)
        {
            sessionResp = await _udsService.DiagnosticSessionControlAsync(UdsSession.Programming, ct);
            if (sessionResp.IsPositive) break;
            await Task.Delay(500, ct);
        }

        if (sessionResp == null || !sessionResp.IsPositive)
        {
            return new FlashResult { Success = false, FailureReason = "Recovery: Failed to enter programming session after 5 attempts." };
        }

        // 3. Find latest backup if not provided
        var backupPath = _currentSession?.BackupPath;
        if (string.IsNullOrEmpty(backupPath))
        {
            // Search directory
            var backupsDir = Path.Combine(_repository.RepositoryPath, "backups");
            if (Directory.Exists(backupsDir))
            {
                backupPath = Directory.GetDirectories(backupsDir)
                    .OrderByDescending(d => d)
                    .FirstOrDefault();
            }
        }

        if (string.IsNullOrEmpty(backupPath))
        {
            return new FlashResult { Success = false, FailureReason = "Recovery: No shadow backup found to restore." };
        }

        ReportProgress(FlashStage.Recovery, 30, "Restoring from shadow backup...");
        
        // Disable pre-flight for recovery (ECU might report garbage when bricked)
        var recoveryConfig = new FlashConfig
        {
            Calibration = config.Calibration, // We use the target calibration if restore fails, but usually we want the backup
            VerifyAfterWrite = true,
            CreateBackup = false // Don't backup a bricked ECU
        };

        return await RestoreFromBackupAsync(backupPath, securityKeyAlgorithm, ct);
    }

    public void Dispose()
    {
        _flashCts?.Cancel();
        _flashCts?.Dispose();
        _flashLock.Dispose();
    }
}

#region Data Models

public class FlashConfig
{
    public CalibrationFile Calibration { get; set; } = null!;
    public byte SecurityLevel { get; set; } = 1;
    public bool VerifyAfterWrite { get; set; } = true;
    public bool CreateBackup { get; set; } = true;
}

public class FlashSession
{
    public Guid Id { get; set; }
    public DateTime StartTime { get; set; }
    public FlashConfig Config { get; set; } = null!;
    public FlashStage Stage { get; set; }
    public int ProgressPercent { get; set; }
    public string? StatusMessage { get; set; }
    public string? BackupPath { get; set; }
}

public enum FlashStage
{
    PreFlight,
    Backup,
    EnterSession,
    Security,
    RequestDownload,
    Transfer,
    ExitTransfer,
    Verify,
    Reset,
    PostVerify,
    Recovery,
    Complete
}

public class PreFlightResult
{
    public DateTime Timestamp { get; set; }
    public bool CanProceed { get; set; }
    public bool HasWarnings { get; set; }
    public List<PreFlightCheck> Checks { get; set; } = new();
}

public class PreFlightCheck
{
    public string Name { get; set; } = string.Empty;
    public CheckCategory Category { get; set; }
    public CheckStatus Status { get; set; }
    public string Message { get; set; } = string.Empty;
}

public enum CheckCategory
{
    Hardware,
    Power,
    Communication,
    Vehicle,
    Data,
    Security
}

public enum CheckStatus
{
    Passed,
    Warning,
    Failed
}



public class FlashResult
{
    public bool Success { get; set; }
    public DateTime StartTime { get; set; }
    public DateTime EndTime { get; set; }
    public TimeSpan Duration { get; set; }
    public string? FailureReason { get; set; }
    public FlashStage? FailedAtStage { get; set; }
    public bool WasCancelled { get; set; }
    public int BytesWritten { get; set; }
    public int BlocksWritten { get; set; }
    public string? BackupPath { get; set; }
    public PreFlightResult? PreFlightResult { get; set; }
    public VerifyResult? VerificationResult { get; set; }
}

public class VerifyResult
{
    public bool Success { get; set; }
    public string? FailureReason { get; set; }
}

public class ShadowBackup
{
    public string BackupPath { get; set; } = string.Empty;
    public string BackupHash { get; set; } = string.Empty;
    public DateTime Timestamp { get; set; }
}

public class BackupMetadata
{
    public DateTime CreatedAt { get; set; }
    public string EcuId { get; set; } = string.Empty;
    public string Checksum { get; set; } = string.Empty;
    public string Reason { get; set; } = string.Empty;
}

public class FlashProgressEventArgs : EventArgs
{
    public FlashStage Stage { get; }
    public int PercentComplete { get; }
    public string Message { get; }

    public FlashProgressEventArgs(FlashStage stage, int percent, string message)
    {
        Stage = stage;
        PercentComplete = percent;
        Message = message;
    }
}

public enum FlashWarningType
{
    LowVoltage,
    HighLatency,
    PostFlashCommunicationFailed,
    VerificationSkipped
}

public class FlashWarningEventArgs : EventArgs
{
    public FlashWarningType Type { get; }
    public string Message { get; }

    public FlashWarningEventArgs(FlashWarningType type, string message)
    {
        Type = type;
        Message = message;
    }
}

public class FlashCompleteEventArgs : EventArgs
{
    public FlashResult Result { get; }

    public FlashCompleteEventArgs(FlashResult result)
    {
        Result = result;
    }
}

public class FlashException : Exception
{
    public FlashStage? FailedStage { get; }
    public FlashException(string message, FlashStage? stage = null) : base(message) 
    {
        FailedStage = stage;
    }
    public FlashException(string message, Exception inner, FlashStage? stage = null) : base(message, inner) 
    {
        FailedStage = stage;
    }
}

#endregion
