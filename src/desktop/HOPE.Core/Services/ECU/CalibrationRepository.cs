using System.IO;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using HOPE.Core.Interfaces;

namespace HOPE.Core.Services.ECU;

/// <summary>
/// Git-like version control repository for ECU calibration files.
/// Provides versioning, diffing, and rollback capabilities.
/// </summary>
public class CalibrationRepository : IDisposable
{
    private readonly string _repoPath;
    private readonly string _objectsPath;
    private readonly string _refsPath;
    private readonly string _stagingPath;
    private readonly SemaphoreSlim _lock = new(1, 1);

    private const string HEAD_FILE = "HEAD";
    private const string INDEX_FILE = "index.json";
    private const int BLOCK_SIZE = 4096;

    /// <summary>
    /// Event raised when a new commit is created
    /// </summary>
    public event EventHandler<CalibrationCommitEventArgs>? CommitCreated;

    /// <summary>
    /// Gets the repository path
    /// </summary>
    public string RepositoryPath => _repoPath;

    public CalibrationRepository(string repositoryPath)
    {
        _repoPath = repositoryPath ?? throw new ArgumentNullException(nameof(repositoryPath));
        _objectsPath = Path.Combine(_repoPath, "objects");
        _refsPath = Path.Combine(_repoPath, "refs");
        _stagingPath = Path.Combine(_repoPath, "staging");
    }

    /// <summary>
    /// Initialize a new calibration repository
    /// </summary>
    public async Task InitializeAsync(CancellationToken ct = default)
    {
        await _lock.WaitAsync(ct);
        try
        {
            Directory.CreateDirectory(_repoPath);
            Directory.CreateDirectory(_objectsPath);
            Directory.CreateDirectory(_refsPath);
            Directory.CreateDirectory(_stagingPath);
            Directory.CreateDirectory(Path.Combine(_refsPath, "tags"));
            Directory.CreateDirectory(Path.Combine(_refsPath, "heads"));

            // Create initial HEAD pointing to main branch
            var headPath = Path.Combine(_repoPath, HEAD_FILE);
            if (!File.Exists(headPath))
            {
                await File.WriteAllTextAsync(headPath, "ref: refs/heads/main", ct);
            }

            // Create empty index
            var indexPath = Path.Combine(_repoPath, INDEX_FILE);
            if (!File.Exists(indexPath))
            {
                var index = new CalibrationIndex { Entries = new List<CalibrationIndexEntry>() };
                await File.WriteAllTextAsync(indexPath, JsonSerializer.Serialize(index), ct);
            }
        }
        finally
        {
            _lock.Release();
        }
    }

    /// <summary>
    /// Read calibration data from an ECU
    /// </summary>
    public async Task<CalibrationFile> ReadFromEcuAsync(
        IHardwareAdapter adapter,
        EcuReadConfig config,
        IProgress<CalibrationProgress>? progress = null,
        CancellationToken ct = default)
    {
        if (!adapter.IsConnected)
            throw new InvalidOperationException("Hardware adapter is not connected");

        var calibration = new CalibrationFile
        {
            EcuId = config.EcuId,
            ReadTimestamp = DateTime.UtcNow,
            Metadata = new CalibrationMetadata
            {
                VIN = config.VIN,
                EcuPartNumber = config.PartNumber,
                SoftwareVersion = config.SoftwareVersion
            }
        };

        var blocks = new List<CalibrationBlock>();
        long totalBytes = 0;
        long bytesRead = 0;

        // Calculate total size
        foreach (var region in config.MemoryRegions)
        {
            totalBytes += region.Size;
        }

        // Read each memory region
        foreach (var region in config.MemoryRegions)
        {
            progress?.Report(new CalibrationProgress
            {
                Stage = CalibrationStage.Reading,
                CurrentRegion = region.Name,
                PercentComplete = (int)(bytesRead * 100 / totalBytes)
            });

            var data = await ReadMemoryRegionAsync(adapter, region, ct);

            blocks.Add(new CalibrationBlock
            {
                Name = region.Name,
                StartAddress = region.StartAddress,
                Data = data,
                Checksum = ComputeChecksum(data)
            });

            bytesRead += region.Size;
        }

        calibration.Blocks = blocks;
        calibration.FullChecksum = ComputeFileChecksum(calibration);

        progress?.Report(new CalibrationProgress
        {
            Stage = CalibrationStage.Complete,
            PercentComplete = 100
        });

        return calibration;
    }

    private async Task<byte[]> ReadMemoryRegionAsync(
        IHardwareAdapter adapter,
        MemoryRegion region,
        CancellationToken ct)
    {
        var data = new List<byte>();
        var currentAddress = region.StartAddress;
        var remaining = region.Size;

        while (remaining > 0)
        {
            var blockSize = (int)Math.Min(remaining, BLOCK_SIZE);

            // UDS Read Memory By Address (0x23)
            var request = new byte[7];
            request[0] = 0x23; // ReadMemoryByAddress
            request[1] = 0x24; // Address and length format (2 bytes each)
            request[2] = (byte)(currentAddress >> 24);
            request[3] = (byte)(currentAddress >> 16);
            request[4] = (byte)(currentAddress >> 8);
            request[5] = (byte)(currentAddress & 0xFF);
            request[6] = (byte)blockSize;

            var response = await adapter.SendMessageAsync(request, 5000, ct);

            if (response.Length > 1 && response[0] == 0x63)
            {
                data.AddRange(response.Skip(1));
            }
            else if (response.Length > 2 && response[0] == 0x7F)
            {
                throw new CalibrationException($"ECU rejected read at 0x{currentAddress:X8}: NRC 0x{response[2]:X2}");
            }
            else
            {
                throw new CalibrationException($"Invalid response reading address 0x{currentAddress:X8}");
            }

            currentAddress += (uint)blockSize;
            remaining -= (uint)blockSize;
        }

        return data.ToArray();
    }

    /// <summary>
    /// Stage a calibration file for commit
    /// </summary>
    public async Task StageAsync(CalibrationFile calibration, CancellationToken ct = default)
    {
        await _lock.WaitAsync(ct);
        try
        {
            // Write calibration to staging area
            var stagingFile = Path.Combine(_stagingPath, $"{calibration.EcuId}.cal");
            var json = JsonSerializer.Serialize(calibration, new JsonSerializerOptions { WriteIndented = true });
            await File.WriteAllTextAsync(stagingFile, json, ct);

            // Update index
            var index = await LoadIndexAsync(ct);
            var entry = index.Entries.FirstOrDefault(e => e.EcuId == calibration.EcuId);

            if (entry == null)
            {
                entry = new CalibrationIndexEntry { EcuId = calibration.EcuId };
                index.Entries.Add(entry);
            }

            entry.StagedPath = stagingFile;
            entry.StagedChecksum = calibration.FullChecksum;
            entry.StagedTimestamp = DateTime.UtcNow;

            await SaveIndexAsync(index, ct);
        }
        finally
        {
            _lock.Release();
        }
    }

    /// <summary>
    /// Commit staged calibration with a message
    /// </summary>
    public async Task<string> CommitAsync(string message, string author = "HOPE User", CancellationToken ct = default)
    {
        await _lock.WaitAsync(ct);
        try
        {
            var index = await LoadIndexAsync(ct);
            var stagedEntries = index.Entries.Where(e => !string.IsNullOrEmpty(e.StagedPath)).ToList();

            if (stagedEntries.Count == 0)
                throw new InvalidOperationException("Nothing to commit");

            // Create blob objects for each staged file
            var blobHashes = new Dictionary<string, string>();
            foreach (var entry in stagedEntries)
            {
                var content = await File.ReadAllBytesAsync(entry.StagedPath!, ct);
                var blobHash = await WriteObjectAsync(content, "blob", ct);
                blobHashes[entry.EcuId] = blobHash;
            }

            // Create tree object
            var treeEntries = blobHashes.Select(kv => new TreeEntry
            {
                Mode = "100644",
                Name = $"{kv.Key}.cal",
                Hash = kv.Value
            }).ToList();

            var treeHash = await WriteTreeAsync(treeEntries, ct);

            // Get parent commit
            var headCommit = await GetHeadCommitAsync(ct);

            // Create commit object
            var commit = new CommitObject
            {
                TreeHash = treeHash,
                ParentHash = headCommit,
                Author = author,
                AuthorDate = DateTime.UtcNow,
                Committer = author,
                CommitDate = DateTime.UtcNow,
                Message = message
            };

            var commitJson = JsonSerializer.Serialize(commit);
            var commitHash = await WriteObjectAsync(Encoding.UTF8.GetBytes(commitJson), "commit", ct);

            // Update HEAD
            await UpdateHeadAsync(commitHash, ct);

            // Clear staging
            foreach (var entry in stagedEntries)
            {
                if (File.Exists(entry.StagedPath))
                    File.Delete(entry.StagedPath);

                entry.CommittedHash = blobHashes[entry.EcuId];
                entry.StagedPath = null;
                entry.StagedChecksum = null;
            }

            await SaveIndexAsync(index, ct);

            CommitCreated?.Invoke(this, new CalibrationCommitEventArgs(commitHash, message, author));

            return commitHash;
        }
        finally
        {
            _lock.Release();
        }
    }

    /// <summary>
    /// Create a diff between two commits or calibration files
    /// </summary>
    public async Task<CalibrationDiff> DiffAsync(
        string commitA,
        string commitB,
        CancellationToken ct = default)
    {
        var calA = await LoadCalibrationFromCommitAsync(commitA, ct);
        var calB = await LoadCalibrationFromCommitAsync(commitB, ct);

        return await DiffCalibrationsAsync(calA, calB, ct);
    }

    /// <summary>
    /// Create a diff between two calibration files
    /// </summary>
    public Task<CalibrationDiff> DiffCalibrationsAsync(
        CalibrationFile calA,
        CalibrationFile calB,
        CancellationToken ct = default)
    {
        var diff = new CalibrationDiff
        {
            BaseEcuId = calA.EcuId,
            CompareEcuId = calB.EcuId,
            BaseTimestamp = calA.ReadTimestamp,
            CompareTimestamp = calB.ReadTimestamp
        };

        var changes = new List<BlockChange>();

        // Compare blocks
        foreach (var blockA in calA.Blocks)
        {
            var blockB = calB.Blocks.FirstOrDefault(b => b.StartAddress == blockA.StartAddress);

            if (blockB == null)
            {
                changes.Add(new BlockChange
                {
                    BlockName = blockA.Name,
                    Address = blockA.StartAddress,
                    ChangeType = ChangeType.Removed,
                    OldData = blockA.Data
                });
            }
            else if (!blockA.Data.SequenceEqual(blockB.Data))
            {
                var byteChanges = CompareBytes(blockA.Data, blockB.Data, blockA.StartAddress);
                changes.Add(new BlockChange
                {
                    BlockName = blockA.Name,
                    Address = blockA.StartAddress,
                    ChangeType = ChangeType.Modified,
                    OldData = blockA.Data,
                    NewData = blockB.Data,
                    ByteChanges = byteChanges
                });
            }
        }

        // Find new blocks
        foreach (var blockB in calB.Blocks)
        {
            if (!calA.Blocks.Any(b => b.StartAddress == blockB.StartAddress))
            {
                changes.Add(new BlockChange
                {
                    BlockName = blockB.Name,
                    Address = blockB.StartAddress,
                    ChangeType = ChangeType.Added,
                    NewData = blockB.Data
                });
            }
        }

        diff.Changes = changes;
        diff.TotalBytesChanged = changes.Sum(c => c.ByteChanges?.Count ?? c.NewData?.Length ?? c.OldData?.Length ?? 0);

        return Task.FromResult(diff);
    }

    private List<ByteChange> CompareBytes(byte[] oldData, byte[] newData, uint baseAddress)
    {
        var changes = new List<ByteChange>();
        var maxLen = Math.Max(oldData.Length, newData.Length);

        for (int i = 0; i < maxLen; i++)
        {
            var oldByte = i < oldData.Length ? oldData[i] : (byte?)null;
            var newByte = i < newData.Length ? newData[i] : (byte?)null;

            if (oldByte != newByte)
            {
                changes.Add(new ByteChange
                {
                    Address = baseAddress + (uint)i,
                    OldValue = oldByte,
                    NewValue = newByte
                });
            }
        }

        return changes;
    }

    /// <summary>
    /// Validate the checksum of a calibration file
    /// </summary>
    public async Task<ChecksumValidationResult> ValidateChecksumAsync(
        CalibrationFile calibration,
        CancellationToken ct = default)
    {
        var result = new ChecksumValidationResult
        {
            ExpectedChecksum = calibration.FullChecksum,
            CalculatedChecksum = ComputeFileChecksum(calibration)
        };

        result.IsValid = result.ExpectedChecksum == result.CalculatedChecksum;

        // Validate individual blocks
        result.BlockResults = new List<BlockChecksumResult>();
        foreach (var block in calibration.Blocks)
        {
            var calculatedBlockChecksum = ComputeChecksum(block.Data);
            result.BlockResults.Add(new BlockChecksumResult
            {
                BlockName = block.Name,
                Address = block.StartAddress,
                ExpectedChecksum = block.Checksum,
                CalculatedChecksum = calculatedBlockChecksum,
                IsValid = block.Checksum == calculatedBlockChecksum
            });
        }

        return result;
    }

    /// <summary>
    /// Rollback to a previous commit
    /// </summary>
    public async Task<CalibrationFile> RollbackAsync(string commitHash, CancellationToken ct = default)
    {
        await _lock.WaitAsync(ct);
        try
        {
            var calibration = await LoadCalibrationFromCommitAsync(commitHash, ct);

            // Stage the rolled-back calibration
            await StageAsync(calibration, ct);

            return calibration;
        }
        finally
        {
            _lock.Release();
        }
    }

    /// <summary>
    /// Get the commit history
    /// </summary>
    public async Task<List<CalibrationCommit>> GetHistoryAsync(int limit = 50, CancellationToken ct = default)
    {
        var history = new List<CalibrationCommit>();
        var currentHash = await GetHeadCommitAsync(ct);

        while (!string.IsNullOrEmpty(currentHash) && history.Count < limit)
        {
            var commitObject = await LoadCommitAsync(currentHash, ct);
            if (commitObject == null)
                break;

            history.Add(new CalibrationCommit
            {
                Hash = currentHash,
                ShortHash = currentHash[..7],
                Message = commitObject.Message,
                Author = commitObject.Author,
                Timestamp = commitObject.CommitDate,
                ParentHash = commitObject.ParentHash
            });

            currentHash = commitObject.ParentHash;
        }

        return history;
    }

    /// <summary>
    /// Create a named tag for a commit
    /// </summary>
    public async Task CreateTagAsync(string tagName, string commitHash, string? message = null, CancellationToken ct = default)
    {
        var tagPath = Path.Combine(_refsPath, "tags", tagName);
        var tag = new CalibrationTag
        {
            CommitHash = commitHash,
            TagName = tagName,
            Message = message,
            CreatedAt = DateTime.UtcNow
        };

        await File.WriteAllTextAsync(tagPath, JsonSerializer.Serialize(tag), ct);
    }

    /// <summary>
    /// Export a calibration to a binary file
    /// </summary>
    public async Task ExportToBinaryAsync(
        CalibrationFile calibration,
        string outputPath,
        CancellationToken ct = default)
    {
        await using var stream = File.Create(outputPath);

        foreach (var block in calibration.Blocks.OrderBy(b => b.StartAddress))
        {
            await stream.WriteAsync(block.Data, ct);
        }
    }

    #region Internal Helper Methods

    private async Task<CalibrationIndex> LoadIndexAsync(CancellationToken ct)
    {
        var indexPath = Path.Combine(_repoPath, INDEX_FILE);
        if (!File.Exists(indexPath))
            return new CalibrationIndex { Entries = new List<CalibrationIndexEntry>() };

        var json = await File.ReadAllTextAsync(indexPath, ct);
        return JsonSerializer.Deserialize<CalibrationIndex>(json) ?? new CalibrationIndex { Entries = new List<CalibrationIndexEntry>() };
    }

    private async Task SaveIndexAsync(CalibrationIndex index, CancellationToken ct)
    {
        var indexPath = Path.Combine(_repoPath, INDEX_FILE);
        var json = JsonSerializer.Serialize(index, new JsonSerializerOptions { WriteIndented = true });
        await File.WriteAllTextAsync(indexPath, json, ct);
    }

    private async Task<string> WriteObjectAsync(byte[] content, string type, CancellationToken ct)
    {
        var hash = ComputeSha256Hash(content);
        var dirPath = Path.Combine(_objectsPath, hash[..2]);
        var filePath = Path.Combine(dirPath, hash[2..]);

        if (!File.Exists(filePath))
        {
            Directory.CreateDirectory(dirPath);
            await File.WriteAllBytesAsync(filePath, content, ct);
        }

        return hash;
    }

    private async Task<string> WriteTreeAsync(List<TreeEntry> entries, CancellationToken ct)
    {
        var json = JsonSerializer.Serialize(entries);
        return await WriteObjectAsync(Encoding.UTF8.GetBytes(json), "tree", ct);
    }

    private async Task<string?> GetHeadCommitAsync(CancellationToken ct)
    {
        var headPath = Path.Combine(_repoPath, HEAD_FILE);
        if (!File.Exists(headPath))
            return null;

        var headContent = await File.ReadAllTextAsync(headPath, ct);

        if (headContent.StartsWith("ref: "))
        {
            var refPath = Path.Combine(_repoPath, headContent[5..].Trim());
            if (File.Exists(refPath))
                return await File.ReadAllTextAsync(refPath, ct);
            return null;
        }

        return headContent.Trim();
    }

    private async Task UpdateHeadAsync(string commitHash, CancellationToken ct)
    {
        var headPath = Path.Combine(_repoPath, HEAD_FILE);
        var headContent = await File.ReadAllTextAsync(headPath, ct);

        if (headContent.StartsWith("ref: "))
        {
            var refPath = Path.Combine(_repoPath, headContent[5..].Trim());
            Directory.CreateDirectory(Path.GetDirectoryName(refPath)!);
            await File.WriteAllTextAsync(refPath, commitHash, ct);
        }
        else
        {
            await File.WriteAllTextAsync(headPath, commitHash, ct);
        }
    }

    private async Task<CommitObject?> LoadCommitAsync(string hash, CancellationToken ct)
    {
        var filePath = Path.Combine(_objectsPath, hash[..2], hash[2..]);
        if (!File.Exists(filePath))
            return null;

        var json = await File.ReadAllTextAsync(filePath, ct);
        return JsonSerializer.Deserialize<CommitObject>(json);
    }

    private async Task<CalibrationFile> LoadCalibrationFromCommitAsync(string commitHash, CancellationToken ct)
    {
        var commit = await LoadCommitAsync(commitHash, ct)
            ?? throw new CalibrationException($"Commit {commitHash} not found");

        // Load tree
        var treePath = Path.Combine(_objectsPath, commit.TreeHash[..2], commit.TreeHash[2..]);
        var treeJson = await File.ReadAllTextAsync(treePath, ct);
        var treeEntries = JsonSerializer.Deserialize<List<TreeEntry>>(treeJson)!;

        // Load first calibration blob
        var entry = treeEntries.FirstOrDefault()
            ?? throw new CalibrationException("No calibration files in commit");

        var blobPath = Path.Combine(_objectsPath, entry.Hash[..2], entry.Hash[2..]);
        var calibrationJson = await File.ReadAllTextAsync(blobPath, ct);

        return JsonSerializer.Deserialize<CalibrationFile>(calibrationJson)
            ?? throw new CalibrationException("Failed to deserialize calibration");
    }

    private static string ComputeChecksum(byte[] data)
    {
        using var sha = SHA256.Create();
        var hash = sha.ComputeHash(data);
        return Convert.ToHexString(hash).ToLower();
    }

    private static string ComputeFileChecksum(CalibrationFile calibration)
    {
        using var sha = SHA256.Create();
        using var stream = new MemoryStream();

        foreach (var block in calibration.Blocks.OrderBy(b => b.StartAddress))
        {
            stream.Write(block.Data);
        }

        stream.Position = 0;
        var hash = sha.ComputeHash(stream);
        return Convert.ToHexString(hash).ToLower();
    }

    private static string ComputeSha256Hash(byte[] data)
    {
        using var sha = SHA256.Create();
        var hash = sha.ComputeHash(data);
        return Convert.ToHexString(hash).ToLower();
    }

    #endregion

    public void Dispose()
    {
        _lock.Dispose();
    }
}

#region Data Models

public class CalibrationFile
{
    public string EcuId { get; set; } = string.Empty;
    public DateTime ReadTimestamp { get; set; }
    public CalibrationMetadata Metadata { get; set; } = new();
    public List<CalibrationBlock> Blocks { get; set; } = new();
    public string FullChecksum { get; set; } = string.Empty;
}

public class CalibrationMetadata
{
    public string? VIN { get; set; }
    public string? EcuPartNumber { get; set; }
    public string? SoftwareVersion { get; set; }
    public string? HardwareVersion { get; set; }
    public string? CalibrationVersion { get; set; }
    public Dictionary<string, string> CustomFields { get; set; } = new();
}

public class CalibrationBlock
{
    public string Name { get; set; } = string.Empty;
    public uint StartAddress { get; set; }
    public byte[] Data { get; set; } = Array.Empty<byte>();
    public string Checksum { get; set; } = string.Empty;
}

public class EcuReadConfig
{
    public string EcuId { get; set; } = string.Empty;
    public string? VIN { get; set; }
    public string? PartNumber { get; set; }
    public string? SoftwareVersion { get; set; }
    public List<MemoryRegion> MemoryRegions { get; set; } = new();
}

public class MemoryRegion
{
    public string Name { get; set; } = string.Empty;
    public uint StartAddress { get; set; }
    public uint Size { get; set; }
    public bool IsWritable { get; set; }
}

public class CalibrationDiff
{
    public string BaseEcuId { get; set; } = string.Empty;
    public string CompareEcuId { get; set; } = string.Empty;
    public DateTime BaseTimestamp { get; set; }
    public DateTime CompareTimestamp { get; set; }
    public List<BlockChange> Changes { get; set; } = new();
    public int TotalBytesChanged { get; set; }
}

public class BlockChange
{
    public string BlockName { get; set; } = string.Empty;
    public uint Address { get; set; }
    public ChangeType ChangeType { get; set; }
    public byte[]? OldData { get; set; }
    public byte[]? NewData { get; set; }
    public List<ByteChange>? ByteChanges { get; set; }
}

public class ByteChange
{
    public uint Address { get; set; }
    public byte? OldValue { get; set; }
    public byte? NewValue { get; set; }
}

public enum ChangeType
{
    Added,
    Modified,
    Removed
}

public class ChecksumValidationResult
{
    public bool IsValid { get; set; }
    public string ExpectedChecksum { get; set; } = string.Empty;
    public string CalculatedChecksum { get; set; } = string.Empty;
    public List<BlockChecksumResult> BlockResults { get; set; } = new();
}

public class BlockChecksumResult
{
    public string BlockName { get; set; } = string.Empty;
    public uint Address { get; set; }
    public bool IsValid { get; set; }
    public string ExpectedChecksum { get; set; } = string.Empty;
    public string CalculatedChecksum { get; set; } = string.Empty;
}

public class CalibrationCommit
{
    public string Hash { get; set; } = string.Empty;
    public string ShortHash { get; set; } = string.Empty;
    public string Message { get; set; } = string.Empty;
    public string Author { get; set; } = string.Empty;
    public DateTime Timestamp { get; set; }
    public string? ParentHash { get; set; }
}

public class CalibrationProgress
{
    public CalibrationStage Stage { get; set; }
    public string? CurrentRegion { get; set; }
    public int PercentComplete { get; set; }
    public string? StatusMessage { get; set; }
}

public enum CalibrationStage
{
    Initializing,
    Reading,
    Validating,
    Writing,
    Verifying,
    Complete,
    Failed
}

public class CalibrationCommitEventArgs : EventArgs
{
    public string CommitHash { get; }
    public string Message { get; }
    public string Author { get; }

    public CalibrationCommitEventArgs(string commitHash, string message, string author)
    {
        CommitHash = commitHash;
        Message = message;
        Author = author;
    }
}

#endregion

#region Internal Types

internal class CalibrationIndex
{
    public List<CalibrationIndexEntry> Entries { get; set; } = new();
}

internal class CalibrationIndexEntry
{
    public string EcuId { get; set; } = string.Empty;
    public string? StagedPath { get; set; }
    public string? StagedChecksum { get; set; }
    public DateTime? StagedTimestamp { get; set; }
    public string? CommittedHash { get; set; }
}

internal class CommitObject
{
    public string TreeHash { get; set; } = string.Empty;
    public string? ParentHash { get; set; }
    public string Author { get; set; } = string.Empty;
    public DateTime AuthorDate { get; set; }
    public string Committer { get; set; } = string.Empty;
    public DateTime CommitDate { get; set; }
    public string Message { get; set; } = string.Empty;
}

internal class TreeEntry
{
    public string Mode { get; set; } = string.Empty;
    public string Name { get; set; } = string.Empty;
    public string Hash { get; set; } = string.Empty;
}

internal class CalibrationTag
{
    public string CommitHash { get; set; } = string.Empty;
    public string TagName { get; set; } = string.Empty;
    public string? Message { get; set; }
    public DateTime CreatedAt { get; set; }
}

#endregion

public class CalibrationException : Exception
{
    public CalibrationException(string message) : base(message) { }
    public CalibrationException(string message, Exception inner) : base(message, inner) { }
}
