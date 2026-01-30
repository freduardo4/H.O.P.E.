# Utility Scripts

This directory contains utility scripts for the H.O.P.E. project.

## Scripts

### `find_null_bytes.py`

A utility to detect null bytes (`0x00`) in files, useful for identifying corrupted logs or binary artifacts that shouldn't contain nulls.

**Usage:**
```bash
python scripts/find_null_bytes.py <path_to_file_or_directory>
```
