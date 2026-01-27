import os
import sys

def find_null_bytes(directory):
    for root, dirs, files in os.walk(directory):
        if 'venv' in root or '__pycache__' in root:
            continue
        for file in files:
            if file.endswith('.py'):
                path = os.path.join(root, file)
                try:
                    with open(path, 'rb') as f:
                        content = f.read()
                        if b'\x00' in content:
                            print(f"Found null byte in: {path}")
                            # Print context around null byte
                            idx = content.find(b'\x00')
                            start = max(0, idx - 20)
                            end = min(len(content), idx + 20)
                            print(f"Context: {content[start:end]}")
                except Exception as e:
                    print(f"Error reading {path}: {e}")

if __name__ == '__main__':
    find_null_bytes('src/ai-training')
