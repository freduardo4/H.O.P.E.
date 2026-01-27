import os
import codecs

BOM_UTF8 = codecs.BOM_UTF8
BOM_UTF16_LE = codecs.BOM_UTF16_LE
BOM_UTF16_BE = codecs.BOM_UTF16_BE

def sanitize_file(path):
    with open(path, 'rb') as f:
        content = f.read()

    new_content = content
    if content.startswith(BOM_UTF8):
        print(f"Removing UTF-8 BOM from {path}")
        new_content = content[len(BOM_UTF8):]
    elif content.startswith(BOM_UTF16_LE):
        print(f"Converting UTF-16LE BOM from {path}")
        new_content = content[len(BOM_UTF16_LE):].decode('utf-16-le').encode('utf-8')
    elif content.startswith(BOM_UTF16_BE):
        print(f"Converting UTF-16BE BOM from {path}")
        new_content = content[len(BOM_UTF16_BE):].decode('utf-16-be').encode('utf-8')
    
    # Also check for null bytes in what should be text
    if b'\x00' in new_content:
         print(f"WARNING: Null bytes still found in {path} after stripping BOM")

    if new_content != content:
        with open(path, 'wb') as f:
            f.write(new_content)
        print(f"Fixed {path}")

def main():
    for root, dirs, files in os.walk('.'):
        if 'venv' in root or '__pycache__' in root or '.git' in root:
            continue
        for file in files:
            if file.endswith('.py'):
                sanitize_file(os.path.join(root, file))

if __name__ == '__main__':
    main()
