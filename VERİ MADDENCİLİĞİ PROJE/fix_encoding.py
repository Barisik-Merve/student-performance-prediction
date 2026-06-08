# -*- coding: utf-8 -*-
"""
Fix double-encoded UTF-8 in app.py and rewrite with clean app content.
"""
import sys
import io

# Force UTF-8 output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

APP_PATH = r'c:\Users\HP\OneDrive\Desktop\VERİ MADDENCİLİĞİ PROJE\app.py'

with open(APP_PATH, 'rb') as f:
    raw = f.read()

# The file is double-utf8 encoded (UTF-8 bytes read as latin-1 and re-saved as UTF-8)
# Fix: decode as UTF-8 to get mojibake, then encode each char back to latin-1 bytes, then decode as UTF-8
content_mojibake = raw.decode('utf-8', errors='replace')

fixed_bytes = bytearray()
for char in content_mojibake:
    try:
        fixed_bytes += char.encode('latin-1')
    except (UnicodeEncodeError, ValueError):
        # Can't convert back - keep as UTF-8 (likely already correct or replacement char)
        if char == '\ufffd':
            fixed_bytes += b'?'
        else:
            fixed_bytes += char.encode('utf-8', errors='replace')

fixed_content = fixed_bytes.decode('utf-8', errors='replace')

# Print first 500 chars to verify
print("=== FIRST 500 CHARS ===")
print(fixed_content[:500])
print("=== CHECKS ===")
print("Contains 'Öğrenci':", 'Öğrenci' in fixed_content)
print("Contains 'Başarı':", 'Başarı' in fixed_content)
print("Contains 'Çalışma':", 'Çalışma' in fixed_content)
print("Total length:", len(fixed_content))
