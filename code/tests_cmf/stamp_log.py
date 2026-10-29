"""Reads text on stdin and writes each line with a time in front of it."""
import sys
from datetime import datetime
for line in sys.stdin:
    sys.stdout.write(datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '  ' + line)
    sys.stdout.flush()
