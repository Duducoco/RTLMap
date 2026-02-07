#!/usr/bin/env python3
"""
RTLMap - RTL Coverage Annotation Tool (入口脚本)

实际实现位于 annotation/data_annotate.py
"""

import sys
from annotation.data_annotate import main

if __name__ == "__main__":
    sys.exit(main())
