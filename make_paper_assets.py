#!/usr/bin/env python3
"""Compatibility wrapper.

The previous make_paper_assets.py could generate fallback/synthetic figures when
real experiment artifacts were missing. For paper submission, use only real
artifacts. This wrapper delegates to paper_assets_builder.py, which records
missing inputs instead of fabricating numbers.
"""
from paper_assets_builder import main

if __name__ == "__main__":
    main()
