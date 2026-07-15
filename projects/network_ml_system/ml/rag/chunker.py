"""
Chunking Strategy (Version 1)
-----------------------------

Documents are split along semantic boundaries rather than
fixed token or character counts.

For Markdown documents, headings (#, ##, ###) define the
initial chunk boundaries.

This preserves semantic coherence and improves retrieval
quality.

Future versions may further subdivide large sections based
on retrieval performance.
"""


