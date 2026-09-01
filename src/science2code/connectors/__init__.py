"""Optional adapters that POPULATE a corpus folder from an external source.

Nothing here is imported by the science2code core. The core knows one input:
a folder of PDFs. A connector's whole job is to fill such a folder from
somewhere else (a reference manager, say) and then step out of the way, so the
core stays standalone, dependency free, and unaware the connector exists. The
dependency arrow points one way: a connector may import the core, the core
never imports a connector.
"""
