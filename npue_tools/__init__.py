# NpuEmbeddings -- the python side of the tree, as a package.
#
# The plan (RUST_MIGRATION_PLAN.md, phase 2) moves the 62 `sys.path` hacks in
# tools/ into one package with real imports. That move is not done; this
# package starts with what phase 3 needs and nothing else, because a package
# that exists only to hold a directory name is a second layout to maintain
# while the first one still runs.
#
# What lives here so far: `weights/`, the answer to "where do the numbers come
# from" -- safetensors first, ONNX as the fallback (and the only source for the
# vision families whose graphs are traced from ONNX).

__all__ = ["weights"]
