def compute_key(namespace: str, item_id: int) -> str:
    """Build a cache key from namespace and id."""
    return f"{namespace}:{item_id}"


def invalidate(namespace: str, item_id: int) -> None:
    """Invalidate a single cache entry."""
    key = compute_key(namespace, item_id)
    _store.pop(key, None)


def invalidate_namespace(namespace: str) -> None:
    """Invalidate every entry under a namespace."""
    for key in list(_store.keys()):
        if key.startswith(f"{namespace}:"):
            _store.pop(key)


_store = {}
changed
