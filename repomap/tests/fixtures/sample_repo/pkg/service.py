from pkg.cache import invalidate, invalidate_namespace


def update_item(namespace: str, item_id: int, payload: dict) -> None:
    """Update an item and invalidate its cache entry."""
    _persist(namespace, item_id, payload)
    invalidate(namespace, item_id)


def bulk_update(namespace: str, items: list) -> None:
    """Update many items and invalidate the whole namespace."""
    for item_id, payload in items:
        _persist(namespace, item_id, payload)
    invalidate_namespace(namespace)


def _persist(namespace: str, item_id: int, payload: dict) -> None:
    pass
