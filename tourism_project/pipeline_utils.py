"""Small helpers shared by the pipeline scripts (kept dependency-free on purpose)."""
import time

# HTTP status codes that will not fix themselves, so retrying only wastes time.
_NO_RETRY = (401, 403, 404)


def retry(fn, tries: int = 3, delay: float = 4.0, what: str = "operation"):
    """Call fn(); on a transient failure wait and try again, then re-raise the last error."""
    last = None
    for attempt in range(1, tries + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            last = exc
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status in _NO_RETRY:
                raise
            print(f"  {what} failed (attempt {attempt}/{tries}): {type(exc).__name__}: {exc}")
            if attempt < tries:
                time.sleep(delay * attempt)
    raise last
