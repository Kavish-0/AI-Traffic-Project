from __future__ import annotations

from fastapi import HTTPException, status


def not_implemented(phase: str) -> HTTPException:
    return HTTPException(status.HTTP_501_NOT_IMPLEMENTED, f"Not implemented yet ({phase})")
