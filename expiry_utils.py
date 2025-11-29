"""Utility functions for option expiry dates."""
from datetime import datetime, timedelta, timezone


def get_nearest_friday(target_date: datetime) -> str:
    """Return ``YYYYMMDD`` for the calendar date closest to Friday.

    For dates that are not already on a Friday, the function selects whichever
    Friday (previous or next) is nearer in time.
    """

    weekday = target_date.weekday()  # 0=Mon ... 4=Fri, 5=Sat, 6=Sun
    if weekday == 4:
        return target_date.strftime("%Y%m%d")
    days_backward = (weekday - 4) % 7
    days_forward = (4 - weekday) % 7
    if days_backward <= days_forward:
        target_date -= timedelta(days=days_backward)
    else:
        target_date += timedelta(days=days_forward)
    return target_date.strftime("%Y%m%d")


def get_nearest_friday_expiry(
    current_date: datetime | None = None,
    weeks_ahead: int = 0,
    days_ahead: int | None = None,
) -> str:
    """Return the expiry date for the nearest Friday in ``YYYYMMDD`` format.

    Args:
        current_date: Reference date. Defaults to current UTC time.
        weeks_ahead: Number of additional weeks to look ahead beyond the nearest Friday.
        days_ahead: Absolute day offset to add before normalising to Friday.

    Returns:
        Expiry date string for the target Friday.
    """

    if current_date is None:
        current_date = datetime.now(timezone.utc)

    if days_ahead is not None:
        target = current_date + timedelta(days=days_ahead)
        return get_nearest_friday(target)

    days_to_friday = (4 - current_date.weekday()) % 7
    target = current_date + timedelta(days=days_to_friday + 7 * weeks_ahead)

    # Guard against non-standard dates (e.g., weekends) by normalising to Friday
    return get_nearest_friday(target)
