import time
import requests


def with_retries(func, retries=3, delay=1.5):
    for i in range(retries):
        try:
            return func()
        except requests.exceptions.ConnectionError:
            if i < retries - 1:
                time.sleep(delay * (i + 1))
            else:
                raise


def is_quota_error(exc):
    msg = str(exc).lower()
    return "429" in msg or "quota" in msg