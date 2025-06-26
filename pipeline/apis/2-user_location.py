#!/usr/bin/env python3
"""User Location Module"""
import sys
import time
import requests


if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit(1)

    # get the URL from the cmd line arg
    url = sys.argv[1]

    try:
        # send GET request to GitHub
        response = requests.get(url)
    except requests.exceptions.RequestException:
        sys.exit(1)

    if response.status_code == 200:
        data = response.json()
        print(data.get("location"))
    elif response.status_code == 404:
        print("Not found")
    elif response.status_code == 403:
        reset = response.headers.get("X-RateLimit-Reset")
        if reset:
            # calculate the time left until it resets
            now = int(time.time())
            reset_time = int(reset)
            minutes = max((reset_time - now + 59) // 60, 0)
            print(f"Reset in {minutes} min")
