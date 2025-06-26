#!/usr/bin/env python3
"""Display The Number Of Launches Per Rocket Module"""
import requests


if __name__ == '__main__':
    # get all launches
    launches = requests.get(
        'https://api.spacexdata.com/v4/launches').json()

    # count launches per rocket ID
    rocket_counts = {}
    for launch in launches:
        rocket_id = launch.get('rocket')
        rocket_counts[rocket_id] = rocket_counts.get(rocket_id, 0) + 1

    # get rocket names
    rockets = requests.get(
        'https://api.spacexdata.com/v4/rockets').json()
    rocket_names = {r['id']: r['name'] for r in rockets}

    # convert to list of tuples
    result = []
    for rocket_id, count in rocket_counts.items():
        name = rocket_names.get(rocket_id, "Unknown")
        result.append((name, count))

    # sort by count descending, then by name ascending
    result.sort(key=lambda x: (-x[1], x[0]))

    # print the results
    for name, count in result:
        print(f"{name}: {count}")
