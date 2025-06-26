#!/usr/bin/env python3
"""First launch Module"""
import requests


if __name__ == '__main__':
    # get all upcoming launches
    launches_url = 'https://api.spacexdata.com/v5/launches/upcoming'
    response = requests.get(launches_url)
    launches = response.json()

    # sort by launch date 
    launches.sort(key=lambda x: x['date_unix'])

    # pick the earliest launch
    first = launches[0]

    # extract needed IDs
    rocket_id = first['rocket']
    launchpad_id = first['launchpad']

    # get rocket and launchpad details
    rocket_url = f'https://api.spacexdata.com/v4/rockets/{rocket_id}'
    rocket = requests.get(rocket_url).json()
    launchpad_url = f'https://api.spacexdata.com/v4/launchpads/{launchpad_id}'
    launchpad = requests.get(launchpad_url).json()

    # print the result
    print(f"{first['name']} ({first['date_local']}) "
          f"{rocket['name']} - {launchpad['name']} "
          f"({launchpad['locality']})")
