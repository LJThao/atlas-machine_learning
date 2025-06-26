#!/usr/bin/env python3
"""Available Ships Module"""
import requests


def availableShips(passengerCount):
    """Function to create a method that returns the list of ships that can
    hold a given number of passengers:

    If no ship available, return an empty list.

    """
    url = 'https://swapi-api.hbtn.io/api/starships/'
    ships = []

    while url:
        res = requests.get(url)
        data = res.json()

        for ship in data['results']:
            # get the # of passengers and remove commas
            passengers = ship.get('passengers', '0').replace(',', '')
            if passengers.isdigit() and int(passengers) >= passengerCount:
                ships.append(ship['name'])

        # move to the next
        url = data.get('next')

    return ships
