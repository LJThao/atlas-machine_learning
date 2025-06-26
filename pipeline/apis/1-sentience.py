#!/usr/bin/env python3
"""Sentient Planets Module"""
import requests


def sentientPlanets():
    """Function to create a method that returns the list of names of
    the home planets of all sentient species.

    sentient type is either in the classification or designation attributes.

    """
    species_url = 'https://swapi-api.hbtn.io/api/species/'
    planets_url = 'https://swapi-api.hbtn.io/api/planets/'
    sentient_homeworlds = set()
    ordered_planets = []

    # going through all species pages
    while species_url:
        res = requests.get(species_url)
        data = res.json()

        # checking each species
        for species in data.get('results', []):
            # look for sentient in either
            classification = species.get('classification', '').lower()
            designation = species.get('designation', '').lower()

            if classification == 'sentient' or designation == 'sentient':
                homeworld_url = species.get('homeworld')
                if homeworld_url:
                    try:
                        hw_res = requests.get(homeworld_url)
                        hw_data = hw_res.json()
                        name = hw_data.get('name')
                        if name:
                            sentient_homeworlds.add(name)
                    except:
                        continue

        species_url = data.get('next')

    # loop through all planets and keep only the sentient ones
    while planets_url:
        res = requests.get(planets_url)
        data = res.json()

        for planet in data.get('results', []):
            name = planet.get('name')
            if name in sentient_homeworlds:
                ordered_planets.append(name)

        # move to the next
        planets_url = data.get('next')

    return ordered_planets
