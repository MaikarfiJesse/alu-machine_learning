#!/usr/bin/env python3
"""Create a method that returns the list of names of the home planets
of all sentient species"""


import requests


def sentientPlanets():
    """
    Uses the Swapi API to return the list of home planets
        for all sentient species

    returns:
        [list]: home planets of sentient species
    """
    url = "https://swapi-api.hbtn.io/api/species/?format=json"
    speciesList = []
    while url:
        results = requests.get(url).json()
        speciesList += results.get('results')
        url = results.get('next')
    homePlanets = []
    for species in speciesList:
        if species.get('designation') == 'sentient' or \
           species.get('classification') == 'sentient':
            url = species.get('homeworld')
            if url:
                planet = requests.get(url).json()
                homePlanets.append(planet.get('name'))
    return homePlanets
