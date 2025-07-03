#!/usr/bin/env python3
"""Where can I learn Python? Module"""


def schools_by_topic(mongo_collection, topic):
    """Function that returns the list of school having a specific topic:

    mongo_collection will be the pymongo collection object
    topic (string) will be topic searched

    """
    return list(mongo_collection.find({"topics": topic}))
