#!/usr/bin/env python3
"""List all documents in Python Module"""


def list_all(mongo_collection):
    """Function that lists all documents in a collection:

    mongo_collection will be the pymongo collection object
    Return an empty list if no document in the collection

    """
    return list(mongo_collection.find())
