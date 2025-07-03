#!/usr/bin/env python3
"""Insert a document in Python"""


def insert_school(mongo_collection, **kwargs):
    """Function that inserts a new document in a collection based on kwargs:

    mongo_collection will be the pymongo collection object
    Returns the new _id

    """
    return mongo_collection.insert_one(kwargs).inserted_id
