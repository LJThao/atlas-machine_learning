#!/usr/bin/env python3
"""Log stats Module"""
from pymongo import MongoClient


def log_stats():
    """Function that provides some stats about Nginx logs stored in
    MongoDB:

    Database: logs
    Collection: nginx
    Display examples using dump.zip

    """
    collection = MongoClient().logs.nginx

    print(f"{collection.count_documents({})} logs\nMethods:")
    for method in ["GET", "POST", "PUT", "PATCH", "DELETE"]:
        print(
            f"\tmethod {method}: "
            f"{collection.count_documents({'method': method})}")

    print(f"{collection.count_documents({'method': 'GET',
                                         'path': '/status'})} status check")


if __name__ == "__main__":
    log_stats()
