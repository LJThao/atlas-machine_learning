#!/usr/bin/env python3
"""Multi-reference Question Answering Module"""
from importlib import import_module
semantic_search = import_module("3-semantic_search").semantic_search
answer_question = import_module("0-qa").question_answer
EXIT_COMMANDS = {"exit", "quit", "bye", "goodbye"}


def question_answer(corpus_path):
    """Function that answers questions from multiple reference texts:

    corpus_path is the path to the corpus of reference documents

    """
    while True:
        try:
            q = input("Q: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nA: Goodbye")
            break

        if q.lower() in EXIT_COMMANDS:
            print("A: Goodbye")
            break

        ref = semantic_search(corpus_path, q)
        ans = answer_question(q, ref)
        print("A:", ans or "Sorry, I do not understand your question.")
