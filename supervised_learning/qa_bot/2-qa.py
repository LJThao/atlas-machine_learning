#!/usr/bin/enb python3
"""Answer Questions Module"""
question_answer = __import__('0-qa').question_answer
EXIT_COMMANDS = {"exit", "quit", "bye", "goodbye"}


def answer_loop(reference):
    """Function that answers questions from a reference text:

    reference is the reference text
    If the answer cannot be found in the reference text, respond
    with Sorry, I do not understand your question.

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

        ans = question_answer(q, reference)
        print("A:", ans or "Sorry, I do not understand your question.")
