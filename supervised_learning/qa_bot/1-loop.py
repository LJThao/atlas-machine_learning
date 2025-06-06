#!/usr/bin/env python3
"""Create the loop Module
Create a script that takes in input from the user with the prompt Q: and
prints A: as a response. If the user inputs exit, quit, goodbye, or bye,
case insensitive, print A: Goodbye and exit.
"""

# QA loop
while True:
    question = input("Q: ")
    if question.lower() in ["exit", "quit", "goodbye", "bye"]:
        print("A: Goodbye")
        break
    print("A:")
