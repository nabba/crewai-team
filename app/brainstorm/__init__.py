"""Brainstorm module — interactive idea-generation sessions.

Conducts structured Q/A brainstorming over Signal or CLI using a library of
techniques (SCAMPER, Six Hats, How-Might-We, Reverse, Crazy-8s, Rapid
Ideation, Starbursting). Each technique is a state machine; the facilitator
drives the user through it and the Writer agent generates a final report.

Public surface:
    - ``techniques.registry()`` — name → Technique mapping
    - ``facilitator.start(sender, technique_name, topic)`` — open a session
    - ``facilitator.respond(sender, message)`` — feed the next answer
    - ``facilitator.finish(sender)`` — close + generate report
    - ``signal_handler.try_handle(text, sender)`` — Signal slash-command hook
    - ``cli`` module — ``python -m app.brainstorm`` interactive REPL
"""
