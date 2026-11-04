"""Bounded per-session message history, replayed into the model prompt on each turn."""
from collections import OrderedDict
from langchain_core.messages import HumanMessage, AIMessage
from app.config import settings
class Memory:
    def __init__(self):
        self.sessions = OrderedDict()
    def get(self, session):
        return list(self.sessions.get(session, []))
    def add(self, session, user, assistant):
        messages = self.get(session) + [HumanMessage(content=user), AIMessage(content=assistant)]
        # Recent turns only; also enforce a character ceiling (not a token counter).
        while messages and (
            len(messages) > settings.max_history_messages
            or sum(len(str(m.content)) for m in messages) > settings.max_history_chars
        ):
            messages = messages[2:]
        self.sessions[session] = messages
        self.sessions.move_to_end(session)
        while len(self.sessions) > 100:
            self.sessions.popitem(last=False)
    def clear(self, session):
        self.sessions.pop(session, None)
