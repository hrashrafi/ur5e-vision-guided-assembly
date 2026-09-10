"""
Minimal finite-state-machine base for multi-step task sequences (pick,
transport, and later insert-and-screw). Deliberately small: a sequence is
just a dict of {state_name: handler}, where each handler inspects the
current sim state and returns the next state name (or the same name to
keep waiting, or a designated terminal state to finish/fail).
"""

from dataclasses import dataclass, field


@dataclass
class StateMachine:
    handlers: dict          # {state_name: callable(ctx) -> next_state_name}
    start_state: str
    done_states: tuple = ("DONE",)
    fail_states: tuple = ("FAILED",)
    state: str = field(init=False)
    history: list = field(default_factory=list)

    def __post_init__(self):
        self.state = self.start_state

    def reset(self):
        self.state = self.start_state
        self.history = []

    def tick(self, ctx):
        """Run one handler call for the current state, transition, and
        return the new state name."""
        if self.is_terminal():
            return self.state
        next_state = self.handlers[self.state](ctx)
        if next_state != self.state:
            self.history.append((self.state, next_state))
        self.state = next_state
        return self.state

    def is_done(self):
        return self.state in self.done_states

    def is_failed(self):
        return self.state in self.fail_states

    def is_terminal(self):
        return self.is_done() or self.is_failed()
