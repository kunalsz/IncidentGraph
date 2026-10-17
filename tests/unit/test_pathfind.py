"""Unit tests for pathfind role assignment (pure part)."""

from backend.app.graph.pathfind import assign_roles


class TestAssignRoles:
    def test_three_hop_path(self):
        roles = assign_roles(["service:frontend", "service:checkout",
                              "service:payment"],
                             anchor="service:frontend",
                             candidate="service:payment")
        assert roles == [
            {"id": "service:frontend", "role": "symptom"},
            {"id": "service:checkout", "role": "intermediate"},
            {"id": "service:payment", "role": "candidate_cause"},
        ]

    def test_candidate_wins_at_length_one(self):
        roles = assign_roles(["service:checkout"], anchor="service:checkout",
                             candidate="service:checkout")
        assert roles == [{"id": "service:checkout", "role": "candidate_cause"}]
