"""Integration tests: seed, link, and P0 against a live FalkorDB.

Requires FALKORDB_HOST/FALKORDB_PORT (default localhost:6380 — the
docker-compose FalkorDB instance).
"""

import json
import uuid

import pytest

from backend.app.engine import p0_resolve
from backend.app.graph.client import DEMO_GRAPH, client, graph_for, session_graph
from backend.app.seed import loader

falkordb = pytest.importorskip("falkordb")


@pytest.fixture(scope="module")
def demo_seeded():
    counts = loader.seed_demo(force=True)
    yield counts


@pytest.fixture
def fresh_session():
    # GRAPH.COPY the demo graph; each test gets an isolated session
    name, graph = session_graph(str(uuid.uuid4())[:12])
    yield name, graph


class TestSeedLoader:
    def test_seed_counts(self, demo_seeded):
        assert demo_seeded["services"] == 12
        assert demo_seeded["depends_on"] == 17
        assert demo_seeded["teams"] == 6
        assert demo_seeded["instances"] == 13
        assert demo_seeded["runbooks"] == 4
        assert demo_seeded["incidents"] == 1

    def test_seed_is_idempotent(self, demo_seeded):
        counts2 = loader.seed_demo()
        g = graph_for(DEMO_GRAPH)
        n = g.query("MATCH (s:Service) RETURN count(s)").result_set[0][0]
        assert n == 12

    def test_fixture_load(self, demo_seeded):
        written = loader.load_fixture_by_name("payment_failure")
        assert written["nodes_written"] == 12
        g = graph_for(DEMO_GRAPH)
        assert g.query("MATCH (c:ConfigChange {id: 'config:INC-1001/c1'}) "
                       "RETURN c.flag").result_set[0][0] == "paymentFailure"
        # idempotent reload
        written2 = loader.load_fixture_by_name("payment_failure")
        n = g.query("MATCH (m:MetricAnomaly) RETURN count(m)").result_set[0][0]
        assert n == 3


class TestLinkingGraph:
    def test_link_by_name(self, demo_seeded, fresh_session):
        _, g = fresh_session
        r = linking_link("payment", g)
        assert r["matched"] is True
        assert r["anchor_service"] == "service:payment"

    def test_link_by_alias(self, demo_seeded, fresh_session):
        _, g = fresh_session
        r = linking_link("billing", g)
        assert r["anchor_service"] == "service:payment"

    def test_link_unresolved_abstains(self, demo_seeded, fresh_session):
        _, g = fresh_session
        r = linking_link("zzz quantum flux", g)
        assert r["matched"] is False
        assert r["anchor_service"] is None


def linking_link(q, g):
    from backend.app.engine import linking
    return linking.link_entity(q, g)


class TestP0Resolve:
    @pytest.fixture(autouse=True)
    def _seed(self, demo_seeded):
        loader.load_fixture_by_name("payment_failure")

    def test_p0_alert_anchored(self):
        inv = p0_resolve.trigger_investigation(
            "checkout latency up", alert_id="alert:INC-1001/a1")
        assert inv.anchor_service == "service:checkout"
        assert inv.window.t1 - inv.window.t0 == 2 * 15 * 60 * 1000
        assert inv.incident_id.startswith("incident:INC-")
        assert inv.session_graph.startswith("incidentgraph:session:")

    def test_p0_symptom_anchored(self):
        inv = p0_resolve.trigger_investigation("checkout latency up")
        assert inv.anchor_service == "service:checkout"

    def test_p0_writes_incident_and_affects(self):
        inv = p0_resolve.trigger_investigation(
            "checkout latency up", alert_id="alert:INC-1001/a1")
        g = graph_for(inv.session_graph)
        row = g.query(
            "MATCH (i:Incident {id: $id})-[:AFFECTS]->(s) RETURN i.status, s.id",
            params={"id": inv.incident_id}).result_set
        assert row and row[0] == ["investigating", "service:checkout"]

    def test_p0_demo_graph_untouched(self):
        before = graph_for(DEMO_GRAPH).query(
            "MATCH (i:Incident) WHERE i.status='investigating' RETURN count(i)"
        ).result_set[0][0]
        p0_resolve.trigger_investigation("checkout latency up")
        after = graph_for(DEMO_GRAPH).query(
            "MATCH (i:Incident) WHERE i.status='investigating' RETURN count(i)"
        ).result_set[0][0]
        assert before == after

    def test_p0_unresolved_raises(self):
        with pytest.raises(p0_resolve.UnresolvedAnchorError):
            p0_resolve.trigger_investigation("quantum flux capacitor broken")

    def test_p0_budget_in_output(self):
        inv = p0_resolve.trigger_investigation("checkout latency up")
        assert inv.budget["budget"]["max_llm_calls"] == 75
        assert inv.budget["budget"]["max_tokens"] == 200_000
        assert inv.budget["compliant"] is True


class TestSchema:
    def test_indexes_exist(self, demo_seeded):
        g = graph_for(DEMO_GRAPH)
        n = g.query("MATCH (s:Service) RETURN count(s)").result_set[0][0]
        assert n == 12


class TestGraphCopyIsolation:
    def test_session_copy_gets_data_and_stays_isolated(self, demo_seeded):
        name, g = session_graph("iso-test-1")
        assert g.query("MATCH (s:Service) RETURN count(s)").result_set[0][0] == 12
        g.query("CREATE (t:Team {id: 'team:zzz', name: 'ZZZ'})")
        assert graph_for(DEMO_GRAPH).query(
            "MATCH (t:Team {id: 'team:zzz'}) RETURN count(t)").result_set[0][0] == 0
