"""Token-aware global entity resolution; ambiguity is never an ID tie-break."""
import json
import re
from difflib import SequenceMatcher
from config import tuning
from backend.app.settings import SCOPE_SETTINGS

STOPWORDS = {'is', 'a', 'the', 'service', 'services', 'high', 'low', 'cpu',
             'latency', 'error', 'errors', 'slow', 'up', 'down', 'and', 'on'}


def normalize_name(name):
    return ' '.join(re.findall(r'[a-z0-9]+', (name or '').lower()))


def _fuzzy_threshold(query):
    return 0.85 if len(query) < 8 else tuning.LINK_FUZZY_CUTOFF


def score_match(query, name, aliases=None):
    q = normalize_name(query)
    best = 0.0
    for raw in [name, *(aliases or [])]:
        c = normalize_name(raw)
        if not q or not c:
            continue
        if q == c:
            best = 1.0
        elif len(c) > 3 and f' {c} ' in f' {q} ':
            best = max(best, tuning.LINK_CONTAINS_SCORE)
        elif len(q) >= 4 and len(c) >= 4 and q not in STOPWORDS:
            ratio = SequenceMatcher(None, q, c).ratio()
            if ratio >= _fuzzy_threshold(q):
                best = max(best, ratio)
    return best


def _matches(query, nodes):
    tokens = normalize_name(query).split()
    exact = []
    fuzzy = []
    for node in nodes:
        sid = node.get('service_id') if node['type'] == 'Instance' else node['id']
        if not sid:
            continue
        forms = [node['id'], node.get('name', ''), *node.get('aliases', [])]
        for form in forms:
            words = normalize_name(form).split()
            if not words or all(w in STOPWORDS for w in words):
                continue
            for i in range(len(tokens) - len(words) + 1):
                end = i + len(words)
                if tokens[i:end] == words:
                    exact.append((i, end, sid, node, 1.0))
                elif len(words) == 1 and len(words[0]) >= 4 and tokens[i] not in STOPWORDS:
                    score = score_match(tokens[i], form)
                    if score >= _fuzzy_threshold(tokens[i]):
                        fuzzy.append((i, end, sid, node, score))
    # Longer aliases suppress only their nested components, not separate mentions.
    exact = [m for m in exact if not any(n[0] <= m[0] and n[1] >= m[1]
             and n[1]-n[0] > m[1]-m[0] for n in exact)]
    matches = exact or fuzzy
    by_service = {}
    for m in matches:
        prev = by_service.get(m[2])
        if prev is None or (m[4], m[3]['type'] == 'Service') > (prev[4], prev[3]['type'] == 'Service'):
            by_service[m[2]] = m
    return sorted(by_service.values(), key=lambda m: (-m[4], m[2])), bool(exact)


def pick_best(query, nodes):
    matches, exact = _matches(query, nodes)
    if not matches:
        return None
    if len(matches) > 1 and (exact or matches[0][4] - matches[1][4] < SCOPE_SETTINGS.fuzzy_margin):
        return None
    match = matches[0]
    return {'id': match[3]['id'], 'type': match[3]['type'], 'score': round(match[4], 4)}


def link_entity(query, graph, *, settings=None):
    from backend.app.graph.gateway import ReadGateway
    from backend.app.settings import LINKING_SETTINGS
    settings = settings or LINKING_SETTINGS
    rows, overflow = ReadGateway(graph, settings=settings).read(
        'MATCH (n) WHERE n:Service OR n:Instance '
        'RETURN ' + ', '.join(
            f"CASE WHEN size(toJSON(n.{key})) > $field_chars THEN '__payload_rejected__' ELSE n.{key} END"
            if key != 'labels' else 'labels(n)'
            for key in ('id', 'labels', 'name', 'aliases', 'service_id')) + ' ORDER BY n.id',
        cap=settings.max_catalog)
    if overflow:
        # An omitted node can introduce a competing mention or shared alias.
        # Never resolve against a partial catalog, even with an exact first hit.
        return {'query': query, 'matched': False, 'anchor_service': None,
                'nodes': [], 'via_instance': None, 'reason': 'catalog_overflow',
                'truncated': True, 'catalog_cap': settings.max_catalog}
    from backend.app.graph.gateway import admit_payload, PayloadAdmissionError
    try:
        for row in rows:
            admit_payload(row, field_chars=settings.field_chars)
            if any(value == '__payload_rejected__' for value in row if isinstance(value, str)):
                raise PayloadAdmissionError('catalog property rejected')
            i, labels, name, aliases, sid = row
            if (not isinstance(i, str) or len(labels) != 1 or labels[0] not in ('Service', 'Instance')
                    or name is not None and not isinstance(name, str)
                    or sid is not None and not isinstance(sid, str)
                    or aliases is not None and not isinstance(aliases, (str, list))):
                raise PayloadAdmissionError('invalid catalog identity')
            parsed_aliases = json.loads(aliases) if isinstance(aliases, str) else aliases or []
            if not isinstance(parsed_aliases, list) or any(not isinstance(a, str) for a in parsed_aliases):
                raise PayloadAdmissionError('invalid alias')
            admit_payload(parsed_aliases, field_chars=settings.field_chars)
        if len({r[0] for r in rows}) != len(rows):
            raise PayloadAdmissionError('duplicate catalog identity')
    except (PayloadAdmissionError, TypeError, ValueError):
        return {'query': query, 'matched': False, 'anchor_service': None,
                'nodes': [], 'via_instance': None, 'reason': 'catalog_payload_rejected'}
    nodes = [{'id': i, 'type': labels[0], 'name': name or '',
              'aliases': _parse_list(aliases), 'service_id': sid}
             for i, labels, name, aliases, sid in rows if labels]
    services = {n['id'] for n in nodes if n['type'] == 'Service'}
    nodes = [n for n in nodes if n['type'] == 'Service' or n.get('service_id') in services]
    best = pick_best(query, nodes)
    result = {'query': query, 'matched': best is not None, 'anchor_service': None,
              'nodes': [], 'via_instance': None}
    if best:
        node = next(n for n in nodes if n['id'] == best['id'])
        result.update(anchor_service=node.get('service_id') if node['type'] == 'Instance' else node['id'],
                      nodes=[best], via_instance=node['id'] if node['type'] == 'Instance' else None)
    return result


def _parse_list(value):
    if isinstance(value, list):
        return value
    try:
        result = json.loads(value)
        return result if isinstance(result, list) else []
    except (TypeError, ValueError):
        return []
