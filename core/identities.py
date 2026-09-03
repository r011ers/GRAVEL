"""Identidad canónica de jugadores (normalización mecánica + allowlist)."""

from __future__ import annotations

from typing import Iterable, Sequence

APOSTROPHE_CHARS = (
    "\u00b4",  # acute accent (´)
    "\u2018",  # left single quotation
    "\u2019",  # right single quotation
    "\u02bc",  # modifier letter apostrophe
    "`",
)

HIDDEN_PLAYERS = frozenset({"Richard Rowland Dont USE"})

# canonical display name -> known raw aliases (plus the canonical itself)
ALIAS_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Andy Baetens", ("Andy Baetens", "Andy  Baetens")),
    ("Zvonimir Lesic", ("Zvonimir Lesic", "Lesic Zvonimir")),
    ("Berry Van Peer", ("Berry Van Peer", "Berry van_Peer")),
    ("Danny van Trijp", ("Danny van Trijp", "Danny van_Trijp")),
    ("Keanu van Velzen", ("Keanu van Velzen", "Keanu Van_Velzen")),
    ("Jamai van den Herik", ("Jamai van den Herik", "Jamai Van_Den_Herik")),
    ("Noa-Lynn van Leuven", ("Noa-Lynn van Leuven", "Noa-Lynn van_Leuven_")),
    ("Bradley O'Connor", ("Bradley O'Connor", "Bradley O Connor")),
    ("John O'Shea", ("John O'Shea",)),
)

DO_NOT_MERGE = frozenset(
    {
        "Lee Evans",
        "Lee (ENG) Evans",
        "Lee Evans (WAL)",
        "John McCarthy",
        "Josh McCarthy",
        "Matt Dennant",
        "Matthew Dennant",
        "Llew Bevan",
        "Llew-J Bevan",
        "Josh Richardson",
        "Joshua Richardson",
        "Steve Johnstone",
        "Steven Johnstone",
    }
)


def _replace_apostrophes(value: str) -> str:
    out = value
    for ch in APOSTROPHE_CHARS:
        out = out.replace(ch, "'")
    return out


def normalize_compare(name: str) -> str:
    """Forma comparable: minúsculas, espacios colapsados, `_` → espacio, apóstrofes ASCII."""
    s = str(name).strip().strip("_")
    s = _replace_apostrophes(s)
    s = s.replace("_", " ")
    s = " ".join(s.split())
    return s.lower()


def identity_key(name: str) -> str:
    """Clave de identidad. En nombres de 2 tokens, el orden no importa."""
    n = normalize_compare(name)
    parts = n.split()
    if len(parts) == 2:
        return " ".join(sorted(parts))
    return n


def is_hidden_player(name: str) -> bool:
    raw = str(name).strip()
    if raw in HIDDEN_PLAYERS:
        return True
    return "dont use" in raw.lower()


def _display_score(name: str) -> tuple:
    s = str(name)
    return (
        "_" not in s,
        "  " not in s,
        not s.endswith("_"),
        "'" in s,
        -len(s),
        s,
    )


def preferred_display(names: Sequence[str]) -> str:
    if not names:
        raise ValueError("preferred_display requires at least one name")
    best = max(names, key=_display_score)
    cleaned = " ".join(str(best).strip().strip("_").replace("_", " ").split())
    return cleaned or str(best)


def _allowlist_map() -> dict[str, str]:
    mapping: dict[str, str] = {}
    for canonical, aliases in ALIAS_GROUPS:
        members = (canonical, *aliases)
        keys = set()
        for member in members:
            keys.add(identity_key(member))
            keys.add(normalize_compare(member))
            keys.add(normalize_compare(member).replace("'", ""))
        for key in keys:
            mapping[key] = canonical
    return mapping


_ALLOWLIST = _allowlist_map()


def canonical_player(name: str, all_names: Sequence[str] | None = None) -> str:
    raw = str(name).strip()
    if not raw:
        return raw
    if is_hidden_player(raw):
        return raw

    key = identity_key(raw)
    cmp_key = normalize_compare(raw)
    no_apos = cmp_key.replace("'", "")
    for candidate in (key, cmp_key, no_apos):
        if candidate in _ALLOWLIST:
            return _ALLOWLIST[candidate]

    pool: Iterable[str] = all_names if all_names is not None else (raw,)
    cluster = [
        n
        for n in pool
        if not is_hidden_player(n) and identity_key(n) == key
    ]
    if not cluster:
        cluster = [raw]
    return preferred_display(cluster)


def aliases_of(canonical: str, all_names: Sequence[str]) -> list[str]:
    target = canonical_player(canonical, all_names)
    found = [
        n
        for n in all_names
        if not is_hidden_player(n) and canonical_player(n, all_names) == target
    ]
    if target not in found and not is_hidden_player(target):
        found.append(target)
    return found


def selectable_players(all_names: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for name in all_names:
        if is_hidden_player(name):
            continue
        canonical = canonical_player(name, all_names)
        if canonical in seen:
            continue
        seen.add(canonical)
        out.append(canonical)
    return sorted(out)


def elo_source_name(canonical: str, all_names: Sequence[str], matches_by_name: dict[str, int]) -> str:
    """Nombre crudo cuya fila de ELO se usa (más partidos); no recalcula ELO."""
    aliases = aliases_of(canonical, all_names)
    present = [n for n in aliases if n in matches_by_name]
    if not present:
        return canonical
    return max(present, key=lambda n: (matches_by_name.get(n, 0), n == canonical))
