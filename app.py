import streamlit as st
import random, uuid, json, os, time
from dataclasses import dataclass, field, asdict
from itertools import combinations
from streamlit_autorefresh import st_autorefresh
import pandas as pd
from io import StringIO

SAVE_FILE = "tournament.json"

POD_SIZE = 4
MIN_POD = 3
MAX_POD = 4
WIN_POINTS = 5
TIMEOUT_POINTS = 1

# ---------- Data Models ----------

@dataclass
class Player:
    id: str
    name: str
    points: int = 0
    dropped: bool = False
    opponents: set = field(default_factory=set)
    seed: float = field(default_factory=random.random)

    def to_json(self):
        d = asdict(self)
        d["opponents"] = list(self.opponents)
        return d

    @staticmethod
    def from_json(d):
        p = Player(
            id=d["id"],
            name=d["name"],
            points=d.get("points", 0),
            dropped=d.get("dropped", False),
            seed=d.get("seed", random.random())
        )
        p.opponents = set(d.get("opponents", []))
        return p

@dataclass
class Pod:
    players: list           # list of player ids
    winner: str | None = None   # player id, "Tie", or None
    tie_recipients: list[str] = field(default_factory=list)  # ids who got 1pt on a tie

@dataclass
class Round:
    number: int
    pods: list              # list[Pod]
    start_ts: float | None = None
    duration_minutes: int | None = None
    timeout_awarded: bool = False

# ---------- Persistence ----------

def load_state():
    if "players" in st.session_state:
        return
    if os.path.exists(SAVE_FILE):
        with open(SAVE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        st.session_state.players = {pid: Player.from_json(pd) for pid, pd in data["players"].items()}
        st.session_state.rounds = []
        for r in data["rounds"]:
            pods = [Pod(p["players"], p.get("winner"), p.get("tie_recipients", [])) for p in r["pods"]]
            st.session_state.rounds.append(Round(
                number=r["number"],
                pods=pods,
                start_ts=r.get("start_ts"),
                duration_minutes=r.get("duration_minutes"),
                timeout_awarded=r.get("timeout_awarded", False)
            ))
        st.session_state.pod_history = set(data.get("pod_history", []))  # set of "pid1|pid2|pid3|pid4" (sorted)
    else:
        st.session_state.players = {}
        st.session_state.rounds = []
        st.session_state.pod_history = set()

def save_state():
    with open(SAVE_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "players": {pid: st.session_state.players[pid].to_json() for pid in st.session_state.players},
            "rounds": [{
                "number": r.number,
                "pods": [{"players": p.players, "winner": p.winner, "tie_recipients": p.tie_recipients} for p in r.pods],
                "start_ts": r.start_ts,
                "duration_minutes": r.duration_minutes,
                "timeout_awarded": r.timeout_awarded
            } for r in st.session_state.rounds],
            "pod_history": list(st.session_state.pod_history)
        }, f, indent=2)


def create_backup(round):
    
    with open("backup_"+str(round.number)+".json", "w", encoding="utf-8") as f:
        json.dump({
            "players": {pid: st.session_state.players[pid].to_json() for pid in st.session_state.players},
            "rounds": [{
                "number": r.number,
                "pods": [{"players": p.players, "winner": p.winner} for p in r.pods],
                "start_ts": r.start_ts,
                "duration_minutes": r.duration_minutes,
                "timeout_awarded": r.timeout_awarded
            } for r in st.session_state.rounds],
            "pod_history": list(st.session_state.pod_history)
        }, f, indent=2)


# ---------- Helpers ----------
def rounds_started() -> bool:
    return len(st.session_state.rounds) > 0

def player_has_matches(pid: str) -> bool:
    """Consider a player 'has matches' if they ever appeared in a pod or have points/opponents."""
    p = st.session_state.players.get(pid)
    if not p:
        return False
    if p.points > 0 or p.opponents:
        return True
    for rnd in st.session_state.rounds:
        for pod in rnd.pods:
            if pid in pod.players or pod.winner == pid:
                return True
    return False

def delete_player(pid: str):
    """Delete only before rounds start; otherwise tell user to Drop."""
    if rounds_started():
        st.info("Deletion disabled after Round 1 starts. Use Drop instead.")
        return
    if player_has_matches(pid):
        st.info("Cannot delete a player who has participated. Use Drop instead.")
        return

    # Defensive cleanup (should be no rounds yet, but just in case)
    for rnd in st.session_state.rounds:
        for pod in rnd.pods:
            if pod.winner is None and pid in pod.players:
                pod.players = [x for x in pod.players if x != pid]

    # Remove any pod-history entries that mention this player
    st.session_state.pod_history = set(
        sig for sig in st.session_state.pod_history
        if pid not in sig.split("|")
    )

    # Remove the player
    st.session_state.players.pop(pid, None)
    save_state()
    st.rerun()
    
def pod_key(pid_list):
    return "|".join(sorted(pid_list))

def exact_pod_exists(pid_list):
    return pod_key(pid_list) in st.session_state.pod_history

def record_pods_in_history(pods):
    for pod in pods:
        st.session_state.pod_history.add(pod_key(pod.players))

def not_dropped_players():
    return [p for p in st.session_state.players.values() if not p.dropped]

def can_join(pid, pod):
    # Players can face each other again across rounds; the user only asked to avoid *exact same pod*.
    # So this always returns True; we’ll enforce the "no exact pod repeat" at the pod level.
    return True

def pod_sizes_only_3_4(n: int):
    """Return [4,4,...,3,3,...] or None if impossible (only 3- and 4-player pods allowed)."""
    if n in (1, 2, 5):
        return None
    r = n % 4
    needed_3 = {0: 0, 1: 3, 2: 2, 3: 1}[r]
    if r == 1 and n < 9:  # smallest r==1 split is 9 = 3+3+3
        return None
    num4 = (n - 3 * needed_3) // 4
    return [4] * num4 + [3] * needed_3

def pair_round(duration_minutes: int | None):
    players = [p for p in not_dropped_players()]
    n = len(players)

    if n < MIN_POD:
        st.error("Need at least 3 active (not dropped) players to pair a round.")
        return

    # With only 3- and 4-player pods allowed, 5 players cannot be partitioned.
    if n in (1, 2, 5):
        st.error("Cannot form only 3- or 4-player pods with 5 players. Add or drop a player.")
        return

    # Sort by points desc then seed for stability/randomness
    players.sort(key=lambda p: (-p.points, p.seed))
    ids = [p.id for p in players]   

    # Decide exact pod sizes we want: only 3s and 4s, never >4
    r = n % 4
    # Minimal number of 3-pods to cover leftovers (classic composition)
    needed_3 = {0: 0, 1: 3, 2: 2, 3: 1}[r]
    # (For r==1 we need at least 9 players; n==5 already handled above.)
    if r == 1 and n < 9:
        st.error("With 5 players disallowed, the smallest valid 'r==1' case is 9 (three 3-pods). Add/drop players.")
        return
    num4 = (n - 3 * needed_3) // 4
    # ---- key change: reserve the lowest-point players for the 3-pods ----
    low_block = ids[-3 * needed_3:] if needed_3 else []
    high_block = ids[: n - 3 * needed_3]

    def form_pods_from_pool(pool_ids, pod_size, max_tries=250):
        """Greedy with small local search to avoid exact pod repeats."""
        if not pool_ids:
            return []
        pods_local = None
        base = pool_ids[:]  # keep order mostly (Swiss-y)
        for _ in range(max_tries):
            tmp = base[:]
            # tiny shuffle to escape dead-ends without breaking bracket intent
            random.shuffle(tmp)
            built = []
            ok = True
            while len(tmp) >= pod_size:
                candidate = tmp[:pod_size]
                if exact_pod_exists(candidate):
                    made = False
                    cap = min(len(tmp), max(pod_size + 6, 12))
                    for combo in combinations(tmp[:cap], pod_size):
                        combo = list(combo)
                        if not exact_pod_exists(combo):
                            candidate = combo
                            made = True
                            break
                    if not made:
                        ok = False
                        break
                chosen = set(candidate)
                tmp = [x for x in tmp if x not in chosen]
                built.append(Pod(players=candidate))
            if ok and not tmp:
                pods_local = built
                break
        return pods_local

    pods: list[Pod] = []

    # 1) Build 3-pods from the lowest-point players (priority to low points)
    if needed_3:
        pods3 = form_pods_from_pool(low_block, 3)
        if pods3 is None:
            st.error("Could not form 3-pods without repeats. Toggle a drop or try again.")
            return
        pods.extend(pods3)

    # 2) Build 4-pods from the rest (higher points)
    if num4:
        pods4 = form_pods_from_pool(high_block, 4)
        if pods4 is None:
            st.error("Could not form 4-pods without repeats. Toggle a drop or try again.")
            return
        # show higher bracket first (nice for display)
        pods = pods4 + pods

    # Create round (timer etc.)
    rno = len(st.session_state.rounds) + 1
    rnd = Round(
        number=rno,
        pods=pods,
        start_ts=time.time() if duration_minutes else None,
        duration_minutes=duration_minutes
    )
    st.session_state.rounds.append(rnd)

    # Update opponent history
    players_map = st.session_state.players
    for pod in pods:
        assert 3 <= len(pod.players) <= 4, "Pod size invariant violated"
        for a in pod.players:
            for b in pod.players:
                if a != b:
                    players_map[a].opponents.add(b)

    record_pods_in_history(pods)
    save_state()

def pair_round_swiss(duration_minutes: int | None):
    players = [p for p in not_dropped_players()]
    n = len(players)

    if n < MIN_POD:
        st.warning("Need at least 3 active (not dropped) players to pair a round.")
        return

    sizes = pod_sizes_only_3_4(n)
    if sizes is None:
        st.warning("Current active players cannot be partitioned into only 3- and 4-player pods. Add/drop players.")
        return

    # Swiss order: higher points first (seed tiebreaker for equal points)
    players.sort(key=lambda p: (-p.points, p.seed))
    ids = [p.id for p in players]

    # Minimal number of 3-pods, assigned to the lowest scorers
    num3 = sizes.count(3)
    low_block  = ids[-3 * num3:] if num3 else []
    high_block = ids[: n - 3 * num3]

    pods: list[Pod] = []

    # 1) Build 4-pods from the top block, left-to-right, deterministic (no shuffles)
    for i in range(0, len(high_block), 4):
        pods.append(Pod(players=high_block[i:i+4]))

    # 2) Build 3-pods from the reserved bottom block, left-to-right
    for i in range(0, len(low_block), 3):
        pods.append(Pod(players=low_block[i:i+3]))

    # Commit round
    rno = len(st.session_state.rounds) + 1
    rnd = Round(
        number=rno,
        pods=pods,
        start_ts=time.time() if duration_minutes else None,
        duration_minutes=duration_minutes
    )
    st.session_state.rounds.append(rnd)

    # Update opponent history (optional but harmless for tiebreakers like A-SOS)
    pm = st.session_state.players
    for pod in pods:
        for a in pod.players:
            for b in pod.players:
                if a != b:
                    pm[a].opponents.add(b)

    # (We intentionally do NOT record pod signatures — repetition is allowed.)
    save_state()
def submit_results(rno, winners, custom_ties=None):
    custom_ties = custom_ties or {}
    rnd = st.session_state.rounds[rno-1]
    players = st.session_state.players
    name_to_id = {p.name.lower(): p.id for p in players.values()}

    for i, pod in enumerate(rnd.pods):
        if i not in winners:
            continue
        w = winners[i]

        if w == 'Tie (everyone)':
            pod.winner = "Tie"
            pod.tie_recipients = pod.players[:]
            for pid in pod.players:
                players[pid].points += 1
            continue

        if w == 'Tie (custom)':
            ids = [pid for pid in custom_ties.get(i, []) if pid in pod.players]
            if not ids:
                st.warning(f"No players selected for Tie in Pod {i+1}.")
                continue
            pod.winner = "Tie"
            pod.tie_recipients = ids
            for pid in ids:
                players[pid].points += 1
            continue

        # normal single-winner (accept name or id)
        wid = players[w].id if w in players else name_to_id.get(w.lower())
        if wid is None or wid not in pod.players:
            st.error(f"Winner '{w}' not in Pod {i+1}.")
            return
        pod.winner = wid
        pod.tie_recipients = []
        players[wid].points += WIN_POINTS

    save_state()
    create_backup(rnd)
def change_pod_winner(rno: int, pod_index: int, new_winner_label: str, tie_ids: list[str] | None = None):
    rnd = st.session_state.rounds[rno - 1]
    pod = rnd.pods[pod_index]
    players = st.session_state.players
    name_to_id = {p.name.lower(): p.id for p in players.values()}

    # rollback old
    prev = pod.winner
    if prev is not None:
        if prev == "Tie":
            recips = pod.tie_recipients if pod.tie_recipients else pod.players
            for pid in recips:
                players[pid].points = max(0, players[pid].points - 1)
        else:
            if prev in players:
                players[prev].points = max(0, players[prev].points - WIN_POINTS)

    # apply new
    if new_winner_label in (None, "", "None", "None (clear)"):
        pod.winner = None
        pod.tie_recipients = []
    elif new_winner_label in ("Tie", "Tie (everyone)"):
        pod.winner = "Tie"
        pod.tie_recipients = pod.players[:]
        for pid in pod.players:
            players[pid].points += 1
    elif new_winner_label == "Tie (custom)":
        tie_ids = [pid for pid in (tie_ids or []) if pid in pod.players]
        if not tie_ids:
            st.error(f"No valid players selected for Tie in Pod {pod_index+1}.")
            return
        pod.winner = "Tie"
        pod.tie_recipients = tie_ids
        for pid in tie_ids:
            players[pid].points += 1
    else:
        wid = new_winner_label if new_winner_label in players else name_to_id.get(new_winner_label.lower())
        if wid is None or wid not in pod.players:
            st.error(f"New winner '{new_winner_label}' is not in Pod {pod_index+1}.")
            return
        pod.winner = wid
        pod.tie_recipients = []
        players[wid].points += WIN_POINTS

    save_state()
    create_backup(rnd)
    st.rerun()

def export_standings_csv():
    df = pd.DataFrame(standings_rows())
    return df.to_csv(index=False)

def export_history_csv():
    """Flatten all rounds and pods into a simple CSV table."""
    rows = []
    for rnd in st.session_state.rounds:
        for i, pod in enumerate(rnd.pods, start=1):
            players = [st.session_state.players[pid].name for pid in pod.players]
            if pod.winner == "Tie":
                winner_name = "Tie"
            elif pod.winner and pod.winner in st.session_state.players:
                winner_name = st.session_state.players[pod.winner].name
            elif pod.winner:
                winner_name = pod.winner
            else:
                winner_name = ""
            rows.append({
                "Round": rnd.number,
                "Pod": i,
                "Players": ", ".join(players),
                "Winner": winner_name,
                "Duration (min)": rnd.duration_minutes or "",
                "Timeout Awarded": rnd.timeout_awarded,
            })
    df = pd.DataFrame(rows)
    return df.to_csv(index=False)
def standings_rows():
    players = st.session_state.players

    # Per-appearance SOS (handles repeats & 3/4 pods correctly)
    sos_sum = {pid: 0 for pid in players}
    opp_apps = {pid: 0 for pid in players}

    for rnd in st.session_state.rounds:
        for pod in rnd.pods:
            ids = [pid for pid in pod.players if pid in players]
            for a in ids:
                for b in ids:
                    if a == b:
                        continue
                    sos_sum[a] += players[b].points
                    opp_apps[a] += 1

    a_sos = {
        pid: (sos_sum[pid] / opp_apps[pid]) if opp_apps[pid] > 0 else 0.0
        for pid in players
    }

    # (Optional) keep old unique-opponent SOS for reference/debug
    # uniq_sos = {pid: sum(players[q].points for q in players[pid].opponents if q in players) for pid in players}
    # uniq_cnt = {pid: len([q for q in players[pid].opponents if q in players]) for pid in players}

    ordered = sorted(
        players.values(),
        key=lambda x: (-x.points, -a_sos[x.id], x.name)
    )

    rows = []
    for idx, p in enumerate(ordered, start=1):
        rows.append({
            "Rank": idx,
            "Name": p.name,
            "Points": p.points,
            "A-SOS": round(a_sos[p.id], 3),
            "Opp Apps": opp_apps[p.id],          # how many opponent slots faced
            # "SOS(unique)": uniq_sos.get(p.id, 0),  # optional debug columns
            # "Opp Unique": uniq_cnt.get(p.id, 0),
            "Status": "Dropped" if p.dropped else ""
        })
    return rows

# def award_timeout_points(rno):
#     rnd = st.session_state.rounds[rno-1]
#     if rnd.timeout_awarded:
#         st.info("Timeout points already awarded for this round.")
#         return
#     players = st.session_state.players
#     # Everyone in the round gets 1 point (even if someone already has a win, we won’t double-award)
#     awarded = set()
#     for pod in rnd.pods:
#         for pid in pod.players:
#             if pid not in awarded:
#                 players[pid].points += TIMEOUT_POINTS
#                 awarded.add(pid)
#     rnd.timeout_awarded = True
#     save_state()
#     st.success("Awarded 1 point to everyone for this round.")

def time_remaining(rnd: Round):
    if not (rnd.start_ts and rnd.duration_minutes):
        return None
    end_ts = rnd.start_ts + rnd.duration_minutes * 60
    remaining = int(end_ts - time.time())
    return remaining

# ---------- UI ----------

st.set_page_config(page_title="Commander Tournament", page_icon="🎴", layout="wide")
st.title("Commander Tournament Fall 2025")

load_state()
with st.sidebar:
    st.header("Players")

    # Add player form
    with st.form("add_player_form", clear_on_submit=True):
        name = st.text_input("Add player name")
        submitted = st.form_submit_button("Add Player")
        if submitted and name.strip():
            pid = str(uuid.uuid4())[:8]
            st.session_state.players[pid] = Player(pid, name.strip())
            save_state()
            st.rerun()

    # Reset section
    if st.button("Reset All", use_container_width=True):
        st.session_state.show_reset_confirm = True

    if st.session_state.get("show_reset_confirm", False):
        st.warning("Are you sure you want to reset the game (this can't be undone)?")
        c1, c2 = st.columns(2)  # these columns are inside the sidebar block
        with c1:
            if st.button("✅ Yes", use_container_width=True):
                st.session_state.clear()
                if os.path.exists(SAVE_FILE):
                    os.remove(SAVE_FILE)
                st.rerun()
        with c2:
            if st.button("❌ Cancel", use_container_width=True):
                st.session_state.show_reset_confirm = False
                st.rerun()

    st.divider()
    st.subheader("Active players")

    if st.session_state.players:
        st.caption("Drop = exclude from future pairings. Delete = only before Round 1 starts.")
        if rounds_started():
            st.info("Round 1 has started — deletion disabled. Use Drop instead.")

        for p in sorted(st.session_state.players.values(), key=lambda x: x.name.lower()):
            row_left, row_right = st.columns([8, 2])  # inside sidebar block
            with row_left:
                dropped = st.checkbox(p.name, value=p.dropped, key=f"drop_{p.id}")
                if dropped != p.dropped:
                    p.dropped = dropped
                    save_state()
            with row_right:
                can_delete = (not rounds_started()) and (not player_has_matches(p.id))
                if st.button("🗑️", key=f"del_{p.id}", disabled=not can_delete, help="Delete player (only before Round 1)"):
                    delete_player(p.id)
    else:
        st.info("No players yet — add some above.")

# --- Pairings & Timer ---
st.subheader("Pairings")
st_autorefresh(interval=5000, key="timer_refresh")
left, right = st.columns([1,1])
with left:
    active_count = len(not_dropped_players())
    st.write(f"Active players: **{active_count}**")
    duration = st.number_input("Round timer (minutes, optional)", min_value=0, max_value=180, value=0, step=5)

    show_swiss = st.toggle("Swiss Mode", value=False, key="swiss_toggle")
    if not show_swiss:
        if st.button("Generate Next Round", use_container_width=True, disabled=active_count < MIN_POD):
            pair_round(duration if duration > 0 else None)
            st.rerun()
    else:
        if st.button("Generate Next Round", use_container_width=True, disabled=active_count < MIN_POD):
            pair_round_swiss(duration if duration > 0 else None)
            st.rerun()

with right:
    if st.session_state.rounds:
        rnd = st.session_state.rounds[-1]
        rem = time_remaining(rnd)
        if rem is not None:
            if rem > 0:
                st.metric("Time remaining in current round", f"{rem//60:02d}:{rem%60:02d}")
            else:
                st.metric("Time remaining in current round", "00:00")
                # if not rnd.timeout_awarded:
                #     if st.button("Time's Up → Award 1 point to everyone this round", type="primary"):
                #         award_timeout_points(rnd.number)
                #         st.rerun()
        else:
            st.info("No active timer for the latest round.")

# --- Rounds Display ---
if st.session_state.rounds:
    for rnd in st.session_state.rounds[::-1]:  # latest first
        with st.expander(f"Round {rnd.number}"):
            # Pods
            for i, pod in enumerate(rnd.pods, start=1):
                names = [st.session_state.players[pid].name for pid in pod.players]
                line = f"**Pod {i}:** " + ", ".join(names)
                if pod.winner:
                    if pod.winner == 'Tie':
                        line += f" — Winner: **None/Tie**"
                    else:
                        line += f" — Winner: **{st.session_state.players[pod.winner].name}**"
                st.markdown(line)
            
            # --- Change winners (toggleable) ---
            show_cw = st.toggle("Change Winner", value=False, key=f"cw_toggle_{rnd.number}")

            if show_cw:
                st.markdown("##### Edit results")

                # optional: tiny CSS so select + button align nicely
                st.markdown("""
                    <style>
                    /* reduce vertical gap under selectboxes for a tighter row */
                    div[data-baseweb="select"] > div { margin-bottom: 0.25rem; }
                    </style>
                """, unsafe_allow_html=True)

                for i, pod in enumerate(rnd.pods):
                    # Build options (Clear, Tie, and each player name)
                    options = ["None (clear)", "Tie (everyone)", "Tie (custom)"] + [st.session_state.players[pid].name for pid in pod.players]

                    current_label = "None (clear)"
                    if pod.winner == "Tie":
                        # show as custom if not everyone got a point
                        if pod.tie_recipients and set(pod.tie_recipients) != set(pod.players):
                            current_label = "Tie (custom)"
                        else:
                            current_label = "Tie (everyone)"
                    elif pod.winner in st.session_state.players:
                        current_label = st.session_state.players[pod.winner].name

                    st.markdown(f"**Pod {i+1}**")
                    c1, c2 = st.columns([4, 1])

                    with c1:
                        sel = st.selectbox(
                            "new winner",
                            options,
                            index=options.index(current_label) if current_label in options else 0,
                            key=f"cw_sel_{rnd.number}_{i}",
                            label_visibility="collapsed",
                        )

                    # When Tie (custom), render a checkbox for each player
                    tie_ids = []
                    if sel == "Tie (custom)":
                        st.caption("Who gets 1 point?")
                        cols = st.columns(len(pod.players))
                        for j, pid in enumerate(pod.players):
                            pname = st.session_state.players[pid].name
                            default_checked = pid in (pod.tie_recipients or [])
                            with cols[j]:
                                chk = st.checkbox(pname, key=f"cw_tie_{rnd.number}_{i}_{pid}", value=default_checked)
                                if chk:
                                    tie_ids.append(pid)

                    with c2:
                        if st.button("Apply", key=f"cw_apply_{rnd.number}_{i}", use_container_width=True):
                            change_pod_winner(rnd.number, i, sel, tie_ids=tie_ids)


            # Winner entry (if any pod has no winner and timeout not yet used to end the round)
            # Winner entry (if any pod has no winner)
            if any(p.winner is None for p in rnd.pods):
                st.markdown("#### Enter Winners")
                winners = {}        # pod_index -> selection label
                custom_ties = {}    # pod_index -> [player_ids]

                for i, pod in enumerate(rnd.pods):
                    if pod.winner:
                        continue

                    name_by_id = {pid: st.session_state.players[pid].name for pid in pod.players}
                    options = [name_by_id[pid] for pid in pod.players]
                    options += ["Tie (everyone)", "Tie (custom)"]

                    winners[i] = st.selectbox(
                        f"Winner Pod {i+1}",
                        [""] + options,
                        key=f"w_{rnd.number}_{i}"
                    )

                    if winners[i] == "Tie (custom)":
                        st.caption(f"Select who gets 1 point in Pod {i+1}:")
                        chosen_ids = []
                        cols = st.columns(len(pod.players))
                        for j, pid in enumerate(pod.players):
                            pname = name_by_id[pid]
                            with cols[j]:
                                chk = st.checkbox(pname, key=f"w_custom_{rnd.number}_{i}_{pid}")
                                if chk:
                                    chosen_ids.append(pid)
                        custom_ties[i] = chosen_ids

                if st.button("Save Results", key=f"save_{rnd.number}"):
                    clean = {i: w for i, w in winners.items() if w}
                    submit_results(rnd.number, clean, custom_ties)
                    st.rerun()

# --- Standings ---
st.subheader("Standings")
if st.session_state.players:
    st.dataframe(standings_rows(), hide_index=True, use_container_width=True)
else:
    st.info("Add players to begin.")

with st.expander("📤 Export Data"):
    st.download_button("📥 Game History CSV", export_history_csv(), "tournament_history.csv", "text/csv")
    st.download_button("📊 Standings CSV", export_standings_csv(), "tournament_standings.csv", "text/csv")
