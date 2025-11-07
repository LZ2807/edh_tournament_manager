import streamlit as st
import random, uuid, json, os, time
from dataclasses import dataclass, field, asdict
from itertools import combinations
from streamlit_autorefresh import st_autorefresh

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
    winner: str | None = None

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
            pods = [Pod(p["players"], p.get("winner")) for p in r["pods"]]
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
                "pods": [{"players": p.players, "winner": p.winner} for p in r.pods],
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
    pool = [p.id for p in players]

    # Decide exact pod sizes we want: only 3s and 4s, never >4
    r = n % 4
    # Minimal number of 3-pods to cover leftovers (classic composition)
    needed_3 = {0: 0, 1: 3, 2: 2, 3: 1}[r]
    # (For r==1 we need at least 9 players; n==5 already handled above.)
    if r == 1 and n < 9:
        st.error("With 5 players disallowed, the smallest valid 'r==1' case is 9 (three 3-pods). Add/drop players.")
        return
    num4 = (n - 3 * needed_3) // 4
    sizes = [4] * num4 + [3] * needed_3  # e.g., [4,4,3] etc.

    pods: list[Pod] = []
    MAX_TRIES = 300
    success = False

    for _ in range(MAX_TRIES):
        temp_pool = pool[:]
        random.shuffle(temp_pool)  # try a different order each attempt
        temp_pods: list[Pod] = []
        ok = True

        for size in sizes:
            # Greedy take 'size' players; avoid exact pod repeats
            chosen = temp_pool[:size]
            # If this exact pod existed, try to tweak by swapping one seat
            if exact_pod_exists(chosen):
                made = False
                # Try some alternative combinations among the first few candidates
                cap = min(len(temp_pool), max(size + 6, 12))
                for combo in combinations(temp_pool[:cap], size):
                    combo = list(combo)
                    if not exact_pod_exists(combo):
                        chosen = combo
                        made = True
                        break
                if not made:
                    ok = False
                    break

            # Commit this pod and remove its players from pool
            pid_set = set(chosen)
            temp_pool = [x for x in temp_pool if x not in pid_set]
            temp_pods.append(Pod(players=chosen))

        if ok and len(temp_pool) == 0:
            pods = temp_pods
            success = True
            break

    if not success:
        st.error("Could not form pods without repeats under current constraints. Try clicking again or toggling a drop.")
        return

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

    # Record pod signatures & persist
    record_pods_in_history(pods)
    save_state()
    

def submit_results(rno, winners):
    rnd = st.session_state.rounds[rno-1]
    players = st.session_state.players
    name_to_id = {p.name.lower(): p.id for p in players.values()}

    for i, pod in enumerate(rnd.pods):
        if i not in winners:
            continue
        w = winners[i]
        if w == 'Tie':
            for pid in pod.players:
                pod.winner = "Tie"
                players[pid].points += 1
            continue
        wid = players[w].id if w in players else name_to_id.get(w.lower())
        if wid is None or wid not in pod.players:
            st.error(f"Winner '{w}' not in Pod {i+1}.")
            return
        if pod.winner:
            continue
        pod.winner = wid
        players[wid].points += WIN_POINTS
    save_state()
    create_backup(rnd)

def standings_rows():
    players = st.session_state.players
    # SOS tiebreaker (sum of opponents' points)
    sos = {pid: 0 for pid in players}
    for pid, p in players.items():
        sos[pid] = sum(players[q].points for q in p.opponents if q in players)

    ordered = sorted(players.values(), key=lambda x: (-x.points, -sos[x.id], x.name))
    rows = []
    for idx, p in enumerate(ordered, start=1):
        status = "Dropped" if p.dropped else ""
        rows.append({"Rank": idx, "Name": p.name, "Points": p.points, "SOS": sos[p.id], "Status": status})
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
    if st.button("Generate Next Round", use_container_width=True, disabled=active_count < MIN_POD):
        pair_round(duration if duration > 0 else None)
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

            # Winner entry (if any pod has no winner and timeout not yet used to end the round)
            if any(p.winner is None for p in rnd.pods):
                st.markdown("#### Enter Winners")
                winners = {}
                for i, pod in enumerate(rnd.pods):
                    if pod.winner:
                        continue
                    options = [st.session_state.players[pid].name for pid in pod.players]
                    options.append("Tie")
                    winners[i] = st.selectbox(f"Winner Pod {i+1}", [""] + options, key=f"w_{rnd.number}_{i}")
                if st.button("Save Results", key=f"save_{rnd.number}"):
                    clean = {i: w for i, w in winners.items() if w}
                    submit_results(rnd.number, clean)
                    st.rerun()

            # # Timeout controls for this round
            # rem = time_remaining(rnd)
            # if rem is not None and rem <= 0 and not rnd.timeout_awarded:
            #     if st.button(f"Time's Up (Round {rnd.number}) → Award 1 point to everyone", key=f"to_{rnd.number}"):
            #         award_timeout_points(rnd.number)
            #         st.rerun()

# --- Standings ---
st.subheader("Standings")
if st.session_state.players:
    st.dataframe(standings_rows(), hide_index=True, use_container_width=True)
else:
    st.info("Add players to begin.")
