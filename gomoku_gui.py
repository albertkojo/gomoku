"""
Gomoku (Five-in-a-Row) — 15x15 board with a Tkinter GUI

Features
--------
• Player vs Player (PvP)
• Player vs Computer (PvC)
• Three AI difficulties (Easy / Medium / Hard)
• Subtle UI touches: star points, last-move marker, winning-line glow

How it works (high level)
-------------------------
• The Board class is the game state: a 15x15 grid with helpers for placing, undoing,
  checking neighbors, and detecting winners.
• The AI evaluates the board with a heuristic that rewards patterns like open fours
  and threes; Hard mode adds a shallow alpha–beta (negamax) search.
• The GUI (GomokuGUI) draws the grid and stones, tracks whose turn it is,
  processes clicks, and queries the AI for moves when needed.

Run
---
python gomoku_gui.py
"""

import math
import random
import tkinter as tk
from tkinter import ttk

# ------------------ Game constants ------------------
BOARD_SIZE = 15                 # Number of intersections per side
WIN_LEN = 5                     # Stones in a row needed to win
EMPTY = "."                    # Empty cell marker (kept simple for readability)
X_STONE = "X"                  # Black stone
O_STONE = "O"                  # White stone

# GUI constants
CANVAS_SIZE = 660               # Pixel size of the drawing canvas (square)
MARGIN = 30                     # Padding around the grid
GRID_SIZE = CANVAS_SIZE - 2 * MARGIN  # Pixel span occupied by the grid
CELL = GRID_SIZE // (BOARD_SIZE - 1)  # Pixel distance between lines / intersections

# AI difficulties (used by a Tkinter Combobox)
DIFFICULTIES = ["Easy", "Medium", "Hard"]

# Directions to scan lines: horizontal, vertical, two diagonals
DIRS = [(1, 0), (0, 1), (1, 1), (1, -1)]

# Heuristic pattern scores — tuned rough magnitudes, not exact theory
PATTERN_SCORES = {
    "FIVE": 10_000_000,
    "OPEN_FOUR": 1_000_000,
    "CLOSED_FOUR": 100_000,
    "OPEN_THREE": 10_000,
    "CLOSED_THREE": 2_000,
    "OPEN_TWO": 500,
    "CLOSED_TWO": 100,
}

# ------------------ Utilities ------------------
def in_bounds(x: int, y: int) -> bool:
    """Return True if (x, y) is a valid board coordinate."""
    return 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE


def stone_name(p: str) -> str:
    """Human-friendly label for a stone symbol."""
    return "Black" if p == X_STONE else "White"


# ------------------ Model ------------------
class Board:
    """Game state container for a Gomoku position.

    Attributes
    ----------
    grid : list[list[str]]
        2D array (row-major: [y][x]) storing EMPTY / X_STONE / O_STONE.
    moves_played : int
        Count of placed stones; useful for draw detection and AI quick checks.
    last_move : tuple[int,int,str] | None
        (x, y, player) for UI highlighting and small optimizations.
    """

    def __init__(self):
        # Initialize an empty board
        self.grid = [[EMPTY for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
        self.moves_played = 0
        self.last_move = None

    def reset(self) -> None:
        """Clear the board and counters to start a new game."""
        for y in range(BOARD_SIZE):
            for x in range(BOARD_SIZE):
                self.grid[y][x] = EMPTY
        self.moves_played = 0
        self.last_move = None

    def place(self, x: int, y: int, player: str) -> bool:
        """Place a stone for `player` at (x, y).

        Returns False if out of bounds or cell is occupied; True otherwise.
        This method *mutates* the board and remembers the last move.
        """
        if not in_bounds(x, y) or self.grid[y][x] != EMPTY:
            return False
        self.grid[y][x] = player
        self.moves_played += 1
        self.last_move = (x, y, player)
        return True

    def unplace(self, x: int, y: int) -> bool:
        """Undo a placement at (x, y) if present. Used by the AI search."""
        if in_bounds(x, y) and self.grid[y][x] != EMPTY:
            self.grid[y][x] = EMPTY
            self.moves_played -= 1
            self.last_move = None
            return True
        return False

    def is_full(self) -> bool:
        """Return True if the board is completely filled (draw)."""
        return self.moves_played >= BOARD_SIZE * BOARD_SIZE

    def winner(self) -> str | None:
        """Fast winner probe for evaluation.

        Scans from every cell in the four directions; returns the stone symbol
        if a 5+ chain is found, otherwise None. This version does not return
        the actual cells (the GUI uses a more detailed method below).
        """
        for y in range(BOARD_SIZE):
            for x in range(BOARD_SIZE):
                p = self.grid[y][x]
                if p == EMPTY:
                    continue
                for dx, dy in DIRS:
                    if self._count_line(x, y, dx, dy, p) >= WIN_LEN:
                        return p
        return None

    def neighbors(self, distance: int = 2) -> set[tuple[int, int]]:
        """Cells within `distance` of any occupied cell.

        Used to limit AI search to plausible moves around current action.
        If the board is empty, we return the center as the only candidate.
        """
        occ = []
        for y in range(BOARD_SIZE):
            for x in range(BOARD_SIZE):
                if self.grid[y][x] != EMPTY:
                    occ.append((x, y))

        if not occ:
            # Favor center as the first move on an empty board
            return {(BOARD_SIZE // 2, BOARD_SIZE // 2)}

        cand: set[tuple[int, int]] = set()
        for (ox, oy) in occ:
            for dy in range(-distance, distance + 1):
                for dx in range(-distance, distance + 1):
                    x, y = ox + dx, oy + dy
                    if in_bounds(x, y) and self.grid[y][x] == EMPTY:
                        cand.add((x, y))
        return cand

    def _count_line(self, x: int, y: int, dx: int, dy: int, player: str) -> int:
        """Count consecutive stones for `player` starting at (x, y) forward in (dx, dy)."""
        c = 0
        cx, cy = x, y
        while in_bounds(cx, cy) and self.grid[cy][cx] == player:
            c += 1
            cx += dx
            cy += dy
        return c

    def has_winner_after(self, x: int, y: int, player: str) -> bool:
        """Check if placing at (x, y) for `player` created a 5-in-a-row.

        This is centered around the last move and expands in both directions
        of each line; it is more efficient than re-scanning the whole board.
        """
        for dx, dy in DIRS:
            cnt = 1
            # forward direction
            cx, cy = x + dx, y + dy
            while in_bounds(cx, cy) and self.grid[cy][cx] == player:
                cnt += 1
                cx += dx
                cy += dy
            # backward direction
            cx, cy = x - dx, y - dy
            while in_bounds(cx, cy) and self.grid[cy][cx] == player:
                cnt += 1
                cx += dx
                cy += dy
            if cnt >= WIN_LEN:
                return True
        return False

    def winning_line(self) -> tuple[str | None, list[tuple[int, int]] | None]:
        """Return (winner_symbol, list_of_five_cells) for UI highlighting.

        We only return the first exact 5 cells in a found chain to draw a neat
        golden line over the winning stones.
        """
        for y in range(BOARD_SIZE):
            for x in range(BOARD_SIZE):
                p = self.grid[y][x]
                if p == EMPTY:
                    continue
                for dx, dy in DIRS:
                    # Ensure (x, y) is the start of a chain: previous cell must not be `p`.
                    px, py = x - dx, y - dy
                    if in_bounds(px, py) and self.grid[py][px] == p:
                        continue
                    # Collect a chain forward
                    cells: list[tuple[int, int]] = []
                    cx, cy = x, y
                    while in_bounds(cx, cy) and self.grid[cy][cx] == p:
                        cells.append((cx, cy))
                        cx += dx
                        cy += dy
                    if len(cells) >= WIN_LEN:
                        return p, cells[:WIN_LEN]
        return None, None


# ------------------ Heuristic evaluation ------------------
def evaluate_board(board: 'Board', me: str, opp: str) -> int:
    """Score the position from the perspective of `me`.

    • Immediate wins/losses are short-circuited with very large scores.
    • Otherwise, we scan every 5-cell window (in all lines) and sum pattern
      scores for both sides, returning the difference (my score − their score).
    """
    w = board.winner()
    if w == me:
        return PATTERN_SCORES["FIVE"]
    if w == opp:
        return -PATTERN_SCORES["FIVE"]

    total_me = 0
    total_opp = 0

    # Iterate through every maximal line (start-of-line detection avoids duplicates)
    for y in range(BOARD_SIZE):
        for x in range(BOARD_SIZE):
            for dx, dy in DIRS:
                # Only start at true line starts — the previous cell must be out of bounds
                prevx, prevy = x - dx, y - dy
                if in_bounds(prevx, prevy):
                    continue

                # Compute how many cells this line has from (x, y) forward
                length = 0
                cx, cy = x, y
                while in_bounds(cx, cy):
                    length += 1
                    cx += dx
                    cy += dy

                # Slide a 5-cell window along this line and evaluate
                for start in range(length - WIN_LEN + 1):
                    cells = []
                    sx, sy = x + dx * start, y + dy * start
                    valid = True
                    for k in range(WIN_LEN):
                        cx, cy = sx + dx * k, sy + dy * k
                        if not in_bounds(cx, cy):
                            valid = False
                            break
                        cells.append(board.grid[cy][cx])
                    if not valid:
                        continue
                    total_me += score_window(cells, me, opp)
                    total_opp += score_window(cells, opp, me)

    return total_me - total_opp


def score_window(cells: list[str], me: str, opp: str) -> int:
    """Score a specific 5-cell window for `me`.

    • If both players occupy the window, it has no immediate pattern value.
    • Otherwise count how many of my stones (or opponent's) and empties it has
      and return a heuristic score. These magnitudes drive search priorities.

    Note: Openness (both ends empty) is *approximated* here; for simplicity we
    look only at counts within the window, which works reasonably for shallow depths.
    """
    s = set(cells)
    if me in s and opp in s:
        return 0

    me_cnt = cells.count(me)
    opp_cnt = cells.count(opp)
    empty_cnt = cells.count(EMPTY)

    # Immediate success — five in a row
    if me_cnt == 5:
        return PATTERN_SCORES["FIVE"]
    if opp_cnt == 5:
        # Not scoring opponent here (caller handles opp windows separately)
        return 0

    # Lightweight pattern signals (open vs closed is approximated)
    if me_cnt == 4 and empty_cnt == 1:
        return PATTERN_SCORES["OPEN_FOUR"]
    if me_cnt == 3 and empty_cnt == 2:
        return PATTERN_SCORES["OPEN_THREE"]
    if me_cnt == 2 and empty_cnt == 3:
        return PATTERN_SCORES["OPEN_TWO"]

    # Fallbacks — kept for flexibility if you want to expand the model later
    if me_cnt == 4 and empty_cnt == 1:
        return PATTERN_SCORES["CLOSED_FOUR"]
    if me_cnt == 3 and empty_cnt == 2:
        return PATTERN_SCORES["CLOSED_THREE"]
    if me_cnt == 2 and empty_cnt == 3:
        return PATTERN_SCORES["CLOSED_TWO"]
    return 0


# ------------------ AI ------------------
def ai_move(board: 'Board', me: str = O_STONE, opp: str = X_STONE, difficulty: str = "Medium") -> tuple[int, int]:
    """Pick a move for `me` at the chosen `difficulty`.

    Easy   : random among nearby candidates (neighbors).
    Medium : one-ply evaluation with immediate-win and simple block detection.
    Hard   : shallow negamax (alpha–beta) with move ordering.
    """
    # First move: prefer center (classic opening)
    if board.moves_played == 0 and board.grid[BOARD_SIZE // 2][BOARD_SIZE // 2] == EMPTY:
        return (BOARD_SIZE // 2, BOARD_SIZE // 2)

    # Candidate moves near existing stones keep the branching factor small
    candidates = list(board.neighbors(distance=2))
    if not candidates:
        # Shouldn't happen often, but fall back to any empty cell
        empties = [(x, y) for y in range(BOARD_SIZE) for x in range(BOARD_SIZE) if board.grid[y][x] == EMPTY]
        return random.choice(empties)

    # 1) Easy: purely random among local candidates
    if difficulty == "Easy":
        return random.choice(candidates)

    # 2) Medium: greedy one-ply eval + instant win and simple threat block
    if difficulty == "Medium":
        best_score = -math.inf
        best_moves: list[tuple[int, int]] = []
        for (x, y) in candidates:
            board.place(x, y, me)

            # Immediate win?
            if board.has_winner_after(x, y, me):
                board.unplace(x, y)
                return (x, y)

            # Look for opponent immediate win next to current positions and add a bonus if we block it
            opp_threat = False
            for (ox, oy) in board.neighbors(distance=1):
                if board.grid[oy][ox] == EMPTY:
                    board.place(ox, oy, opp)
                    if board.has_winner_after(ox, oy, opp):
                        opp_threat = True
                    board.unplace(ox, oy)
                    if opp_threat:
                        break

            score = evaluate_board(board, me, opp)
            if opp_threat:
                score += 50_000  # Favor defensive blocks

            board.unplace(x, y)

            # Track best-scoring moves (tie-break randomly later)
            if score > best_score:
                best_score = score
                best_moves = [(x, y)]
            elif score == best_score:
                best_moves.append((x, y))

        return random.choice(best_moves)

    # 3) Hard: Negamax with alpha–beta pruning (depth-limited) and ordering
    depth = 3  # shallow but effective with good ordering on a 15x15 board

    # Move ordering by static evaluation tends to help alpha–beta a lot
    ordered: list[tuple[tuple[int, int], int]] = []
    for (x, y) in candidates:
        board.place(x, y, me)
        s = evaluate_board(board, me, opp)
        board.unplace(x, y)
        ordered.append(((x, y), s))
    ordered.sort(key=lambda t: t[1], reverse=True)

    best, best_move = -math.inf, None
    alpha, beta = -math.inf, math.inf

    # Limit breadth a bit (top N candidates) for speed
    for (x, y), _ in ordered[:24]:
        board.place(x, y, me)
        if board.has_winner_after(x, y, me):  # immediate killer move
            board.unplace(x, y)
            return (x, y)
        val = -negamax(board, depth - 1, -beta, -alpha, opp, me)
        board.unplace(x, y)
        if val > best:
            best, best_move = val, (x, y)
        alpha = max(alpha, val)
        if alpha >= beta:  # alpha–beta cutoff
            break

    if best_move is None:
        return random.choice(candidates)
    return best_move


def negamax(board: 'Board', depth: int, alpha: int, beta: int, me: str, opp: str) -> int:
    """Negamax search with alpha–beta pruning.

    Returns a score from the perspective of the current player `me`.
    """
    # Terminal checks
    w = board.winner()
    if w == me:
        return PATTERN_SCORES["FIVE"]
    elif w == opp:
        return -PATTERN_SCORES["FIVE"]
    if depth == 0 or board.is_full():
        return evaluate_board(board, me, opp)

    best = -math.inf

    # Restrict to local moves; if none, fall back to all empties
    candidates = list(board.neighbors(distance=2))
    if not candidates:
        candidates = [(x, y) for y in range(BOARD_SIZE) for x in range(BOARD_SIZE) if board.grid[y][x] == EMPTY]

    # Order by static evaluation to induce earlier cutoffs
    ordered: list[tuple[tuple[int, int], int]] = []
    for (x, y) in candidates:
        board.place(x, y, me)
        s = evaluate_board(board, me, opp)
        board.unplace(x, y)
        ordered.append(((x, y), s))
    ordered.sort(key=lambda t: t[1], reverse=True)

    # Explore a capped number of moves for speed
    for (x, y), _ in ordered[:20]:
        board.place(x, y, me)
        if board.has_winner_after(x, y, me):
            board.unplace(x, y)
            return PATTERN_SCORES["FIVE"]
        val = -negamax(board, depth - 1, -beta, -alpha, opp, me)
        board.unplace(x, y)
        if val > best:
            best = val
        if best > alpha:
            alpha = best
        if alpha >= beta:  # prune
            break
    return best


# ------------------ GUI ------------------
class GomokuGUI:
    """Tkinter front-end for drawing the board and coordinating turns."""

    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("Gomoku — Five-in-a-Row")

        # Model: a fresh board
        self.board = Board()

        # Game mode & settings (Tk variables are bindable to widgets)
        self.mode = tk.StringVar(value="PvC")            # "PvC" or "PvP"
        self.difficulty = tk.StringVar(value="Medium")
        self.first_player = tk.StringVar(value="Human")  # In PvC: who starts

        # Turn tracking and outcome state
        self.current = X_STONE
        self.game_over = False
        self.win_positions: list[tuple[int, int]] | None = None
        self.winner_player: str | None = None

        # Layout: canvas + sidebar
        self.canvas = tk.Canvas(
            root,
            width=CANVAS_SIZE,
            height=CANVAS_SIZE,
            bg="#f5deb3",            # wood-ish background color
            highlightthickness=0,
        )
        self.canvas.grid(row=0, column=0, padx=(10, 0), pady=10, sticky="nsew")

        sidebar = ttk.Frame(root)
        sidebar.grid(row=0, column=1, padx=10, pady=10, sticky="ns")

        # --- Controls ---
        ttk.Label(sidebar, text="Mode").pack(anchor="w", pady=(0, 2))
        ttk.Radiobutton(sidebar, text="Player vs Computer", variable=self.mode, value="PvC", command=self.on_mode_change).pack(anchor="w")
        ttk.Radiobutton(sidebar, text="Player vs Player", variable=self.mode, value="PvP", command=self.on_mode_change).pack(anchor="w")

        # PvC-specific controls (difficulty, who starts)
        self.pvc_frame = ttk.Frame(sidebar)
        self.pvc_frame.pack(anchor="w", pady=(10, 0), fill="x")
        ttk.Label(self.pvc_frame, text="Difficulty").pack(anchor="w")
        ttk.Combobox(self.pvc_frame, textvariable=self.difficulty, values=DIFFICULTIES, state="readonly").pack(anchor="w", fill="x")

        ttk.Label(self.pvc_frame, text="First Player").pack(anchor="w", pady=(8, 0))
        ttk.Radiobutton(self.pvc_frame, text="Human", variable=self.first_player, value="Human").pack(anchor="w")
        ttk.Radiobutton(self.pvc_frame, text="Computer", variable=self.first_player, value="Computer").pack(anchor="w")

        ttk.Separator(sidebar, orient="horizontal").pack(fill="x", pady=10)

        self.new_btn = ttk.Button(sidebar, text="New Game", command=self.new_game)
        self.new_btn.pack(fill="x")

        ttk.Separator(sidebar, orient="horizontal").pack(fill="x", pady=10)

        self.status = tk.StringVar(value="Black to move")
        ttk.Label(sidebar, textvariable=self.status, wraplength=180).pack(anchor="w", pady=(0, 10))

        # Mouse clicks on the board
        self.canvas.bind("<Button-1>", self.on_click)

        # Initial paint + status
        self.draw_board()
        self.update_status()

        # If computer starts in PvC, trigger an opening move shortly after startup
        self.root.after(200, self.maybe_computer_first)

        # Make UI resize nicely (canvas grows)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)

    # ---- Drawing helpers ----
    def draw_board(self) -> None:
        """Redraw the entire board: grid, star points, stones, markers, and winning line."""
        self.canvas.delete("all")

        # Grid lines
        for i in range(BOARD_SIZE):
            x0 = MARGIN + i * CELL
            self.canvas.create_line(MARGIN, MARGIN + i * CELL, MARGIN + (BOARD_SIZE - 1) * CELL, MARGIN + i * CELL)
            self.canvas.create_line(x0, MARGIN, x0, MARGIN + (BOARD_SIZE - 1) * CELL)

        # Star points like on a Go board (purely aesthetic guidance)
        for sx in [3, 7, 11]:
            for sy in [3, 7, 11]:
                cx, cy = self.to_canvas(sx, sy)
                self.canvas.create_oval(cx - 3, cy - 3, cx + 3, cy + 3, fill="#333", outline="")

        # Stones (draw each occupied cell as a filled circle)
        r = CELL * 0.40  # stone radius
        for y in range(BOARD_SIZE):
            for x in range(BOARD_SIZE):
                p = self.board.grid[y][x]
                if p == EMPTY:
                    continue
                cx, cy = self.to_canvas(x, y)
                fill = "#111" if p == X_STONE else "#eee"  # black / white stone
                outline = "#000"

                # Highlight winning stones with a golden glow ring
                if self.win_positions and (x, y) in self.win_positions:
                    glow_r = r + 6
                    self.canvas.create_oval(
                        cx - glow_r, cy - glow_r, cx + glow_r, cy + glow_r, outline="#ffbf00", width=3
                    )

                self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r, fill=fill, outline=outline, width=2)

        # Mark the last move with a small red square (if game not finished)
        if self.board.last_move and not self.win_positions:
            lx, ly, _ = self.board.last_move
            cx, cy = self.to_canvas(lx, ly)
            self.canvas.create_rectangle(cx - 6, cy - 6, cx + 6, cy + 6, outline="#d00", width=2)

        # If someone won, draw a thick golden line through the winning five
        if self.win_positions:
            (sx, sy) = self.win_positions[0]
            (ex, ey) = self.win_positions[-1]
            x1, y1 = self.to_canvas(sx, sy)
            x2, y2 = self.to_canvas(ex, ey)
            self.canvas.create_line(x1, y1, x2, y2, fill="#ffbf00", width=4)

    def to_canvas(self, x: int, y: int) -> tuple[int, int]:
        """Convert board coordinates (x, y) to canvas pixels."""
        return (MARGIN + x * CELL, MARGIN + y * CELL)

    def from_canvas(self, px: int, py: int) -> tuple[int, int] | None:
        """Snap an (px, py) click to the nearest board intersection, if close enough."""
        x = round((px - MARGIN) / CELL)
        y = round((py - MARGIN) / CELL)
        if in_bounds(x, y):
            # Ensure the click is reasonably close to an intersection to avoid misclicks
            gx, gy = self.to_canvas(x, y)
            if abs(px - gx) <= CELL * 0.45 and abs(py - gy) <= CELL * 0.45:
                return x, y
        return None

    # ---- Game flow ----
    def on_mode_change(self) -> None:
        """Show/Hide PvC controls when toggling modes, then start a new game."""
        if self.mode.get() == "PvC":
            self.pvc_frame.pack(anchor="w", pady=(10, 0), fill="x")
        else:
            self.pvc_frame.forget()
        self.new_game()

    def new_game(self) -> None:
        """Reset the board and UI state for a fresh game."""
        self.board.reset()
        self.game_over = False
        self.current = X_STONE
        self.win_positions = None
        self.winner_player = None
        self.draw_board()
        self.update_status()
        # If computer is set to start, give it a tiny delay to feel responsive
        self.root.after(200, self.maybe_computer_first)

    def update_status(self, msg: str | None = None) -> None:
        """Update the status label with game messages and turn info."""
        if msg:
            self.status.set(msg)
            return
        if self.game_over:
            if self.winner_player:
                self.status.set(f"Winner: {stone_name(self.winner_player)} 🎉  (New Game to play again)")
            else:
                self.status.set("Draw. (New Game to play again)")
        else:
            turn_name = stone_name(self.current)
            if self.mode.get() == "PvP":
                self.status.set(f"{turn_name} to move")
            else:
                who = "Human" if self.human_turn() else "Computer"
                self.status.set(f"{turn_name} to move — {who}")

    def human_turn(self) -> bool:
        """In PvC, return True if it's the human's turn; always True in PvP."""
        if self.mode.get() == "PvP":
            return True
        human_symbol = X_STONE if self.first_player.get() == "Human" else O_STONE
        return self.current == human_symbol

    def maybe_computer_first(self) -> None:
        """If Computer is set to start in PvC, let it make the opening move."""
        if (
            self.mode.get() == "PvC"
            and self.first_player.get() == "Computer"
            and not self.game_over
            and self.current == X_STONE
            and self.board.moves_played == 0
        ):
            self.root.after(150, self.do_computer_move)

    def on_click(self, event: tk.Event) -> None:
        """Handle a board click: place a stone if it's a valid human move."""
        if self.game_over:
            return
        xy = self.from_canvas(event.x, event.y)
        if not xy:
            return
        x, y = xy
        # In PvC, ignore clicks when it's not the human's turn
        if self.mode.get() == "PvC" and not self.human_turn():
            return
        if not self.board.place(x, y, self.current):
            return
        self.post_move_update()

        # If game continues and it's now the computer's turn, ask AI to move
        if not self.game_over and self.mode.get() == "PvC" and not self.human_turn():
            self.root.after(150, self.do_computer_move)

    def do_computer_move(self) -> None:
        """Ask the AI for a move and place it if the game is still live."""
        if self.game_over:
            return
        me = self.current
        opp = X_STONE if me == O_STONE else O_STONE
        move = ai_move(self.board, me=me, opp=opp, difficulty=self.difficulty.get())
        if move:
            x, y = move
            self.board.place(x, y, me)
            self.post_move_update()

    def post_move_update(self) -> None:
        """After any move: check win/draw, flip turns, redraw, and refresh status."""
        # Winner check with explicit winning cells for pretty UI
        winner, positions = self.board.winning_line()
        if winner:
            self.game_over = True
            self.winner_player = winner
            self.win_positions = positions
            self.draw_board()
            self.update_status()
            return

        # Draw?
        if self.board.is_full():
            self.game_over = True
            self.winner_player = None
            self.win_positions = None
            self.draw_board()
            self.update_status()
            return

        # Otherwise, swap the current player and continue
        self.current = O_STONE if self.current == X_STONE else X_STONE
        self.draw_board()
        self.update_status()


# ------------------ Main ------------------
def main() -> None:
    """Create the Tk app window and launch the GUI event loop."""
    root = tk.Tk()
    # Try a nicer ttk theme if available (optional aesthetics)
    try:
        if "clam" in ttk.Style().theme_names():
            ttk.Style().theme_use("clam")
    except Exception:
        pass

    app = GomokuGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
