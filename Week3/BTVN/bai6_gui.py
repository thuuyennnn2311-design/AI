"""
Bài 6: Giao diện đồ họa (GUI) cho TicTacToe NxN
Dùng thư viện tkinter (có sẵn trong Python)
Tích hợp AI Minimax & Alpha-Beta Pruning từ Bài 3, 4

Chạy: python bai6_gui.py
"""

import tkinter as tk
from tkinter import ttk, messagebox
import math
import time
import threading

# ─────────────────────────────────────────
#  THUẬT TOÁN AI (gộp từ Bài 3 & 4)
# ─────────────────────────────────────────
DIRECTIONS = [(0, 1), (1, 0), (1, 1), (1, -1)]

def check_win(board, n, r, c, player, k):
    for dr, dc in DIRECTIONS:
        count = 1
        for d in range(1, k):
            nr, nc = r + d*dr, c + d*dc
            if 0 <= nr < n and 0 <= nc < n and board[nr][nc] == player:
                count += 1
            else:
                break
        for d in range(1, k):
            nr, nc = r - d*dr, c - d*dc
            if 0 <= nr < n and 0 <= nc < n and board[nr][nc] == player:
                count += 1
            else:
                break
        if count >= k:
            return True
    return False

def get_win_cells(board, n, r, c, player, k):
    for dr, dc in DIRECTIONS:
        cells = [(r, c)]
        for d in range(1, k):
            nr, nc = r + d*dr, c + d*dc
            if 0 <= nr < n and 0 <= nc < n and board[nr][nc] == player:
                cells.append((nr, nc))
            else:
                break
        for d in range(1, k):
            nr, nc = r - d*dr, c - d*dc
            if 0 <= nr < n and 0 <= nc < n and board[nr][nc] == player:
                cells.append((nr, nc))
            else:
                break
        if len(cells) >= k:
            return cells
    return []

def get_candidate_moves(board, n, radius=2):
    candidates = set()
    has_piece = False
    for r in range(n):
        for c in range(n):
            if board[r][c] != '':
                has_piece = True
                for dr in range(-radius, radius+1):
                    for dc in range(-radius, radius+1):
                        nr, nc = r+dr, c+dc
                        if 0 <= nr < n and 0 <= nc < n and board[nr][nc] == '':
                            candidates.add((nr, nc))
    if not has_piece:
        mid = n // 2
        return [(mid, mid)]
    return list(candidates) if candidates else []

def evaluate(board, n, k, ai, human):
    score = 0
    def window_score(cells):
        ac = sum(1 for p in cells if p == ai)
        hc = sum(1 for p in cells if p == human)
        if ac and hc:
            return 0
        if ac == k: return 100000
        if hc == k: return -100000
        if ac: return 10**ac
        if hc: return -(10**hc)*1.1
        return 0
    for r in range(n):
        for c in range(n):
            for dr, dc in DIRECTIONS:
                end_r, end_c = r+(k-1)*dr, c+(k-1)*dc
                if 0 <= end_r < n and 0 <= end_c < n:
                    score += window_score([board[r+i*dr][c+i*dc] for i in range(k)])
    return score

def move_order_score(board, n, r, c, k, ai, human):
    s = 0
    for dr, dc in DIRECTIONS:
        for pl, m in [(ai, 1), (human, 1.1)]:
            cnt = 0
            for d in range(-(k-1), k):
                nr, nc = r+d*dr, c+d*dc
                if 0 <= nr < n and 0 <= nc < n and board[nr][nc] == pl:
                    cnt += 1
            s += (10**cnt)*m
    return s

def alphabeta(board, n, depth, is_max, alpha, beta, ai, human, k,
              last_r=-1, last_c=-1, counter=[0]):
    counter[0] += 1
    opp = human if is_max else ai
    if last_r >= 0 and check_win(board, n, last_r, last_c, opp, k):
        return (-1000 - depth) if is_max else (1000 + depth)
    if depth == 0:
        return evaluate(board, n, k, ai, human)
    moves = get_candidate_moves(board, n)
    if not moves:
        return evaluate(board, n, k, ai, human)
    cur = ai if is_max else human
    moves.sort(key=lambda m: move_order_score(board, n, m[0], m[1], k, ai, human),
               reverse=is_max)
    if is_max:
        best = -math.inf
        for r, c in moves:
            board[r][c] = cur
            val = alphabeta(board, n, depth-1, False, alpha, beta, ai, human, k, r, c, counter)
            board[r][c] = ''
            best = max(best, val)
            alpha = max(alpha, best)
            if beta <= alpha: break
        return best
    else:
        best = math.inf
        for r, c in moves:
            board[r][c] = cur
            val = alphabeta(board, n, depth-1, True, alpha, beta, ai, human, k, r, c, counter)
            board[r][c] = ''
            best = min(best, val)
            beta = min(beta, best)
            if beta <= alpha: break
        return best

def minimax(board, n, depth, is_max, ai, human, k,
            last_r=-1, last_c=-1, counter=[0]):
    counter[0] += 1
    opp = human if is_max else ai
    if last_r >= 0 and check_win(board, n, last_r, last_c, opp, k):
        return (-1000 - depth) if is_max else (1000 + depth)
    if depth == 0:
        return evaluate(board, n, k, ai, human)
    moves = get_candidate_moves(board, n)
    if not moves:
        return evaluate(board, n, k, ai, human)
    cur = ai if is_max else human
    if is_max:
        best = -math.inf
        for r, c in moves:
            board[r][c] = cur
            best = max(best, minimax(board, n, depth-1, False, ai, human, k, r, c, counter))
            board[r][c] = ''
        return best
    else:
        best = math.inf
        for r, c in moves:
            board[r][c] = cur
            best = min(best, minimax(board, n, depth-1, True, ai, human, k, r, c, counter))
            board[r][c] = ''
        return best

def find_best_move(board, n, depth, ai, human, k, use_ab=True):
    counter = [0]
    best_score = -math.inf
    best_move = None
    moves = get_candidate_moves(board, n)
    moves.sort(key=lambda m: move_order_score(board, n, m[0], m[1], k, ai, human),
               reverse=True)
    t0 = time.time()
    for r, c in moves:
        board[r][c] = ai
        if use_ab:
            s = alphabeta(board, n, depth-1, False, -math.inf, math.inf,
                          ai, human, k, r, c, counter)
        else:
            s = minimax(board, n, depth-1, False, ai, human, k, r, c, counter)
        board[r][c] = ''
        if s > best_score:
            best_score = s
            best_move = (r, c)
    elapsed = time.time() - t0
    return best_move, counter[0], elapsed


# ─────────────────────────────────────────
#  GIAO DIỆN ĐỒ HỌA (tkinter)
# ─────────────────────────────────────────
class TicTacToeGUI:
    # Màu sắc
    BG          = "#1e1e2e"
    PANEL_BG    = "#2a2a3e"
    CELL_BG     = "#2d2d42"
    CELL_HOVER  = "#3a3a55"
    CELL_WIN    = "#264d3b"
    X_COLOR     = "#7cb4f5"
    O_COLOR     = "#f5a97f"
    TEXT_MAIN   = "#cdd6f4"
    TEXT_MUTED  = "#6c7086"
    ACCENT      = "#89b4fa"
    SUCCESS     = "#a6e3a1"
    WARNING     = "#f9e2af"
    BTN_BG      = "#363650"
    BTN_HOVER   = "#45456a"

    def __init__(self, root):
        self.root = root
        self.root.title("TicTacToe — Minimax & Alpha-Beta")
        self.root.configure(bg=self.BG)
        self.root.resizable(True, True)

        # Trạng thái game
        self.N = 5
        self.K = 4
        self.depth = 3
        self.ai_side = 'O'
        self.use_ab = True
        self.board = []
        self.current = 'X'
        self.game_over = False
        self.thinking = False
        self.scores = {'X': 0, 'O': 0, 'Draw': 0}
        self.buttons = []

        self._build_ui()
        self.new_game()

    # ── Xây dựng UI ──────────────────────
    def _build_ui(self):
        # Tiêu đề
        hdr = tk.Frame(self.root, bg=self.BG)
        hdr.pack(fill='x', padx=20, pady=(16, 4))
        tk.Label(hdr, text="TicTacToe AI", font=("Helvetica", 22, "bold"),
                 bg=self.BG, fg=self.ACCENT).pack(side='left')
        tk.Label(hdr, text="Minimax & Alpha-Beta Pruning",
                 font=("Helvetica", 12), bg=self.BG, fg=self.TEXT_MUTED).pack(
                     side='left', padx=(12, 0), pady=(6, 0))

        # Thanh cài đặt
        ctrl = tk.Frame(self.root, bg=self.PANEL_BG, pady=10)
        ctrl.pack(fill='x', padx=20, pady=6)

        def lbl(parent, text):
            return tk.Label(parent, text=text, bg=self.PANEL_BG,
                            fg=self.TEXT_MUTED, font=("Helvetica", 11))

        # Kích thước N
        f1 = tk.Frame(ctrl, bg=self.PANEL_BG)
        f1.pack(side='left', padx=14)
        lbl(f1, "Kích thước N").pack()
        self.size_var = tk.StringVar(value="5")
        cb_size = ttk.Combobox(f1, textvariable=self.size_var,
                               values=["3","4","5","7","10","12","15"],
                               width=5, state='readonly')
        cb_size.pack()

        # Thắng khi K
        f2 = tk.Frame(ctrl, bg=self.PANEL_BG)
        f2.pack(side='left', padx=14)
        lbl(f2, "Thắng K quân").pack()
        self.k_var = tk.StringVar(value="4")
        cb_k = ttk.Combobox(f2, textvariable=self.k_var,
                             values=["3","4","5","6"],
                             width=5, state='readonly')
        cb_k.pack()

        # Thuật toán
        f3 = tk.Frame(ctrl, bg=self.PANEL_BG)
        f3.pack(side='left', padx=14)
        lbl(f3, "Thuật toán").pack()
        self.algo_var = tk.StringVar(value="Alpha-Beta")
        cb_algo = ttk.Combobox(f3, textvariable=self.algo_var,
                               values=["Alpha-Beta","Minimax"],
                               width=12, state='readonly')
        cb_algo.pack()

        # AI đi
        f4 = tk.Frame(ctrl, bg=self.PANEL_BG)
        f4.pack(side='left', padx=14)
        lbl(f4, "AI đi").pack()
        self.ai_var = tk.StringVar(value="O")
        cb_ai = ttk.Combobox(f4, textvariable=self.ai_var,
                              values=["X","O","Người vs Người"],
                              width=13, state='readonly')
        cb_ai.pack()

        # Độ sâu
        f5 = tk.Frame(ctrl, bg=self.PANEL_BG)
        f5.pack(side='left', padx=14)
        lbl(f5, "Độ sâu AI").pack()
        self.depth_var = tk.StringVar(value="3")
        cb_d = ttk.Combobox(f5, textvariable=self.depth_var,
                             values=["1","2","3","4","5"],
                             width=5, state='readonly')
        cb_d.pack()

        # Nút
        btn_frame = tk.Frame(ctrl, bg=self.PANEL_BG)
        btn_frame.pack(side='right', padx=14)
        self._btn(btn_frame, "▶ Ván mới", self.apply_and_new).pack(pady=2)
        self._btn(btn_frame, "↺ Reset điểm", self.reset_scores).pack(pady=2)

        # Thanh trạng thái
        self.status_var = tk.StringVar(value="Chào mừng! Nhấn 'Ván mới' để bắt đầu.")
        self.status_lbl = tk.Label(self.root, textvariable=self.status_var,
                                   font=("Helvetica", 13, "bold"),
                                   bg=self.BG, fg=self.SUCCESS, pady=6)
        self.status_lbl.pack()

        # Thống kê AI
        self.ai_info_var = tk.StringVar(value="")
        tk.Label(self.root, textvariable=self.ai_info_var,
                 font=("Helvetica", 10), bg=self.BG, fg=self.TEXT_MUTED).pack()

        # Bảng điểm
        score_frame = tk.Frame(self.root, bg=self.BG)
        score_frame.pack(pady=6)
        self.score_labels = {}
        for player, color in [('X', self.X_COLOR), ('Draw', self.TEXT_MUTED), ('O', self.O_COLOR)]:
            f = tk.Frame(score_frame, bg=self.PANEL_BG, padx=18, pady=8)
            f.pack(side='left', padx=8)
            lname = "X" if player == 'X' else "Hoà" if player == 'Draw' else "O"
            tk.Label(f, text=f"{'X' if player=='X' else '○ O' if player=='O' else '─'}",
                     font=("Helvetica", 14, "bold"), bg=self.PANEL_BG, fg=color).pack()
            lbl_s = tk.Label(f, text="0", font=("Helvetica", 20, "bold"),
                             bg=self.PANEL_BG, fg=color)
            lbl_s.pack()
            tk.Label(f, text=lname, font=("Helvetica", 9),
                     bg=self.PANEL_BG, fg=self.TEXT_MUTED).pack()
            self.score_labels[player] = lbl_s

        # Bàn cờ
        self.board_frame = tk.Frame(self.root, bg=self.BG)
        self.board_frame.pack(pady=10)

    def _btn(self, parent, text, cmd):
        b = tk.Button(parent, text=text, command=cmd,
                      bg=self.BTN_BG, fg=self.TEXT_MAIN,
                      font=("Helvetica", 11), relief='flat',
                      activebackground=self.BTN_HOVER, activeforeground=self.TEXT_MAIN,
                      padx=12, pady=4, cursor='hand2')
        b.bind('<Enter>', lambda e: b.configure(bg=self.BTN_HOVER))
        b.bind('<Leave>', lambda e: b.configure(bg=self.BTN_BG))
        return b

    # ── Điều khiển game ──────────────────
    def apply_and_new(self):
        self.N = int(self.size_var.get())
        self.K = min(int(self.k_var.get()), self.N)
        self.depth = int(self.depth_var.get())
        self.use_ab = self.algo_var.get() == "Alpha-Beta"
        ai_v = self.ai_var.get()
        self.ai_side = None if ai_v == "Người vs Người" else ai_v
        self.new_game()

    def new_game(self):
        self.board = [['' for _ in range(self.N)] for _ in range(self.N)]
        self.current = 'X'
        self.game_over = False
        self.thinking = False
        self._build_board()
        self._update_status()
        if self.ai_side == 'X':
            self.root.after(200, self._ai_move)

    def reset_scores(self):
        self.scores = {'X': 0, 'O': 0, 'Draw': 0}
        self._update_scores()

    # ── Bàn cờ ───────────────────────────
    def _build_board(self):
        for w in self.board_frame.winfo_children():
            w.destroy()
        self.buttons = []
        cell_size = max(34, min(72, 520 // self.N))
        font_size = max(12, int(cell_size * 0.45))

        for r in range(self.N):
            row_btns = []
            for c in range(self.N):
                btn = tk.Button(
                    self.board_frame,
                    text='',
                    width=2, height=1,
                    font=("Helvetica", font_size, "bold"),
                    bg=self.CELL_BG,
                    fg=self.TEXT_MAIN,
                    relief='flat',
                    activebackground=self.CELL_HOVER,
                    cursor='hand2',
                    command=lambda rr=r, cc=c: self._human_click(rr, cc)
                )
                btn.grid(row=r, column=c, padx=1, pady=1,
                         ipadx=cell_size//6, ipady=cell_size//8)
                btn.bind('<Enter>', lambda e, b=btn: self._on_hover(b, True))
                btn.bind('<Leave>', lambda e, b=btn: self._on_hover(b, False))
                row_btns.append(btn)
            self.buttons.append(row_btns)

    def _on_hover(self, btn, enter):
        if btn['text'] == '' and not self.game_over and not self.thinking:
            btn.configure(bg=self.CELL_HOVER if enter else self.CELL_BG)

    def _render_cell(self, r, c):
        val = self.board[r][c]
        btn = self.buttons[r][c]
        if val == 'X':
            btn.configure(text='X', fg=self.X_COLOR)
        elif val == 'O':
            btn.configure(text='O', fg=self.O_COLOR)
        else:
            btn.configure(text='', fg=self.TEXT_MAIN, bg=self.CELL_BG)

    def _highlight_win(self, cells):
        for r, c in cells:
            self.buttons[r][c].configure(bg=self.CELL_WIN)

    # ── Lượt chơi ────────────────────────
    def _human_click(self, r, c):
        if self.game_over or self.thinking or self.board[r][c]:
            return
        if self.ai_side and self.current == self.ai_side:
            return
        self._place(r, c)

    def _place(self, r, c):
        self.board[r][c] = self.current
        self._render_cell(r, c)

        if check_win(self.board, self.N, r, c, self.current, self.K):
            wc = get_win_cells(self.board, self.N, r, c, self.current, self.K)
            self._highlight_win(wc)
            self.game_over = True
            self.scores[self.current] += 1
            self._update_scores()
            msg = f"{'🎉 Bạn thắng!' if self.current != self.ai_side else '🤖 AI thắng!'}"
            if self.ai_side is None:
                msg = f"🏆 {self.current} thắng!"
            self.status_var.set(msg)
            self.status_lbl.configure(fg=self.SUCCESS)
            return

        if all(self.board[r2][c2] for r2 in range(self.N) for c2 in range(self.N)):
            self.game_over = True
            self.scores['Draw'] += 1
            self._update_scores()
            self.status_var.set("🤝 Hoà!")
            self.status_lbl.configure(fg=self.WARNING)
            return

        self.current = 'O' if self.current == 'X' else 'X'
        self._update_status()

        if self.ai_side and self.current == self.ai_side:
            self.root.after(80, self._ai_move)

    def _ai_move(self):
        if self.game_over:
            return
        self.thinking = True
        self.status_var.set(f"🤔 AI ({self.current}) đang suy nghĩ...")
        self.status_lbl.configure(fg=self.WARNING)
        self.root.update()

        def run():
            b_copy = [row[:] for row in self.board]
            move, nodes, elapsed = find_best_move(
                b_copy, self.N, self.depth,
                self.current,
                'O' if self.current == 'X' else 'X',
                self.K, self.use_ab
            )
            algo = "Alpha-Beta" if self.use_ab else "Minimax"
            self.ai_info_var.set(
                f"[{algo}] Nút duyệt: {nodes:,}  |  Thời gian: {elapsed:.3f}s"
            )
            self.thinking = False
            if move:
                self.root.after(0, lambda: self._place(*move))

        t = threading.Thread(target=run, daemon=True)
        t.start()

    # ── Helpers ──────────────────────────
    def _update_status(self):
        side_name = f"{self.current}"
        if not self.game_over:
            if self.ai_side and self.current == self.ai_side:
                msg = f"Lượt AI ({self.current})..."
            else:
                msg = f"Lượt {self.current} — Click ô để đi"
            self.status_var.set(msg)
            self.status_lbl.configure(fg=self.TEXT_MAIN)

    def _update_scores(self):
        for k, lbl in self.score_labels.items():
            lbl.configure(text=str(self.scores[k]))


# ─────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────
if __name__ == '__main__':
    root = tk.Tk()

    # Style ttk
    style = ttk.Style()
    style.theme_use('clam')
    style.configure('TCombobox',
                    fieldbackground='#363650',
                    background='#363650',
                    foreground='#cdd6f4',
                    selectbackground='#89b4fa',
                    selectforeground='#1e1e2e')

    app = TicTacToeGUI(root)
    root.mainloop()