from __future__ import annotations
from enum import Enum
import random

class Color(Enum):
	RED = "R"
	GREEN = "G"
	BLUE = "B"
	BLACK = "K"
	YELLOW = "Y"

# ANSI color codes for terminal output
ANSI_COLORS = {
    Color.RED: "\033[91m",      # Bright red
    Color.GREEN: "\033[92m",    # Bright green
    Color.BLUE: "\033[94m",     # Bright blue
    Color.BLACK: "\033[37m",    # Light grey (since pure black might not be visible)
    Color.YELLOW: "\033[93m",   # Bright yellow
}
ANSI_RESET = "\033[0m"  # Reset to default color

class Piece:
    def __init__(self, structure: list[list[Color | None]]):
        self.structure = structure

    def get_rotated(self, times: int = 0) -> Piece:
        """Return a new piece rotated clockwise by 90 degrees, times times."""
        if times == 0:
            return self

        current = self.structure
        
        for _ in range(times):
            current = [
                [current[1][0], current[0][0]],
                [current[1][1], current[0][1]]
            ]
        
        return Piece(current)

piece_placement: list[list[Piece | None]] = [[None for _ in range(5)] for _ in range(5)]
used_field: list[list[bool | bool]] = [[False for _ in range(5)] for _ in range(5)]

pieces = [
   Piece([[Color.YELLOW, Color.BLUE],[Color.GREEN, None]]),
   Piece([[Color.YELLOW, Color.BLUE],[Color.GREEN, None]]),
   Piece([[Color.RED, Color.BLUE],[Color.BLACK, None]]),
   Piece([[Color.RED, Color.GREEN],[Color.YELLOW, None]]),
   Piece([[Color.RED, Color.YELLOW],[Color.BLUE, None]]),
   Piece([[Color.GREEN, Color.YELLOW],[Color.BLACK, None]]),
   Piece([[Color.BLACK, Color.GREEN],[Color.BLUE, None]]),
   Piece([[Color.BLACK, None],[Color.RED, None]]),
   Piece([[Color.BLACK, None],[Color.RED, None]]),
]

def render_letter(letter: Color | None) -> str:
    if letter is None:
        return " "
    return f"{ANSI_COLORS[letter]}{letter.value}{ANSI_RESET}"

def render_piece(piece: Piece) -> str:
    """Renders a piece as a string for terminal output."""
    result = []
    
    # Top border
    top_border = ""
    for col in range(2):
        if piece.structure[0][col] is not None:
            top_border += "+---"
    if top_border:
        top_border += "+"
        result.append(top_border)
    
    # Top row content
    top_content = ""
    for col in range(2):
        if piece.structure[0][col] is not None:
            top_content += f"| {render_letter(piece.structure[0][col])} "
    if top_content:
        top_content += "|"
        result.append(top_content)
    
    # Middle border
    middle_border = ""
    for col in range(2):
        if piece.structure[0][col] is not None or piece.structure[1][col] is not None:
            middle_border += "+---"
    if middle_border:
        middle_border += "+"
        result.append(middle_border)
    
    # Bottom row content
    bottom_content = ""
    for col in range(2):
        if piece.structure[1][col] is not None:
            bottom_content += f"| {render_letter(piece.structure[1][col])} "
    if bottom_content:
        bottom_content += "|"
        result.append(bottom_content)
    
    # Bottom border
    bottom_border = ""
    for col in range(2):
        if piece.structure[1][col] is not None:
            bottom_border += "+---"
    if bottom_border:
        bottom_border += "+"
        result.append(bottom_border)
    
    return "\n".join(result)

def print_colors_with_codes():
    """Print all colors in terminal with their ANSI color codes"""
    print("Colors and their ANSI codes:")
    print("=" * 40)
    
    for color in Color:
        ansi_code = ANSI_COLORS[color]
        # Print color name, ANSI code, and colored text
        print(f"{ansi_code}●●● {color.name} ●●●{ANSI_RESET}")
    
    print("=" * 40)
    print()
    
def print_sudoku_pieces():
    print("Sudoku pieces:")
    print("-" * 30)
    for i, piece in enumerate(pieces, 1):
        print(f"Piece {i}:")
        print(render_piece(piece))
        print()

def print_board(board: list[list[Color | None]]):
    print("+---+---+---+---+---+")
    for row in board:
        row_str = "|"
        for cell in row:
            if cell is None:
                row_str += "  |"
            else:
                row_str += f" {render_letter(cell)} |"
        print(row_str)
        print("+---+---+---+---+---+")
    print()

def test_graphics():
    print_colors_with_codes()
    print_sudoku_pieces()
    print_board(
        [
            [c for c in Color],
            [c for c in Color],
            [c for c in Color],
            [c for c in Color],
            [c for c in Color]
        ]
    )
    
class SudokuSolver:
    def __init__(self, show_progress=False):
        self.board = [[None for _ in range(5)] for _ in range(5)]
        self.used_pieces = [False] * len(pieces)
        self.solutions = []
        self.show_progress = show_progress
        self.attempts = 0
        self.max_depth = 0

    def is_valid_placement(self, piece: Piece, row: int, col: int) -> bool:
        """Check if placing a piece at (row, col) is valid"""
        # Check if piece fits within board bounds
        for r in range(2):
            for c in range(2):
                if piece.structure[r][c] is not None:
                    board_row, board_col = row + r, col + c
                    if board_row >= 5 or board_col >= 5:
                        return False
                    # Check if cell is already occupied
                    if self.board[board_row][board_col] is not None:
                        return False
        return True

    def place_piece(self, piece: Piece, row: int, col: int):
        """Place a piece on the board at (row, col)"""
        for r in range(2):
            for c in range(2):
                if piece.structure[r][c] is not None:
                    self.board[row + r][col + c] = piece.structure[r][c]

    def remove_piece(self, piece: Piece, row: int, col: int):
        """Remove a piece from the board at (row, col)"""
        for r in range(2):
            for c in range(2):
                if piece.structure[r][c] is not None:
                    self.board[row + r][col + c] = None

    def is_valid_board(self) -> bool:
        """Check if current board state satisfies sudoku constraints"""
        # Check rows for unique colors
        for row in self.board:
            colors_in_row = [cell for cell in row if cell is not None]
            if len(colors_in_row) != len(set(colors_in_row)):
                return False

        # Check columns for unique colors
        for col in range(5):
            colors_in_col = [self.board[row][col] for row in range(5) if self.board[row][col] is not None]
            if len(colors_in_col) != len(set(colors_in_col)):
                return False

        return True

    def is_complete(self) -> bool:
        """Check if puzzle is complete (all pieces used and board filled)"""
        # Check if all pieces are used
        if not all(self.used_pieces):
            return False
        
        # Check if board is completely filled
        for row in self.board:
            for cell in row:
                if cell is None:
                    return False
        
        # Check final constraints
        return self.is_valid_board()

    def solve(self, depth=0) -> bool:
        """Optimized backtracking solver with progress tracking"""
        self.attempts += 1
        self.max_depth = max(self.max_depth, depth)
        
        # Show progress periodically - less frequent for performance
        if self.show_progress and self.attempts % 10000 == 0:
            pieces_placed = sum(self.used_pieces)
            print(f"Progress: {self.attempts:,} attempts | Pieces: {pieces_placed}/9 | Depth: {depth}")
        
        # If all pieces are used, check if solution is valid
        if all(self.used_pieces):
            if self.is_complete():
                # Store a copy of the solution
                solution = [row[:] for row in self.board]
                self.solutions.append(solution)
                if self.show_progress:
                    print(f"\n🎉 SOLUTION FOUND after {self.attempts:,} attempts!")
                return True
            return False

        # Early pruning: check if current state can lead to valid solution
        if not self.can_be_completed():
            return False

        # Try placing each unused piece at each valid position
        # Use smarter ordering: try pieces with fewer placement options first
        piece_options = []
        for piece_idx, piece in enumerate(pieces):
            if not self.used_pieces[piece_idx]:
                valid_placements = self.count_valid_placements(piece)
                piece_options.append((valid_placements, piece_idx, piece))
        
        # Sort by number of valid placements (constraint satisfaction heuristic)
        piece_options.sort()
        
        for _, piece_idx, piece in piece_options:
            # Try all 4 rotations of the piece
            for rotation in range(4):
                rotated_piece = piece.get_rotated(rotation)
                
                # Try all positions on the board
                for row in range(5):
                    for col in range(5):
                        if self.is_valid_placement(rotated_piece, row, col):
                            # Place the piece
                            self.place_piece(rotated_piece, row, col)
                            self.used_pieces[piece_idx] = True
                            
                            # Check if placement maintains sudoku constraints
                            if self.is_valid_board():
                                # Recursively try to place remaining pieces
                                if self.solve(depth + 1):
                                    return True
                            
                            # Backtrack
                            self.remove_piece(rotated_piece, row, col)
                            self.used_pieces[piece_idx] = False
        
        return False

    def count_valid_placements(self, piece: Piece) -> int:
        """Count how many valid placements a piece has on current board"""
        count = 0
        for rotation in range(4):
            rotated_piece = piece.get_rotated(rotation)
            for row in range(5):
                for col in range(5):
                    if self.is_valid_placement(rotated_piece, row, col):
                        # Temporarily place to check constraints
                        self.place_piece(rotated_piece, row, col)
                        if self.is_valid_board():
                            count += 1
                        self.remove_piece(rotated_piece, row, col)
        return count

    def can_be_completed(self) -> bool:
        """Check if current board state can potentially lead to a complete solution"""
        # Check if any row or column already has duplicate colors
        for row in range(5):
            colors_in_row = [self.board[row][col] for col in range(5) if self.board[row][col] is not None]
            if len(colors_in_row) != len(set(colors_in_row)):
                return False
        
        for col in range(5):
            colors_in_col = [self.board[row][col] for row in range(5) if self.board[row][col] is not None]
            if len(colors_in_col) != len(set(colors_in_col)):
                return False
        
        return True

    def show_current_state(self, depth, final=False):
        """Display current board state and progress info"""
        import os
        if not final:
            os.system('clear' if os.name == 'posix' else 'cls')  # Clear screen
        
        pieces_placed = sum(self.used_pieces)
        print(f"{'='*50}")
        print(f"Attempt #{self.attempts} | Depth: {depth} | Pieces placed: {pieces_placed}/{len(pieces)}")
        print(f"Max depth reached: {self.max_depth}")
        print(f"{'='*50}")
        
        # Show which pieces are used
        print("Pieces status:")
        for i, used in enumerate(self.used_pieces):
            status = "✅" if used else "⭕"
            print(f"  Piece {i+1}: {status}")
        
        print("\nCurrent board:")
        self.print_current_board()
        
        # if not final:
        #     time.sleep(0.1)  # Small delay to see the progress

    def print_current_board(self):
        """Print the current state of the board"""
        print("+---+---+---+---+---+")
        for row in self.board:
            row_str = "|"
            for cell in row:
                if cell is None:
                    row_str += " . |"
                else:
                    row_str += f" {render_letter(cell)} |"
            print(row_str)
            print("+---+---+---+---+---+")
        print()

    def print_solution(self):
        """Print the solved board"""
        if not self.solutions:
            print("No solution found!")
            return
        
        print("Solution found!")
        print("=" * 25)
        print_board(self.solutions[0])
        
        # Verify solution
        self.verify_solution()

    def verify_solution(self):
        """Verify that the solution meets all constraints"""
        if not self.solutions:
            return
            
        board = self.solutions[0]
        print("Verification:")
        
        # Check rows
        print("Row constraints:")
        for i, row in enumerate(board):
            colors = [cell.name for cell in row]
            unique_colors = set(colors)
            print(f"  Row {i+1}: {colors} -> {len(unique_colors) == 5} (unique: {sorted(unique_colors)})")
        
        # Check columns
        print("Column constraints:")
        for j in range(5):
            col = [board[i][j].name for i in range(5)]
            unique_colors = set(col)
            print(f"  Col {j+1}: {col} -> {len(unique_colors) == 5} (unique: {sorted(unique_colors)})")

def solve_riddle(show_progress=False):
    """Main solver function"""
    print("Starting Wood Sudoku Solver...")
    print("Puzzle constraints:")
    print("- Each piece used exactly once")
    print("- Each row has unique colors")
    print("- Each column has unique colors")
    print("- All cells filled")
    print()
    
    solver = SudokuSolver(show_progress=show_progress)
    
    if show_progress:
        print("Solving with progress display...")
        print("Press Ctrl+C to stop if it takes too long")
    else:
        print("Solving...")
    
    try:
        if solver.solve():
            solver.print_solution()
            print(f"\nSolver statistics:")
            print(f"- Total attempts: {solver.attempts}")
            print(f"- Maximum depth reached: {solver.max_depth}")
        else:
            print("No solution exists for this puzzle!")
            print(f"Tried {solver.attempts} combinations")
    except KeyboardInterrupt:
        print(f"\n\nSolver interrupted by user after {solver.attempts} attempts")
        print(f"Maximum depth reached: {solver.max_depth}")
        if solver.solutions:
            solver.print_solution()
        else:
            print("No solution found yet.")

def main():
    print("Wood Sudoku Puzzle")
    print("=" * 30)
    print()
    
    # Show the pieces
    print_sudoku_pieces()
    
    # Ask user if they want to see progress
    import sys
    show_progress = False
    if len(sys.argv) > 1 and sys.argv[1] == "--progress":
        show_progress = True
        print("🔍 Progress mode enabled - this will be slower but shows the solving process")
    else:
        print("💨 Fast mode - use '--progress' flag to see solving progress")
    
    # Solve the puzzle
    solve_riddle(show_progress=show_progress)

if __name__ == "__main__":
    main()
