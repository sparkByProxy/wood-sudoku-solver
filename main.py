from __future__ import annotations
from enum import Enum
import time
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


class Color(Enum):
    RED = "R"
    GREEN = "G"
    BLUE = "B"
    BLACK = "K"
    YELLOW = "Y"


# ANSI color codes for terminal output
ANSI_COLORS = {
    Color.RED: "\033[91m",
    Color.GREEN: "\033[92m",
    Color.BLUE: "\033[94m",
    Color.BLACK: "\033[37m",
    Color.YELLOW: "\033[93m",
}
ANSI_RESET = "\033[0m"


@dataclass
class Position:
    row: int
    col: int


@dataclass
class PlacementInfo:
    piece_id: int
    rotation: int
    position: Position
    cells: List[Tuple[int, int, Color]]  # (row, col, color) tuples


class FastPiece:
    def __init__(self, piece_id: int, structure: list[list[Color | None]]):
        self.piece_id = piece_id
        self.structure = structure
        # Precompute all rotations and their cell positions
        self.rotations = self._precompute_rotations()

    def _precompute_rotations(self) -> List[List[Tuple[int, int, Color]]]:
        """Precompute all 4 rotations and their occupied cells"""
        rotations = []
        current = self.structure

        for rotation in range(4):
            cells = []
            for r in range(2):
                for c in range(2):
                    if current[r][c] is not None:
                        cells.append((r, c, current[r][c]))
            rotations.append(cells)

            # Rotate for next iteration
            current = [
                [current[1][0], current[0][0]],
                [current[1][1], current[0][1]],
            ]

        return rotations


class SudokuSolver:
    def __init__(self, show_progress=False):
        self.board = [[None for _ in range(5)] for _ in range(5)]
        self.solutions = []
        self.show_progress = show_progress
        self.attempts = 0

        # Create pieces
        self.pieces = self._create_fast_pieces()

        # Fast constraint tracking using sets
        self.row_colors = [set() for _ in range(5)]
        self.col_colors = [set() for _ in range(5)]
        self.used_pieces = set()
        self.placed_pieces = []  # Stack of placements for backtracking

        # Track which piece is at each position for solution display
        self.piece_positions = (
            {}
        )  # (row, col) -> (piece_id, rotation, original_position)

        # Precompute all valid placements
        self.valid_placements = self._precompute_valid_placements()

    def _create_fast_pieces(self) -> List[FastPiece]:
        """Create fast pieces with precomputed rotations"""
        piece_structures = [
            [[Color.YELLOW, Color.BLUE], [Color.GREEN, None]],
            [[Color.YELLOW, Color.BLUE], [Color.GREEN, None]],
            [[Color.RED, Color.BLUE], [Color.BLACK, None]],
            [[Color.RED, Color.GREEN], [Color.YELLOW, None]],
            [[Color.RED, Color.YELLOW], [Color.BLUE, None]],
            [[Color.GREEN, Color.YELLOW], [Color.BLACK, None]],
            [[Color.BLACK, Color.GREEN], [Color.BLUE, None]],
            [[Color.BLACK, None], [Color.RED, None]],
            [[Color.BLACK, None], [Color.RED, None]],
        ]

        return [
            FastPiece(i, structure)
            for i, structure in enumerate(piece_structures)
        ]

    def _precompute_valid_placements(self) -> Dict[int, List[PlacementInfo]]:
        """Precompute all valid placements for each piece"""
        placements = {}

        for piece in self.pieces:
            piece_placements = []

            for rotation in range(4):
                cells = piece.rotations[rotation]

                # Try all possible positions
                for start_row in range(5):
                    for start_col in range(5):
                        # Check if piece fits at this position
                        valid = True
                        absolute_cells = []

                        for r, c, color in cells:
                            abs_row, abs_col = start_row + r, start_col + c
                            if abs_row >= 5 or abs_col >= 5:
                                valid = False
                                break
                            absolute_cells.append((abs_row, abs_col, color))

                        if valid:
                            placement = PlacementInfo(
                                piece_id=piece.piece_id,
                                rotation=rotation,
                                position=Position(start_row, start_col),
                                cells=absolute_cells,
                            )
                            piece_placements.append(placement)

            placements[piece.piece_id] = piece_placements

        return placements

    def can_place(self, placement: PlacementInfo) -> bool:
        """Ultra-fast constraint checking using precomputed sets"""
        for row, col, color in placement.cells:
            # Check if cell is occupied
            if self.board[row][col] is not None:
                return False

            # Check color constraints instantly using sets
            if color in self.row_colors[row] or color in self.col_colors[col]:
                return False

        return True

    def place_piece(self, placement: PlacementInfo) -> None:
        """Place piece and update constraint tracking"""
        for row, col, color in placement.cells:
            self.board[row][col] = color
            self.row_colors[row].add(color)
            self.col_colors[col].add(color)
            # Track piece position for display
            self.piece_positions[(row, col)] = (
                placement.piece_id,
                placement.rotation,
                placement.position,
            )

        self.used_pieces.add(placement.piece_id)
        self.placed_pieces.append(placement)

    def remove_piece(self, placement: PlacementInfo) -> None:
        """Remove piece and update constraint tracking"""
        for row, col, color in placement.cells:
            self.board[row][col] = None
            self.row_colors[row].remove(color)
            self.col_colors[col].remove(color)
            # Remove piece position tracking
            if (row, col) in self.piece_positions:
                del self.piece_positions[(row, col)]

        self.used_pieces.remove(placement.piece_id)
        self.placed_pieces.pop()

    def get_most_constrained_piece(self) -> Optional[int]:
        """Get the piece with fewest valid placements (MRV heuristic)"""
        if len(self.used_pieces) == len(self.pieces):
            return None

        min_placements = float("inf")
        best_piece = None

        for piece_id in range(len(self.pieces)):
            if piece_id in self.used_pieces:
                continue

            valid_count = 0
            for placement in self.valid_placements[piece_id]:
                if self.can_place(placement):
                    valid_count += 1

            if valid_count < min_placements:
                min_placements = valid_count
                best_piece = piece_id

                # If no valid placements, fail fast
                if valid_count == 0:
                    return best_piece

        return best_piece

    def solve(self) -> bool:
        """Optimized solver with constraint propagation and heuristics"""
        self.attempts += 1

        # Success condition
        if len(self.used_pieces) == len(self.pieces):
            # Verify solution is complete and valid
            if self.is_complete_solution():
                self.solutions.append([row[:] for row in self.board])
                return True
            return False

        # Get most constrained piece (MRV - Minimum Remaining Values)
        piece_id = self.get_most_constrained_piece()
        if piece_id is None:
            return False

        # Try all valid placements for this piece
        valid_placements = [
            p for p in self.valid_placements[piece_id] if self.can_place(p)
        ]

        # Sort placements by how much they constrain future moves (LCV - Least Constraining Value)
        valid_placements.sort(key=lambda p: self.count_conflicts(p))

        for placement in valid_placements:
            # Make move
            self.place_piece(placement)

            # Forward checking: ensure remaining pieces can still be placed
            if self.has_solution_potential():
                if self.solve():
                    return True

            # Backtrack
            self.remove_piece(placement)

        return False

    def count_conflicts(self, placement: PlacementInfo) -> int:
        """Count how many future placements this move would eliminate"""
        conflicts = 0

        # Temporarily place the piece
        self.place_piece(placement)

        # Count valid placements for remaining pieces
        for piece_id in range(len(self.pieces)):
            if piece_id in self.used_pieces:
                continue

            for other_placement in self.valid_placements[piece_id]:
                if not self.can_place(other_placement):
                    conflicts += 1

        # Remove the piece
        self.remove_piece(placement)

        return conflicts

    def has_solution_potential(self) -> bool:
        """Quick check if remaining pieces can theoretically be placed"""
        for piece_id in range(len(self.pieces)):
            if piece_id in self.used_pieces:
                continue

            # Check if this piece has at least one valid placement
            has_valid_placement = False
            for placement in self.valid_placements[piece_id]:
                if self.can_place(placement):
                    has_valid_placement = True
                    break

            if not has_valid_placement:
                return False

        return True

    def is_complete_solution(self) -> bool:
        """Verify the solution is complete and valid"""
        # Check all cells are filled
        for row in range(5):
            for col in range(5):
                if self.board[row][col] is None:
                    return False

        # Check each row has all 5 colors
        for row in range(5):
            if len(self.row_colors[row]) != 5:
                return False

        # Check each column has all 5 colors
        for col in range(5):
            if len(self.col_colors[col]) != 5:
                return False

        return True

    def render_letter(self, color: Color) -> str:
        """Render colored letter for terminal display"""
        return f"{ANSI_COLORS[color]}{color.value}{ANSI_RESET}"

    def print_solution(self):
        """Print the solved board with colors and piece positions"""
        if not self.solutions:
            print("No solution found!")
            return

        print("\n🎉 SOLUTION FOUND! 🎉")
        print("=" * 50)

        # Print the colored board
        print("\n📋 Solved Board:")
        print("+---+---+---+---+---+")

        board = self.solutions[0]
        for row in board:
            row_str = "|"
            for cell in row:
                row_str += f" {self.render_letter(cell)} |"
            print(row_str)
            print("+---+---+---+---+---+")

        print(f"\nSolved in {self.attempts} attempts!")

        # Show piece placement information
        self.show_piece_placements()

        # Verify constraints
        self.verify_solution(board)

    def show_piece_placements(self):
        """Show which piece is placed at which position"""
        print("\n🧩 Piece Placement Details:")
        print("=" * 50)

        # Group positions by piece
        piece_locations = {}
        for (row, col), (
            piece_id,
            rotation,
            original_pos,
        ) in self.piece_positions.items():
            if piece_id not in piece_locations:
                piece_locations[piece_id] = []
            piece_locations[piece_id].append(
                (row, col, rotation, original_pos)
            )

        # Display each piece's placement
        for piece_id in sorted(piece_locations.keys()):
            locations = piece_locations[piece_id]

            # Find the top-left corner (original position)
            min_row = min(row for row, col, _, _ in locations)
            min_col = min(
                col for row, col, _, _ in locations if row == min_row
            )

            # Get rotation info
            _, _, rotation, _ = locations[
                0
            ]  # All cells of same piece have same rotation

            print(f"\n🔸 Piece {piece_id + 1}:")
            print(f"   Position: Row {min_row + 1}, Column {min_col + 1}")
            print(f"   Rotation: {rotation * 90}° clockwise")
            print(f"   Occupies cells:")

            # Show all cells this piece occupies
            for row, col, _, _ in sorted(locations):
                color = self.board[row][col]
                print(
                    f"     • ({row + 1}, {col + 1}) = {self.render_letter(color)}"
                )

    def verify_solution(self, board):
        """Verify the solution meets all constraints"""
        print("\n✅ Verification:")

        all_valid = True

        # Check rows
        print("\n🔄 Row Constraints:")
        for i, row in enumerate(board):
            colors = {cell for cell in row}
            if len(colors) == 5 and colors == set(Color):
                print(f"  Row {i+1}: ✅ All 5 colors unique")
            else:
                print(f"  Row {i+1}: ❌ Missing or duplicate colors")
                all_valid = False

        # Check columns
        print("\n📏 Column Constraints:")
        for j in range(5):
            colors = {board[i][j] for i in range(5)}
            if len(colors) == 5 and colors == set(Color):
                print(f"  Col {j+1}: ✅ All 5 colors unique")
            else:
                print(f"  Col {j+1}: ❌ Missing or duplicate colors")
                all_valid = False

        print("\n🧩 Piece Usage:")
        used_count = len(
            set(piece_id for piece_id, _, _ in self.piece_positions.values())
        )
        if used_count == 9:
            print(f"  ✅ All 9 pieces used exactly once")
        else:
            print(f"  ❌ Only {used_count}/9 pieces used")
            all_valid = False

        if all_valid:
            print("\n🎊 Perfect solution! All constraints satisfied!")
        else:
            print("\n⚠️  Solution has constraint violations!")


def solve():
    """Main fast solver function"""
    print("🚀 Ultra-Fast Wood Sudoku Solver with Piece Tracking")
    print("=" * 60)
    print("Using advanced constraint satisfaction techniques:")
    print("• Precomputed piece placements")
    print("• Instant constraint validation")
    print("• Most Constrained Variable (MCV) heuristic")
    print("• Least Constraining Value (LCV) heuristic")
    print("• Forward checking and early pruning")
    print("• Complete piece position tracking")
    print()

    start_time = time.time()
    solver = SudokuSolver()

    print("Solving...")
    if solver.solve():
        end_time = time.time()
        solver.print_solution()

        print(f"\n⚡ Performance Statistics:")
        print(f"• Solving time: {end_time - start_time:.4f} seconds")
        print(f"• Total attempts: {solver.attempts:,}")
        if end_time - start_time > 0:
            print(
                f"• Speed: {solver.attempts / (end_time - start_time):.0f} attempts/second"
            )

    else:
        end_time = time.time()
        print("❌ No solution exists for this puzzle!")
        print(f"Verified in {end_time - start_time:.4f} seconds")
        print(f"Tried {solver.attempts:,} combinations")


if __name__ == "__main__":
    solve()
