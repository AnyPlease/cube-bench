def format_rubiks_cube(scrambled_cube_str):
    """
    Parses a textual net layout of a 3x3 Rubik's Cube into a structured dictionary
    with keys: Top, Left, Front, Right, Back, Bottom.

    Assumes input lines are:

                 [y][y][y]
                 [y][y][y]
                 [y][y][y]
    [o][o][o][b][b][b][r][r][r][g][g][g]
    [r][r][r][g][g][g][o][o][o][b][b][b]
    [r][r][r][g][g][g][o][o][o][b][b][b]
                 [w][w][w]
                 [w][w][w]
                 [w][w][w]

    Each line has 36 chars in the middle rows, 9 chars in the top/bottom rows (after stripping whitespace).
    """
    def parse_row(segment):
        """Extracts the 3 colors from a 9-character substring like '[x][x][x]' and maps single-letter color codes to full names."""
        color_map = {
            'y': 'Yellow', 'g': 'Green',
            'r': 'Red', 'o': 'Orange',
            'b': 'Blue', 'w': 'White'}
        # We assume segment has exactly 9 characters, each color is at index 1, 4, 7.
        chars = [segment[1], segment[4], segment[7]]
        return [color_map.get(c, c) for c in chars]

    # Split into lines and filter out any empty or whitespace-only lines
    lines = [line for line in scrambled_cube_str.split('\n') if line.strip()]

    # Prepare arrays for each face
    top = []; left = []; front = []; right = []; back = []; bottom = []

    # For the top (lines 0..2): parse the stripped line
    for i in range(3):
        segment = lines[i].strip()
        top.append(parse_row(segment))

    # For the middle (lines 3..5): parse the full line in 9-char chunks
    for i in range(3, 6):
        line = lines[i].strip()
        left.append(parse_row(line[0:9]))
        front.append(parse_row(line[9:18]))
        right.append(parse_row(line[18:27]))
        back.append(parse_row(line[27:36]))

    # For the bottom (lines 6..8): parse the stripped line
    for i in range(6, 9):
        segment = lines[i].strip()
        bottom.append(parse_row(segment))

    # Combine faces into dictionary
    cube_dict = {
        "Top": top,
        "Left": left,
        "Front": front,
        "Right": right,
        "Back": back,
        "Bottom": bottom  # Changed "Down" to "Bottom" for consistency
    }

    return cube_dict, format_cube_dict_as_string(cube_dict)

def format_cube_dict_as_string(cube_dict):
    """Formats the cube dictionary into a clear, readable string showing each face and its colors."""
    face_strings = []
    for face_name, rows in cube_dict.items():
        face_strings.append(f"{face_name}:")
        for row in rows:
            # Format each row as a string for better readability
            face_strings.append(f"  {row}")
        face_strings.append("")  # Add a blank line for clarity between faces

    return "\n".join(face_strings)

if __name__ == "__main__":
    # Example usage
    scrambled_cube = "         [y][y][w]\n         [y][y][w]\n         [w][w][w]\n[g][g][b][o][g][g][r][o][o][g][r][r]\n[o][r][r][b][g][g][r][o][o][g][b][b]\n[o][r][r][b][o][o][g][b][b][r][b][b]\n         [y][y][y]\n         [w][w][w]\n         [y][y][w]\n"
    cube_dict, cube_str = format_rubiks_cube(scrambled_cube)
    print(cube_str)
    print(cube_dict)