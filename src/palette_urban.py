# Urban Layout Styles guide from https://urbaninstitute.github.io/graphics-styleguide/

from typing import Final, Dict, Tuple

# Single colors
CYAN: Final[str] = '#1696d2'
GRAY: Final[str] = '#d2d2d2'
BLACK: Final[str] = '#000000'
YELLOW: Final[str] = '#fdbf11'
MAGENTA: Final[str] = '#ec008b'
GREEN: Final[str] = '#55b748'
RED: Final[str] = '#db2b27'
OCEAN: Final[str] = '#0a4c6a'
DARK: Final[str] = '#332d2f'

# Color shades
CYAN_SHADES: Final[list[str]] = ["#CFE8F3", "#A2D4EC", "#73BFE2", "#46ABDB",
                                 "#1696D2", "#12719E", "#0A4C6A", "#062635"]
GRAY_SHADES: Final[list[str]] = ["#D5D5D4", "#ADABAC", "#848081", "#5C5859",
                                 "#332D2F", "#262223", "#1A1717", "#0E0C0D"]
YELLOW_SHADES: Final[list[str]] = ["#FFF2CF", "#FCE39E", "#FDD870", "#FCCB41",
                                   "#FDBF11", "#E88E2D", "#CA5800", "#843215"]
MAGENTA_SHADES: Final[list[str]] = ["#F5CBDF", "#EB99C2", "#E46AA7", "#E54096",
                                    "#EC008B", "#AF1F6B", "#761548", "#351123"]
GREEN_SHADES: Final[list[str]] = ["#DCEDD9", "#BCDEB4", "#98CF90", "#78C26D",
                                  "#55B748", "#408941", "#2C5C2D", "#1A2E19"]
RED_SHADES: Final[list[str]] = ["#F8D5D4", "#F1AAA9", "#E9807D", "#E25552",
                                "#DB2B27", "#A4201D", "#6E1614", "#370B0A"]

# Categorical groups
BAR: Final[Dict[int, list[str]]] = {
    1: [CYAN],
    2: [CYAN, YELLOW],
    3: [CYAN, YELLOW, BLACK],
    4: [GREEN, MAGENTA, YELLOW, CYAN],
    5: [CYAN, OCEAN, MAGENTA, YELLOW, DARK],
    6: [CYAN, GREEN, MAGENTA, YELLOW, DARK, OCEAN],
    'political': [CYAN, RED]
}

BAR_EDGE: Final[Dict[int, list[str]]] = {
    1: [CYAN_SHADES[-2]],
    2: [CYAN_SHADES[-2], YELLOW_SHADES[-2]],
    3: [CYAN_SHADES[-2], YELLOW_SHADES[-2], GRAY_SHADES[-4]],
    4: [GRAY_SHADES[-4], MAGENTA_SHADES[-2], YELLOW_SHADES[-2], CYAN_SHADES[-2]],
    5: [CYAN_SHADES[-2], CYAN_SHADES[-4], MAGENTA_SHADES[-2], YELLOW_SHADES[-2], GRAY_SHADES[-4]],
    6: [CYAN_SHADES[-2], GREEN_SHADES[-2], MAGENTA_SHADES[-2], YELLOW_SHADES[-2], GRAY_SHADES[-4], CYAN_SHADES[-4]],
    'political': [CYAN_SHADES[-2], RED_SHADES[-2]]
}

BOX: Final[Dict[int, list[str]]] = {
    1: [CYAN],
    2: [CYAN, YELLOW],
    3: [CYAN, YELLOW, MAGENTA],
    4: [GREEN, MAGENTA, YELLOW, CYAN],
    5: [CYAN, CYAN_SHADES[-4], MAGENTA, YELLOW, GREEN],
    6: [CYAN, GREEN, MAGENTA, YELLOW, RED, CYAN_SHADES[-4]],
    'political': [CYAN, RED]
}

SEQUENTIAL: Final[Dict[str, list[str]]] = {
    'CYAN': CYAN_SHADES,
    'GRAY': GRAY_SHADES,
    'YELLOW': YELLOW_SHADES,
    'MAGENTA': MAGENTA_SHADES,
    'GREEN': GREEN_SHADES,
    'RED': RED_SHADES
}

# Heatmap continuous scale
HEATMAP_SCALE: Final[list[str]] = ['#ca5800', '#fdbf11', '#fdd870', '#fff2cf',
                                   '#cfe8f3', '#73bfe2', '#1696d2', '#0a4c6a']

# Args for Plotly class
PALETTE_ARGS = {
    'bar_colors': BAR,
    'bar_edge_colors': BAR_EDGE,
    'box_colors': BOX,
    'sequential': SEQUENTIAL
}


def hex_to_rgb(hex_color: str) -> tuple[int, ...]:
    """Convert hex color to RGB."""
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))


def generate_shades(
        light_tone: str,
        dark_tone: str,
        n_shades: int
) -> list[str]:
    """Generate n shades of a color between two hex colors."""

    rgb_light = hex_to_rgb(light_tone)
    rgb_dark = hex_to_rgb(dark_tone)

    intermediate_colors = []
    for i in range(1, n_shades + 1):
        ratio = i / (n_shades + 1)
        r = int(rgb_light[0] * ratio + rgb_dark[0] * (1 - ratio))
        g = int(rgb_light[1] * ratio + rgb_dark[1] * (1 - ratio))
        b = int(rgb_light[2] * ratio + rgb_dark[2] * (1 - ratio))
        intermediate_color = "#{:02X}{:02X}{:02X}".format(r, g, b)
        intermediate_colors.append(intermediate_color)

    return intermediate_colors
