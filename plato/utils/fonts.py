"""
Colours and fonts for logging messages
"""


def colourize(message, colour="yellow", style="bold"):
    """Returns the message in input colour and style"""
    reset = "\033[0m"
    colours = {
        "green": "\033[92m",
        "blue": "\033[94m",
        "yellow": "\033[93m",
        "red": "\033[91m",
    }
    styles = {"standard": "", "bold": "\033[1m", "underline": "\033[4m"}

    if not (colour in colours and style in styles):
        raise ValueError(
            f"Your colour '{colour}' or your style '{style}' is not supported."
            f"\nThe supported colours are: {', '.join(colours)}. \nThe supported styles are: {', '.join(styles)}."
        )

    return colours[colour] + styles[style] + message + reset
