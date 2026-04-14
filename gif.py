from pathlib import Path
import difflib
import re

from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parent
STEPS_FILE = ROOT / "steps.txt"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_GIF = OUTPUT_DIR / "steps_generation.gif"

WIDTH, HEIGHT = 1400, 450
OUTER_MARGIN_X = 42
TEXT_BOX_MARGIN_X = 54
TEXT_INSET_X = 78
PANEL_X0, PANEL_Y0 = OUTER_MARGIN_X, 20
PANEL_X1, PANEL_Y1 = WIDTH - OUTER_MARGIN_X, HEIGHT - 24
TEXT_X0, TEXT_Y0 = TEXT_INSET_X, 158
TEXT_X1, TEXT_Y1 = WIDTH - TEXT_INSET_X, HEIGHT - 44

BG = "#ffffff"
PANEL = "#fffdf8"
PANEL_2 = "#fffdf8"
TEXT = "#171717"
MUTED = "#78716c"
ACCENT = "#2563eb"
ACCENT_2 = "#111111"
ACCENT_3 = "#a8a29e"
NEW_BG = "#f6e7a1"
NEW_TEXT = "#171717"
BAR_BG = "#ddd6ce"
CARD = "#fffdf8"
CARD_2 = "#f7f3ed"
OUTLINE = "#e7e0d7"
SUCCESS = "#2563eb"
HIGHLIGHT_BORDER = "#d64b4b"

TOKEN_RE = re.compile(r"\[[^\]]+\]|[A-Za-z0-9_]+(?:'[A-Za-z0-9_]+)*|[^\w\s]+|\s+")


def parse_steps(text: str):
    parts = re.split(r"=+ Step (\d+) =+\n\n", text.strip())
    steps = []
    for i in range(1, len(parts), 2):
        step_no = int(parts[i])
        body = parts[i + 1].strip()
        steps.append((step_no, body))
    return steps


def tokenize(text: str):
    return TOKEN_RE.findall(text)


def diff_tokens(prev_text: str, curr_text: str):
    prev_tokens = tokenize(prev_text)
    curr_tokens = tokenize(curr_text)
    matcher = difflib.SequenceMatcher(a=prev_tokens, b=curr_tokens, autojunk=False)

    new_indices = set()
    for tag, _i1, _i2, j1, j2 in matcher.get_opcodes():
        if tag in ("insert", "replace"):
            for j in range(j1, j2):
                if not curr_tokens[j].isspace():
                    new_indices.add(j)

    return [(tok, idx in new_indices) for idx, tok in enumerate(curr_tokens)]


# def load_font(size: int, bold: bool = False, family: str = "sans"):
#     if family == "serif":
#         candidates = [
#             "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
#             "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
#         ]
#     elif family == "mono":
#         candidates = [
#             "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
#             "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
#         ]
#     else:
#         candidates = [
#             "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
#             "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
#         ]
#     for path in candidates:
#         if Path(path).exists():
#             return ImageFont.truetype(path, size=size)
#     return ImageFont.load_default()

from pathlib import Path
from PIL import ImageFont

def load_font(size: int, bold: bool = False, family: str = "sans"):
    """macOS 字体加载函数"""
    
    # macOS 常用字体路径
    font_map = {
        "sans": {
            False: "/System/Library/Fonts/Helvetica.ttc",
            True: "/System/Library/Fonts/Helvetica.ttc",  # Helvetica 包含粗体
        },
        "serif": {
            False: "/System/Library/Fonts/Times.ttc",
            True: "/System/Library/Fonts/Times.ttc",
        },
        "mono": {
            False: "/System/Library/Fonts/Menlo.ttc",
            True: "/System/Library/Fonts/Menlo.ttc",
        }
    }
    
    font_path = font_map.get(family, font_map["sans"])[bold]
    
    if Path(font_path).exists():
        try:
            # 对于 macOS 的 .ttc 文件，可能需要指定索引
            return ImageFont.truetype(font_path, size=size, index=0)
        except:
            pass
    
    # 备选方案：使用 Arial
    arial_path = "/System/Library/Fonts/Arial.ttf"
    if Path(arial_path).exists():
        try:
            return ImageFont.truetype(arial_path, size=size)
        except:
            pass
    
    # 最后的备选
    return ImageFont.load_default()

FONT_TITLE = load_font(48, bold=True, family="sans")
FONT_SUB = load_font(20, family="sans")
FONT_STEP = load_font(22, bold=True, family="mono")
FONT_TEXT = load_font(25, family="sans")
FONT_META = load_font(22, family="mono")
FONT_SMALL = load_font(18, family="mono")


def text_size(draw: ImageDraw.ImageDraw, text: str, font):
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0], box[3] - box[1]


def text_advance(draw: ImageDraw.ImageDraw, text: str, font):
    return draw.textlength(text, font=font)


def wrap_runs(draw: ImageDraw.ImageDraw, runs, font, max_width):
    lines = []
    line = []
    line_width = 0.0

    for token, is_new in runs:
        if token == "\n":
            if line:
                while line and line[-1][0].isspace():
                    line_width -= line[-1][2]
                    line.pop()
                if line:
                    lines.append(line)
            line = []
            line_width = 0
            continue

        token_advance = text_advance(draw, token, font)

        if not line and token.isspace():
            continue

        if line and (line_width + token_advance > max_width) and not token.isspace():
            while line and line[-1][0].isspace():
                line_width -= line[-1][2]
                line.pop()
            if line:
                lines.append(line)
            line = []
            line_width = 0
            if token.isspace():
                continue

        line.append((token, is_new, token_advance))
        line_width += token_advance

    if line:
        while line and line[-1][0].isspace():
            line.pop()
        if line:
            lines.append(line)

    return lines


def draw_round_rect(draw, box, radius, fill, outline=None, width=1):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def hex_to_rgba(value: str, alpha: int = 255):
    value = value.lstrip("#")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4)) + (alpha,)


def lerp_color(a, b, t: float):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def draw_vertical_gradient(img, top_color: str, bottom_color: str):
    top = hex_to_rgba(top_color)[:3]
    bottom = hex_to_rgba(bottom_color)[:3]
    draw = ImageDraw.Draw(img)
    for y in range(HEIGHT):
        t = y / max(HEIGHT - 1, 1)
        draw.line((0, y, WIDTH, y), fill=lerp_color(top, bottom, t), width=1)


def add_glow(img, box, color, blur=70):
    glow = Image.new("RGBA", img.size, (0, 0, 0, 0))
    glow_draw = ImageDraw.Draw(glow)
    glow_draw.ellipse(box, fill=color)
    glow = glow.filter(ImageFilter.GaussianBlur(blur))
    return Image.alpha_composite(img, glow)


def draw_shadowed_panel(img, box, radius, fill, outline=None, width=1, shadow_alpha=135):
    shadow = Image.new("RGBA", img.size, (0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(shadow)
    shadow_box = (box[0], box[1] + 12, box[2], box[3] + 12)
    shadow_draw.rounded_rectangle(shadow_box, radius=radius, fill=(0, 0, 0, shadow_alpha))
    shadow = shadow.filter(ImageFilter.GaussianBlur(18))
    img = Image.alpha_composite(img, shadow)
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)
    return img


def draw_pill(draw, box, text, font, fill, outline, text_fill):
    draw.rounded_rectangle(box, radius=999, fill=fill, outline=outline, width=1)
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    text_w = right - left
    text_h = bottom - top
    x = box[0] + (box[2] - box[0] - text_w) / 2
    y = box[1] + (box[3] - box[1] - text_h) / 2 - 1
    draw.text((x, y), text, font=font, fill=text_fill)


def render_frame(index: int, step_no: int, curr_text: str, prev_text: str, total_steps: int):
    img = Image.new("RGBA", (WIDTH, HEIGHT), hex_to_rgba(BG, 255))
    draw = ImageDraw.Draw(img)
    # Main panel
    img = draw_shadowed_panel(
        img,
        (PANEL_X0, PANEL_Y0, PANEL_X1, PANEL_Y1),
        radius=12,
        fill=hex_to_rgba(PANEL, 250),
        outline=hex_to_rgba(OUTLINE, 255),
        width=1,
        shadow_alpha=10,
    )
    draw = ImageDraw.Draw(img)

    # Step info
    draw.text((72, 46), f"STEP {step_no}", font=FONT_STEP, fill=MUTED)
    draw.text((WIDTH - 160, 48), f"{step_no:02d} / {total_steps:02d}", font=FONT_SMALL, fill=MUTED)

    # Progress bar
    bar_x0, bar_y0 = 72, 76
    bar_x1, bar_y1 = WIDTH - 72, 80
    draw_round_rect(draw, (bar_x0, bar_y0, bar_x1, bar_y1), radius=2, fill=BAR_BG)
    fill_w = int((bar_x1 - bar_x0) * (step_no / max(total_steps, 1)))
    if fill_w > 0:
        draw_round_rect(draw, (bar_x0, bar_y0, bar_x0 + fill_w, bar_y1), radius=2, fill=ACCENT)

    runs = diff_tokens(prev_text, curr_text)
    total_tokens = sum(1 for tok, _ in runs if not tok.isspace())
    new_tokens = sum(1 for tok, is_new in runs if is_new and not tok.isspace())
    draw.text((72, 92), f"LENGTH {total_tokens}", font=FONT_SMALL, fill=MUTED)
    draw.text((214, 92), "•", font=FONT_SMALL, fill=ACCENT_3)
    draw.text((236, 92), f"NEW {new_tokens}", font=FONT_SMALL, fill=MUTED)

    # Text box
    text_box = (TEXT_BOX_MARGIN_X, 114, WIDTH - TEXT_BOX_MARGIN_X, HEIGHT - 36)
    img = draw_shadowed_panel(
        img,
        text_box,
        radius=10,
        fill=hex_to_rgba(PANEL_2, 255),
        outline=hex_to_rgba(OUTLINE, 255),
        width=1,
        shadow_alpha=6,
    )
    draw = ImageDraw.Draw(img)
    draw.line((92, 142, WIDTH - 92, 142), fill=hex_to_rgba("#ece5dc", 255), width=1)

    max_width = TEXT_X1 - TEXT_X0
    lines = wrap_runs(draw, runs, FONT_TEXT, max_width)
    line_height = 38
    y = 158

    for line in lines:
        x = TEXT_X0
        for token, is_new, token_advance in line:
            if token.isspace():
                x += token_advance
                continue

            if is_new:
                left, top, right, bottom = draw.textbbox((x, y), token, font=FONT_TEXT)
                draw.text((x, y), token, font=FONT_TEXT, fill=NEW_TEXT)
                underline_y = bottom + 5
                draw.line((left, underline_y, right, underline_y), fill=HIGHLIGHT_BORDER, width=4)
                draw.line((left, underline_y + 3, right, underline_y + 3), fill=hex_to_rgba("#a92e2e", 255), width=2)
            else:
                draw.text((x, y), token, font=FONT_TEXT, fill=TEXT)
            x += token_advance

        y += line_height
        if y > TEXT_Y1:
            break

    return img.convert("RGB")


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    steps_text = STEPS_FILE.read_text(encoding="utf-8")
    steps = parse_steps(steps_text)

    frames = []
    durations = []

    for i, (step_no, curr_text) in enumerate(steps):
        prev_text = steps[i - 1][1] if i > 0 else ""
        frame = render_frame(i, step_no, curr_text, prev_text, steps[-1][0])
        frames.append(frame)
        durations.append(380)

    # Hold on the final frame a bit longer
    for _ in range(4):
        frames.append(frames[-1].copy())
        durations.append(900)

    frames[0].save(
        OUTPUT_GIF,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=False,
    )

    print(f"Saved to: {OUTPUT_GIF}")


if __name__ == "__main__":
    main()