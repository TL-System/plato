import traceback

__all__ = ['format_in_box', 'format_exception_traceback']


def format_exception_traceback(exception: BaseException = None):
    if exception:
        format_frames = traceback.format_exception(type(exception), exception, exception.__traceback__)
    else:
        format_frames = traceback.format_exc()
    format_frames = ''.join(format_frames)
    return format_frames


def format_in_box(lines, horizon='*', vertical='|', corner='*', indent=0, margin=1):
    lines = list(lines)
    max_width = max([len(line) for line in lines])
    lines = [''] * margin + lines + [''] * margin
    result = [' ' * indent + corner + horizon * (max_width + margin * 2) + corner]
    result.extend([' ' * indent + vertical + ' ' * margin + line + ' ' * (margin + max_width - len(line)) + vertical
                   for line in lines])
    result.append(' ' * indent + corner + horizon * (max_width + margin * 2) + corner)
    return '\n'.join(result)
