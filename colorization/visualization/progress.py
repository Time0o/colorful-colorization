def display_progress(i, i_end, msg='processing image'):
    fmt = "{} {}/{}"

    ljust = len(fmt.format(msg, i_end, i_end))

    end = '\n' if i == i_end - 1 else ''

    print('\r' + fmt.format(msg, i + 1, i_end).ljust(ljust), end=end)
