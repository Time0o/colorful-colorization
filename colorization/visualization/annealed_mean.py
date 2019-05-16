from ..util.image import images_in_directory, imread, predict_color
from .plot import subplots


def annealed_mean_demo(model, image_dir, ts=None, verbose=False):
    # temperature parameters
    t_orig = model.network.decode_q.T

    if ts is None:
        ts = [1, .77, .58, .38, .29, .14, 0]

    assert len(ts) % 2 == 1

    # run predictions
    image_paths = images_in_directory(image_dir)

    _, axes = subplots(len(image_paths), len(ts), use_gridspec=True)

    for c, t in enumerate(ts):
        if verbose:
            print("running prediction for T = {}".format(t))

        model.network.decode_q.T = t

        for r, path in enumerate(image_paths):
            axes[r, c].imshow(predict_color(model, imread(path)))

    # reset temperature parameter
    model.network.decode_q.T = t_orig

    # add titles
    for i, (t, ax) in enumerate(zip(ts, axes[0, :])):
        title = {
            0: "Mean",
            len(ts) // 2: "Annealed Mean",
            len(ts) - 1: "Mode"
        }.get(i, '')

        if title:
            title += '\n'

        if t == 0:
            title += "$T\\rightarrow{}$"
        else:
            title += "$T={}$"

        ax.set_title(title.format(t))
