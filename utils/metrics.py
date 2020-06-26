import numpy as np
from skimage.util import montage
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


def dice_wp(preds, labels):
    """Compute Dice per case between prediction and label"""
    smooth = 1.
    if preds.shape != labels.shape:
        preds = preds.argmax(axis=1)
    dice = np.zeros(preds.shape[0])
    for i in range(preds.shape[0]):
        dice[i] = ((2 * (labels[i] * preds[i]).sum()) + smooth) / (labels[i].sum() + preds[i].sum() + smooth)
    return dice


def update_mxboard_metric_v1(sw, data, global_step, metric_names='r', prefix=None, not_write_to_mxboard=False, best_score=.0):
    """Log metrics to mxboard
    Compared to original update_mxboard_metric method, thes v1 method does not show metrics of individuals on the tensorboard"""

    metric_names = list(metric_names) if not isinstance(metric_names, list) else metric_names
    metrics = {
        'dice': ('dice', dice_wp),
    }
    metric_list = {}
    for metric_name in metric_names:
        fn_name = metrics[metric_name][0]
        fn = metrics[metric_name][1]
        metric = np.asarray([fn(pred, label) for pred, label in zip(data[1], data[2])])
        if not not_write_to_mxboard:
            sw.add_scalar('metrics/%smean_%s' % (prefix, fn_name), metric.mean(), global_step=global_step)
        print(metric)
        metric_list[metric_name] = metric.mean()

    # Show figures if averaged dice higher certain Dice values
    if 'dice' in metric_list.keys():
        if (metric_list['dice'] > .89) and (metric_list['dice'] > best_score):  #or (global_step == 1799):
            case_id = ['Case%02d' % id for id in [0, 1, 14, 15, 26, 27, 39, 40]]
            for i in range(len(case_id)):
                im = np.squeeze(data[0][i])
                pred = data[1][i].argmax(axis=1).squeeze()
                label = data[2][i].squeeze()
                sw.add_image(case_id[i], plot_and_retrieve(im, pred, label), global_step=global_step)

    return metric_list


def plot_and_retrieve(im, pred, label):
    """Create a figure with contour, then save it as an numpy array"""
    fig = Figure(figsize=(8, 8), dpi=100)
    # A canvas must be manually attached to the figure (pyplot would automatically
    # do it).  This is done by instantiating the canvas with the figure as
    # argument.
    canvas = FigureCanvasAgg(fig)
    ax = fig.gca()
    # Threshold the input image
    thr = im[label == 1].max()
    im[im > thr] = thr
    # Plot
    ax.imshow(montage(im), cmap='gray')
    ax.contour(montage(pred), colors='r', linewidths=.3)
    ax.contour(montage(label), colors='y', linewidths=.3)
    # Setting
    ax.axis('off')
    fig.tight_layout(pad=0)
    # To remove the huge white borders
    ax.margins(0)
    canvas.draw()
    s, (width, height) = canvas.print_to_buffer()

    # Option 2a: Convert to a NumPy array.
    X = np.fromstring(s, np.uint8).reshape((height, width, 4))[:, :, :-1]
    return (X / 255).transpose(2, 0, 1)
