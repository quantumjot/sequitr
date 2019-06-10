#!/usr/bin/env python
#-------------------------------------------------------------------------------
# Name:     Sequitr
# Purpose:  Sequitr is a small, lightweight Python library for common image
#           processing tasks in optical microscopy, in particular, single-
#           molecule imaging, super-resolution or time-lapse imaging of cells.
#           Sequitr implements fully convolutional neural networks for image
#           segmentation and classification. Modelling of the PSF is also
#           supported, and the library is designed to integrate with
#           BayesianTracker.
#
# Authors:  Alan R. Lowe (arl) a.lowe@ucl.ac.uk
#
# License:  See LICENSE.md
#
# Created:  23/03/2018
#-------------------------------------------------------------------------------


from itertools import izip
import numpy as np
import matplotlib.pyplot as plt

DEFAULT_LABELS = []

def confusion_matrix(y_true, y_pred, labels=DEFAULT_LABELS, display=False):
    """ confusion_matrix

    Create a confusion matrix to count the pc identity between the human and
    computer labeling of training data.

    Args:
        y_true: the true value (i.e. label)
        y_pred: the predicted value
        labels: the actual labels


    Notes:
        This is only a proxy for the scikit-learn confusion matrix code now!
    """
    from sklearn.metrics import confusion_matrix as sk_confusion_matrix
    conf_matrix = sk_confusion_matrix(y_true, y_pred)

    # display it
    if display:
        plot_confusion_matrix(conf_matrix, labels)

    return conf_matrix



def plot_confusion_matrix(c,
                          labels=DEFAULT_LABELS,
                          scores=True,
                          fmt='%.3f',
                          save=None,
                          normalise=True,
                          epsilon=1e-99):
    """ plot_confusion_matrix

    Plot the confusion matrix as an array, with labels.

    Args:
        c:
        labels:
        scores:
        fmt:
        save:
        normalise:

    Notes:
        Code to add centred scores in each box was modified from here:
        http://stackoverflow.com/questions/25071968/
            heatmap-with-text-in-each-cell-with-matplotlibs-pyplot

        TODO(arl): also plot the absolute counts

    """

    # note the plot maps as x - columns, y - rows
    # tensorflow confusion matrix:
    #   The matrix columns represent the prediction labels and the rows
    #   represent the real labels.
    # x axis -> prediction, y axis -> real

    # transpose the confusion matrix to have real on x, predictions on y
    c = c.T
    c_norm = c / (c.astype(np.float).sum(axis=0, keepdims=True)+epsilon)

    fig, ax = plt.subplots(figsize=(10,6))
    heatmap = ax.pcolor(c_norm, cmap=plt.cm.viridis, vmin=0., vmax=1.)

    counts = np.ravel(c).astype(np.int)

    if scores:
        # plot the stats in the cells
        heatmap.update_scalarmappable()
        for p, color, count, acc in izip(heatmap.get_paths(),
                                          heatmap.get_facecolors(),
                                          counts,
                                          heatmap.get_array()):
            x, y = p.vertices[:-2, :].mean(0)
            if np.all(color[:2] > 0.5):
                color = (0.0, 0.0, 0.0)
            else:
                color = (1.0, 1.0, 1.0)

            txt = "{0:d} \n ({1:.3f})".format(count, acc)
            ax.text(x, y, txt, ha="center", va="center", color=color)

    # put the major ticks at the middle of each cell
    ax.set_xticks(np.arange(c_norm.shape[0])+0.5, minor=False)
    ax.set_yticks(np.arange(c_norm.shape[1])+0.5, minor=False)

    # want a more natural, table-like display
    #ax.invert_yaxis()
    ax.set_xticklabels([l.title() for l in labels], minor=False, rotation='vertical')
    ax.set_yticklabels([l.title() for l in labels], minor=False)

    plt.axis('image')

    plt.xlabel(r'Ground truth')
    plt.ylabel(r'Prediction')
    plt.title('Confusion matrix ({0} examples)'.format(np.sum(confusion_matrix).astype(np.int)))
    plt.colorbar(heatmap).set_label('Normalised class accuracy')

    # Tweak spacing to prevent clipping of tick-labels
    #plt.margins(0.2)
    plt.subplots_adjust(bottom=.25, left=.25)

    # save the figure out if given the filename as a kwarg
    if save is not None:
        if isinstance(save, basestring):
            plt.savefig(save, dpi=144)
            plt.close()
        else:
            raise TypeError('Filename should be a string')
    else:
        plt.show()


if __name__ == "__main__":
    pass
