import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
import pickle as pl
import os
import sklearn
from itertools import product

plt.style.use('seaborn')
FONT_SIZE = 16

LINESTYLES = ['-', '-.', ':']
MARKERS = ["+", "o", "*", "s", ".", "1", "2", "3", "4"]
MARKER_SIZE_COLORS = [(4, "white"), (8, "red"), (12, "yellow"), (16, "lightgreen")]

def change_fontsize(ax, fs):
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(fs)

sample_data = {
    'label1' : np.random.randint(2, 10, size=(10)),
    'label2' : np.random.randint(2, 10, size=(10)).astype(np.float)/ 100,
}
sample_data = {key: (val, None) for key, val in sample_data.iteritems()}


# TODO add usages
def autolabel(rects, ax):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        if type(height) in [np.int, np.uint, int, np.int8, np.int16, np.int32, np.int64]:
            ax.text(rect.get_x() + rect.get_width() / 2., 1.05 * height,
                    '%d' % int(height),
                    ha='center', va='bottom', fontsize=FONT_SIZE)
        else:
            ax.text(rect.get_x() + rect.get_width() / 2., 1.05 * height,
                    '{:.02f}'.format(height),
                    ha='center', va='bottom', fontsize=FONT_SIZE)


# TODO move this code to public repository
def barplot(fig=None, ax=None, data_dict=sample_data, height_text=True, bar_spacing=1.0):
    '''
    :param fig:
    :param ax:
    :param data_dict: { label1 : (heights1, yerr1), label2: (heights2, yerr2)}
    :return:
    '''

    for key, val in data_dict.iteritems():
        assert type(val) == tuple

    x = data_dict.pop('x', None)
    if x is None:
        x = np.arange(len(data_dict.values()[0][0])) * float(bar_spacing)

    num_bars = float(len(data_dict.values()))
    width = 1/(num_bars + 1) * bar_spacing

    i = 0
    for label, data in data_dict.iteritems():
        height, yerr = data
        if yerr is None:
            yerr = np.zeros_like(height)
        r = ax.bar(x + i*width - 0.5 + width/2,
                   yerr = yerr, height=height, width=width,
                   label=label, align='center')
        if height_text:
            autolabel(r, ax)
        i += 1
    ax.legend(fontsize=FONT_SIZE)
    ax.set_xticks(x)
    change_fontsize(ax, FONT_SIZE)
    return fig, ax


# TODO move this code to public repository
class FigSaver():
    def __init__(self, save_dir='.', plt_savefig_format='png', plt_savefig_kwargs={'dpi' : 250}):
        self._save_dir = save_dir
        self._cur_fig_num = 0
        self._plt_savefig_kwargs=plt_savefig_kwargs
        self._plt_savefig_format = plt_savefig_format

    def save_fig(self, fig, fname=''):
        '''

        :param fig:
        :param fname: file name without extension. will automatically get an ordering number beforehand
        :return:
        '''
        fname = '{:02d}_{}'.format(self._cur_fig_num, fname)
        pkl_fname = os.path.join(self._save_dir, '{}.{}'.format(fname, 'pickle'))
        plt_savefig_fname = os.path.join(self._save_dir, '{}.{}'.format(fname, self._plt_savefig_format))
        pl.dump(fig, file(pkl_fname, 'w')) # this might only work for python 2.7
        plt.savefig(plt_savefig_fname, **self._plt_savefig_kwargs)
        return self

    def next(self):
        '''
        increment figure counter. This method returns self so that you can chain methods goteher like
        fig_saver.next().next()
        :return:
        '''
        self._cur_fig_num += 1
        return self

    def set(self, value):
        # TODO magic function?
        self._cur_fig_num = value
        return self

    def reset(self):
        self._cur_fig_num = 0
        return self


def plot_cm(ax, cm, classes=np.arange(2), normalize=False):
    '''

    :param ax: ax object to plot confusion matrix in
    :param cm:
    :param classes:
    :param normalize:
    :return:
    '''
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.set_title('Normalized Confusion Matrix', fontsize=FONT_SIZE)
    else:
        ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Oranges)
        ax.set_title('Absolute Confusion Matrix', fontsize=FONT_SIZE)
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, fontsize=FONT_SIZE)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes, rotation=90, fontsize=FONT_SIZE)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], fmt),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black", fontsize=FONT_SIZE)

    ax.set_ylabel('True label', fontsize=FONT_SIZE)
    ax.set_xlabel('Predicted label', fontsize=FONT_SIZE)
    change_fontsize(ax, FONT_SIZE)
    return ax



if __name__ == '__main__':
    fig, ax = plt.subplots()
    fig, ax = barplot(fig, ax, bar_spacing=5)
    fig.show()

    f_saver = FigSaver()
    f_saver.save_fig(fig, 'test')