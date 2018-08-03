import time
from itertools import product
import pandas as pd
import numpy as np
import seaborn as sns
from tqdm import tqdm

#TODO add multiprocess support

def timerfunc(func):
    """
    A timer decorator
    https://www.blog.pythonlibrary.org/2016/05/24/python-101-an-intro-to-benchmarking-your-code/
    """
    def function_timer(*args, **kwargs):
        """
        A nested function for timing other functions
        """
        start = time.time()
        value = func(*args, **kwargs)
        end = time.time()
        runtime = end - start
        msg = "The runtime for {func} took {time} seconds to complete"
        print(msg.format(func=func.__name__,
                         time=runtime))
        return value
    return function_timer


def time_execution(function, *args):
    tic = time.time()
    o = function(*args)
    toc = time.time()
    return o, toc-tic

def benchmark_function(param_grid, function, param_names, n_times=100):
    '''
    performs benchmark of function over parameters in parametergrid
    :param param_grid: list of lists of parameters
    :param function:
    :return:
    '''
    param_names += ['runtime', 'num_execution']
    opt_tuples = list(product(*param_grid))
    data = { k : [] for k in param_names }
    for opt_tuple in tqdm(opt_tuples):
            for i in range(n_times):
                args = opt_tuple
                for k, v in zip(param_names, opt_tuple):
                    data[k].append(v)
                _, runtime = time_execution(function, *args)
                data['num_execution'].append(i)
                data['runtime'].append(runtime)
    df = pd.DataFrame(data)
    return df


if __name__ == '__main__':
    def func(N, pow):
        import time
        time.sleep(float(pow) / 10000 * N)
        return np.arange(N) ** pow


    param_grid = [
        [10, 100, 1000],
        [2, 3, 4],
    ]
    param_names = ['N', 'powers']

    df = benchmark_function(param_grid, func, param_names, n_times=10)
    df.to_csv('benchmark.csv')

    g = sns.FacetGrid(df, col='N', row='powers', sharex=False)
    g = g.map(sns.distplot, 'runtime', hist=True, kde=False)
    g.fig.savefig('example_benchmark.png')

    df2 = df.groupby(['N', 'powers']).mean().reset_index().drop('num_execution', 1)
    from pylab import *
    plt.style.use('seaborn')

    fig, ax = plt.subplots()
    df2.groupby('powers').plot(x='N', y='runtime', subplots=True, ax=ax, legend=True)
    ax.legend()
    plt.savefig('example_benchmark_2.png')



    x='N'
    y='runtime'
    from itertools import *
    compare_params=['powers']

    def roundrobin(*iterables):
        "roundrobin('ABC', 'D', 'EF') --> A D E B F C"
        # Recipe credited to George Sakkis
        num_active = len(iterables)
        nexts = cycle(iter(it).__next__ for it in iterables)
        while num_active:
            try:
                for next in nexts:
                    yield next()
            except StopIteration:
                # Remove the iterator we just exhausted from the cycle.
                num_active -= 1
                nexts = cycle(islice(nexts, num_active))

    with plt.style.context('seaborn'):
        fig, ax = plt.subplots()
        df2 = df[compare_params+[x, y]]
        for name, group in df2.groupby(compare_params):
            datum = group.reset_index().groupby(x).mean().reset_index().drop('index', axis=1)
            label_template = '{}={}' * len(compare_params)
            if not hasattr(name, '__iter__'): #check if list
                name = [name]
            label_vals = list(roundrobin(compare_params, name))
            label = label_template.format(*label_vals)
            datum.plot(x, y, ax=ax, label=label)