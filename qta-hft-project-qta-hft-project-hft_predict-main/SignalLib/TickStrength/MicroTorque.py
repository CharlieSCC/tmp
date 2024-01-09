from SignalLib import mas, run_test
import numpy as np

def func(d, *ns):
    c2d = np.nan_to_num(np.log1p(d['micp']) - np.log1p(d['midp'])) * 10000
    # c2d = np.nan_to_num(d['micp']  / d['midp'], posinf=0, neginf=0) * 10000
    return mas(c2d, ns)


if __name__ == '__main__':

    res = run_test(func, 3, 5, 10, 30, 50, w=2)