from decimal import Decimal

# --------------------------------------------------------------------
# Functions


def roundSigDig(num: float, dig: int):
    """
    Rounds to significant digits.

    Parameters
    ----------
    num: float
        Number to round
    dig: int
        Digits to round to
    """
    num = float(num)
    rounded = f'{Decimal(f"{num}"):.{dig}g}'
    rounded = str(round(num, 1))
    return rounded


def _boldItMath(s: str, bold: bool):
    if bold:
        s = r'$\bm{'+s+r'}$'
    else:
        s = r'$'+s+r'$'
    return s


def formatEstimatorName(name: str, bold=False):
    """
    Formats the given name in latex.

    Parameters
    ----------
    name: string
        String to be formatted.
    bold: bool, optional
        If the output text should be in bold.
    """
    val = name.split('-MSE', 1)[0]

    if val == 'CKDE_LAKDE-LOO_ES':
        val = r'\mbox{CLAKDE}'
        return _boldItMath(val, bold)
    elif val == 'GHP(L)-CKDE(LL)(ES)':
        val = r'\mbox{GC}'
        return _boldItMath(val, bold)
    elif 'KNN(squared_l2)' in val:
        val = r'\mbox{kNN}_{'+val[-1]+r'}'
        # val = r'\mbox{kNN(} L^2_2 \mbox{)}_{'+val[-1]+r'}'
        return _boldItMath(val, bold)
    elif 'KNN(manhattan)' in val:
        val = r'\mbox{kNN(} L_1 \mbox{)}_{'+val[-1]+r'}'
        return _boldItMath(val, bold)
    elif name == 'bayesClassifierError':
        return 'Bayes Classifier'

    if "_" in val:
        acro, subscript = val.split('_', 1)
        if bold:
            val = '$'+r'\bm{\mbox{'+acro+r'}_{'+subscript+r'}}$'
        else:
            val = '$'+r'\mbox{'+acro+r'}_{'+subscript+r'}$'

    if val == 'NaiveBayesError':
        return _boldItMath(r"\mbox{NB}", bold)
    elif "J2" in val:
        return _boldItMath(r"\mbox{CKDE}", bold)
    elif ("GKDE" in val) & ("silverman" in val):
        return _boldItMath(r"\mbox{GKDE}", bold)
    elif ("KDE" in val) & ("L^2" in val):
        return _boldItMath(r'\mbox{KDE '+acro+r'}_{'+subscript[-1]+r'}', bold)
    else:
        return val
