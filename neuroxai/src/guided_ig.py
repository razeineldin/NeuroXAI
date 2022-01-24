# Guided Integrated Gradient
from .integrated_grads import compute_grads
from ..utils.process import get_last_layer

import numpy as np
import tensorflow as tf
import math

# A very small number for comparing floating point values.
EPSILON = 1e-9

def l1_distance(x1, x2):
    """Returns L1 distance between two points."""
    return np.abs(x1 - x2).sum()

def translate_x_to_alpha(x, x_input, x_baseline):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(x_input - x_baseline != 0,
                                        (x - x_baseline) / (x_input - x_baseline), np.nan)

def translate_alpha_to_x(alpha, x_input, x_baseline):
    assert 0 <= alpha <= 1.0
    return x_baseline + (x_input - x_baseline) * alpha

def get_guided_integrated_grads(model, io_imgs, class_id, LAYER_NAME=None, MODALITY="FLAIR", XAI_MODE="classification",
                                                                        STEPS=5, FRAC=0.5, MAX_DIST=1.0):
#                                                                        STEPS=200, FRAC=0.25, MAX_DIST=0.02):
    modality_dict = {"FLAIR": 0, "T1": 1, "T1CE": 2, "T2": 3}

    baseline = np.zeros_like(io_imgs)
    x_input = np.asarray(io_imgs, dtype=np.float64)
    x_baseline = np.asarray(baseline, dtype=np.float64)
    x = x_baseline.copy()
    l1_total = l1_distance(x_input, x_baseline)
    attr = np.zeros_like(x_input, dtype=np.float64)

    # If the input is equal to the baseline then the attribution is zero.
    total_diff = x_input - x_baseline
    if np.abs(total_diff).sum() == 0:
        return attr

    # Iterate through every step.
    for step in range(STEPS):
        #print("Step:({}/{})".format(step+1, STEPS))
        # Calculate gradients and make a copy.
        grad_actual = compute_grads(model, x, class_id, LAYER_NAME, MODALITY, XAI_MODE)
        grad = grad_actual.copy()
        # Calculate current step alpha and the ranges of allowed values for this
        # step.
        alpha = (step + 1.0) / STEPS
        alpha_min = max(alpha - MAX_DIST, 0.0)
        alpha_max = min(alpha + MAX_DIST, 1.0)
        x_min = translate_alpha_to_x(alpha_min, x_input, x_baseline)
        x_max = translate_alpha_to_x(alpha_max, x_input, x_baseline)
        # The goal of every step is to reduce L1 distance to the input.
        # `l1_target` is the desired L1 distance after completion of this step.
        l1_target = l1_total * (1 - (step + 1) / STEPS)

        # Iterate until the desired L1 distance has been reached.
        gamma = np.inf
        while gamma > 1.0:
            x_old = x.copy()
            x_alpha = translate_x_to_alpha(x, x_input, x_baseline)
            x_alpha[np.isnan(x_alpha)] = alpha_max
            # All features that fell behind the [alpha_min, alpha_max] interval in
            # terms of alpha, should be assigned the x_min values.
            x[x_alpha < alpha_min] = x_min[x_alpha < alpha_min]

            # Calculate current L1 distance from the input.
            l1_current = l1_distance(x, x_input)
            # If the current L1 distance is close enough to the desired one then
            # update the attribution and proceed to the next step.
            if math.isclose(l1_target, l1_current, rel_tol=EPSILON, abs_tol=EPSILON):
                attr += (x - x_old) * grad_actual
                break

            # Features that reached `x_max` should not be included in the selection.
            # Assign very high gradients to them so they are excluded.
            grad[x == x_max] = np.inf

            # Select features with the lowest absolute gradient.
            threshold = np.quantile(np.abs(grad), FRAC, interpolation='lower')
            s = np.logical_and(np.abs(grad) <= threshold, grad != np.inf)

            # Find by how much the L1 distance can be reduced by changing only the
            # selected features.
            l1_s = (np.abs(x - x_max) * s).sum()

            # Calculate ratio `gamma` that show how much the selected features should
            # be changed toward `x_max` to close the gap between current L1 and target
            # L1.
            if l1_s > 0:
                gamma = (l1_current - l1_target) / l1_s
            else:
                gamma = np.inf

            if gamma > 1.0:
                # Gamma higher than 1.0 means that changing selected features is not
                # enough to close the gap. Therefore change them as much as possible to
                # stay in the valid range.
                x[s] = x_max[s]
            else:
                assert gamma > 0, gamma
                x[s] = translate_alpha_to_x(gamma, x_max, x)[s]
            # Update attribution to reflect changes in `x`.
            attr += (x - x_old) * grad_actual
    return attr[0]
