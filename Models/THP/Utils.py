import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer.Models import get_non_pad_mask


def softplus(x, beta):
    """Numerically stable softplus with a strictly positive trainable scale."""
    positive_beta = F.softplus(beta) + 1e-6
    return F.softplus(positive_beta * x) / positive_beta


def _interval_values(time, non_pad_mask):
    """Return valid interval lengths and their normalized duration."""
    interval_mask = non_pad_mask[:, 1:].bool()
    raw_diff = time[:, 1:] - time[:, :-1]
    diff_time = torch.where(interval_mask, raw_diff, torch.zeros_like(raw_diff))
    if torch.any(diff_time < 0):
        raise ValueError("Event times must be non-decreasing")
    scaled_duration = diff_time / (time[:, :-1] + 1.0)
    return diff_time, scaled_duration, interval_mask


def compute_integral_trapezoid(model, data, time, non_pad_mask):
    """Integrate the total marked intensity with the trapezoidal rule."""
    diff_time, scaled_duration, _ = _interval_values(time, non_pad_mask)
    history_hid = model.linear(data[:, :-1, :])
    start_lambda = softplus(history_hid, model.beta).sum(dim=-1)
    end_lambda = softplus(
        history_hid + model.alpha * scaled_duration.unsqueeze(-1),
        model.beta,
    ).sum(dim=-1)
    return 0.5 * (start_lambda + end_lambda) * diff_time


def compute_integral_unbiased(
        model, data, time, non_pad_mask, type_mask=None, num_samples=20,
        deterministic=False):
    """Monte Carlo integration of the total intensity over all event types.

    ``type_mask`` remains in the signature for compatibility with older callers
    but is intentionally unused: a marked TPP survival term must integrate the
    sum of every type intensity, not only the type observed at interval end.
    """
    del type_mask
    num_samples = int(num_samples)
    if num_samples < 1:
        raise ValueError("num_samples must be positive")

    diff_time, scaled_duration, _ = _interval_values(time, non_pad_mask)
    history_hid = model.linear(data[:, :-1, :])
    if deterministic:
        fractions = (
            torch.arange(num_samples, device=data.device, dtype=data.dtype) + 0.5
        ) / num_samples
        fractions = fractions.view(1, 1, num_samples)
    else:
        fractions = torch.rand(
            *diff_time.size(), num_samples, device=data.device, dtype=data.dtype
        )
    sampled_offsets = scaled_duration.unsqueeze(-1) * fractions
    sampled_lambda = softplus(
        history_hid.unsqueeze(-1)
        + model.alpha.view(1, 1, -1, 1) * sampled_offsets.unsqueeze(2),
        model.beta,
    )
    total_lambda = sampled_lambda.sum(dim=2).mean(dim=-1)
    return total_lambda * diff_time


def log_likelihood(
        model, data, time, types, loss_mask=None, integral_method="trapezoid",
        num_samples=20, deterministic=False):
    """Causal marked-TPP log-likelihood for events after the first one."""
    non_pad_mask = get_non_pad_mask(types).squeeze(2)
    diff_time, scaled_duration, interval_mask = _interval_values(
        time, non_pad_mask
    )

    # h_i represents history through event i and parameterizes the intensity
    # over (t_i, t_{i+1}], including the likelihood of event i+1.
    history_hid = model.linear(data[:, :-1, :])
    event_all_lambda = softplus(
        history_hid + model.alpha * scaled_duration.unsqueeze(-1),
        model.beta,
    )
    truth = (types[:, 1:] - 1).clamp(min=0)
    event_lambda = event_all_lambda.gather(
        dim=-1, index=truth.unsqueeze(-1)
    ).squeeze(-1)
    event_ll = torch.log(event_lambda.clamp_min(1e-9))

    score_mask = interval_mask
    if loss_mask is not None:
        score_mask = score_mask & loss_mask[:, 1:].bool()
    event_ll = (event_ll * score_mask.float()).sum(dim=-1)

    if integral_method == "trapezoid":
        non_event_ll = compute_integral_trapezoid(
            model, data, time, non_pad_mask
        )
    elif integral_method == "mc":
        non_event_ll = compute_integral_unbiased(
            model,
            data,
            time,
            non_pad_mask,
            num_samples=num_samples,
            deterministic=deterministic,
        )
    else:
        raise ValueError("Unknown integral method: {}".format(integral_method))
    non_event_ll = (non_event_ll * score_mask.float()).sum(dim=-1)
    return event_ll, non_event_ll


def type_loss(
        prediction, types, loss_func, loss_mask=None,
        target_predicate_ids=None):
    """ Event prediction loss, cross entropy or label smoothing. """

    # convert [1,2,3] based types to [0,1,2]; also convert padding events to -1
    truth = types[:, 1:] - 1
    prediction = prediction[:, :-1, :]
    score_mask = truth.ne(-1)
    if loss_mask is not None:
        score_mask = score_mask & loss_mask[:, 1:].bool()

    target_stats = {}
    if target_predicate_ids:
        safe_truth = truth.clamp(min=0)
        truth_predicate = torch.div(safe_truth, 2, rounding_mode="floor")
        target_mask = torch.zeros_like(score_mask)
        for predicate_id in target_predicate_ids:
            target_mask = target_mask | truth_predicate.eq(int(predicate_id))
        score_mask = score_mask & target_mask

        pair_start = truth_predicate * 2
        pair_indices = torch.stack((pair_start, pair_start + 1), dim=-1)
        pair_indices = pair_indices.clamp(max=prediction.size(-1) - 1)
        conditional_prediction = prediction.gather(-1, pair_indices)
        truth_state = safe_truth.remainder(2)
        pred_state = conditional_prediction.argmax(dim=-1)
        correct = pred_state.eq(truth_state) & score_mask
        correct_num = correct.sum()

        flat_prediction = conditional_prediction.reshape(-1, 2)
        flat_truth = truth_state.reshape(-1)
        if isinstance(loss_func, LabelSmoothingLoss):
            smooth_truth = F.one_hot(flat_truth, num_classes=2).float()
            smooth_truth = (
                smooth_truth * (1 - loss_func.eps) + loss_func.eps / 2
            )
            loss = -(
                smooth_truth * F.log_softmax(flat_prediction, dim=-1)
            ).sum(dim=-1)
        else:
            loss = F.cross_entropy(
                flat_prediction, flat_truth, reduction="none"
            )
        loss = loss.view_as(truth_state)
        for predicate_id in target_predicate_ids:
            predicate_mask = score_mask & truth_predicate.eq(int(predicate_id))
            target_stats[int(predicate_id)] = (
                int((correct & predicate_mask).sum().item()),
                int(predicate_mask.sum().item()),
            )
    else:
        pred_type = prediction.argmax(dim=-1)
        correct_num = ((pred_type == truth) & score_mask).sum()
        if isinstance(loss_func, LabelSmoothingLoss):
            loss = loss_func(prediction, truth.clone())
        else:
            loss = loss_func(prediction.transpose(1, 2), truth)

    loss = loss * score_mask.float()
    loss = torch.sum(loss)
    return loss, correct_num, score_mask.sum(), target_stats


def time_loss(prediction, event_time, loss_mask=None):
    """ Time prediction loss. """

    prediction.squeeze_(-1)

    true = event_time[:, 1:] - event_time[:, :-1]
    prediction = prediction[:, :-1]

    # event time gap prediction
    diff = prediction - true
    if loss_mask is None:
        score_mask = torch.ones_like(true)
    else:
        score_mask = loss_mask[:, 1:].float()
    se = torch.sum(diff * diff * score_mask)
    return se


class LabelSmoothingLoss(nn.Module):
    """
    With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    """

    def __init__(self, label_smoothing, tgt_vocab_size, ignore_index=-100):
        assert 0.0 < label_smoothing <= 1.0
        super(LabelSmoothingLoss, self).__init__()

        self.eps = label_smoothing
        self.num_classes = tgt_vocab_size
        self.ignore_index = ignore_index

    def forward(self, output, target):
        """
        output (FloatTensor): (batch_size) x n_classes
        target (LongTensor): batch_size
        """

        non_pad_mask = target.ne(self.ignore_index).float()

        target[target.eq(self.ignore_index)] = 0
        one_hot = F.one_hot(target, num_classes=self.num_classes).float()
        # This distribution sums to one: eps / K is assigned to every class,
        # with the remaining mass placed on the target class.
        one_hot = one_hot * (1 - self.eps) + self.eps / self.num_classes

        log_prb = F.log_softmax(output, dim=-1)
        loss = -(one_hot * log_prb).sum(dim=-1)
        loss = loss * non_pad_mask
        return loss
