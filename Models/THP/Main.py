import argparse
import gzip
import json
import numpy as np
import pickle
import time
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim

import transformer.Constants as Constants
import Utils

from preprocess.Dataset import get_dataloader
from transformer.Models import Transformer
from tqdm import tqdm


def _merge_target_stats(total, update):
    for predicate_id, (correct, count) in update.items():
        previous_correct, previous_count = total.get(predicate_id, (0, 0))
        total[predicate_id] = (
            previous_correct + int(correct), previous_count + int(count)
        )


def _target_accuracies(stats, predicate_ids):
    result = {}
    for predicate_id in predicate_ids:
        correct, count = stats.get(int(predicate_id), (0, 0))
        result[int(predicate_id)] = (
            float(correct) / count if count else float('nan')
        )
    return result


def _format_target_accuracies(values):
    if not values:
        return ''
    return ', target accuracy: ' + ', '.join(
        '{}={:.5f}'.format(predicate_id, accuracy)
        for predicate_id, accuracy in values.items()
    )


def prepare_dataloader(opt):
    """ Load data and prepare dataloader. """

    def load_data(name, dict_name):
        with Path(name).open('rb') as f:
            payload = pickle.load(f, encoding='latin-1')
            num_types = payload['dim_process']
            metadata = payload.get('metadata', {})
            return payload[dict_name], int(num_types), metadata

    print('[Info] Loading train data...')
    train_data, num_types, train_metadata = load_data(
        Path(opt.data) / 'train.pkl', 'train'
    )
    print('[Info] Loading dev data...')
    dev_data, dev_num_types, dev_metadata = load_data(
        Path(opt.data) / 'dev.pkl', 'dev'
    )
    print('[Info] Loading test data...')
    test_data, test_num_types, test_metadata = load_data(
        Path(opt.data) / 'test.pkl', 'test'
    )
    if len({num_types, dev_num_types, test_num_types}) != 1:
        raise ValueError('dim_process differs across train/dev/test')
    for metadata in (dev_metadata, test_metadata):
        if metadata.get('target_predicate_ids') != train_metadata.get(
                'target_predicate_ids'):
            raise ValueError('target predicate metadata differs across splits')

    trainloader = get_dataloader(
        train_data, opt.batch_size, shuffle=True, num_workers=opt.num_workers
    )
    devloader = get_dataloader(
        dev_data, opt.batch_size, shuffle=False, num_workers=opt.num_workers
    )
    testloader = get_dataloader(
        test_data, opt.batch_size, shuffle=False, num_workers=opt.num_workers
    )
    return trainloader, devloader, testloader, num_types, train_metadata


def train_epoch(model, training_data, optimizer, pred_loss_func, opt):
    """ Epoch operation in training phase. """

    model.train()

    total_event_ll = 0  # cumulative event log-likelihood
    total_time_se = 0  # cumulative time prediction squared-error
    total_event_rate = 0  # cumulative number of correct prediction
    total_num_event = 0  # number of total events
    total_num_pred = 0  # number of predictions
    total_num_time = 0  # number of real-time-gap predictions
    total_target_stats = {}
    for batch in tqdm(training_data, mininterval=2,
                      desc='  - (Training)   ', leave=False):
        """ prepare data """
        (
            event_time, time_gap, event_type, event_loss_mask,
            type_loss_mask, time_loss_mask,
        ) = map(lambda x: x.to(opt.device), batch)
        del time_gap

        """ forward """
        optimizer.zero_grad()

        enc_out, prediction = model(event_type, event_time)

        """ backward """
        # negative log-likelihood
        event_ll, non_event_ll = Utils.log_likelihood(
            model,
            enc_out,
            event_time,
            event_type,
            event_loss_mask,
            integral_method=opt.integral_method,
            num_samples=opt.mc_samples,
            deterministic=False,
        )
        event_loss_sum = -torch.sum(event_ll - non_event_ll)

        # type prediction
        pred_loss_sum, pred_num_event, pred_count, target_stats = Utils.type_loss(
            prediction[0],
            event_type,
            pred_loss_func,
            type_loss_mask,
            target_predicate_ids=opt.target_predicate_ids,
        )

        # time prediction
        se_sum = Utils.time_loss(prediction[1], event_time, time_loss_mask)

        interval_valid = event_type[:, 1:].ne(Constants.PAD)
        event_count = (
            interval_valid & event_loss_mask[:, 1:].bool()
        ).sum()
        time_count = (
            interval_valid & time_loss_mask[:, 1:].bool()
        ).sum()
        loss = event_loss_sum.new_zeros(())
        if event_count.item():
            loss = loss + opt.event_loss_weight * event_loss_sum / event_count
        if pred_count.item():
            loss = loss + opt.type_loss_weight * pred_loss_sum / pred_count
        if time_count.item():
            loss = loss + opt.time_loss_weight * se_sum / time_count
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite training loss')
        loss.backward()
        if opt.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)

        """ update parameters """
        optimizer.step()

        """ note keeping """
        total_event_ll += -event_loss_sum.item()
        total_time_se += se_sum.item()
        total_event_rate += pred_num_event.item()
        total_num_event += event_count.item()
        total_num_pred += pred_count.item()
        total_num_time += time_count.item()
        _merge_target_stats(total_target_stats, target_stats)

    if not total_num_event or not total_num_pred or not total_num_time:
        raise RuntimeError('A training metric has no scored events')
    rmse = float(np.sqrt(total_time_se / total_num_time))
    return (
        total_event_ll / total_num_event,
        total_event_rate / total_num_pred,
        rmse,
        _target_accuracies(total_target_stats, opt.target_predicate_ids),
    )


def eval_epoch(model, validation_data, pred_loss_func, opt,
               prediction_path=None):
    """ Epoch operation in evaluation phase. """

    model.eval()

    total_event_ll = 0  # cumulative event log-likelihood
    total_time_se = 0  # cumulative time prediction squared-error
    total_event_rate = 0  # cumulative number of correct prediction
    total_num_event = 0  # number of total events
    total_num_pred = 0  # number of predictions
    total_num_time = 0  # number of real-time-gap predictions
    total_target_stats = {}
    prediction_rows = []
    sequence_cursor = 0
    with torch.no_grad():
        for batch in tqdm(validation_data, mininterval=2,
                          desc='  - (Validation) ', leave=False):
            """ prepare data """
            (
                event_time, time_gap, event_type, event_loss_mask,
                type_loss_mask, time_loss_mask,
            ) = map(lambda x: x.to(opt.device), batch)
            """ forward """
            enc_out, prediction = model(event_type, event_time)

            """ compute loss """
            event_ll, non_event_ll = Utils.log_likelihood(
                model,
                enc_out,
                event_time,
                event_type,
                event_loss_mask,
                integral_method=opt.integral_method,
                num_samples=opt.mc_samples,
                deterministic=True,
            )
            event_loss_sum = -torch.sum(event_ll - non_event_ll)
            _, pred_num, pred_count, target_stats = Utils.type_loss(
                prediction[0],
                event_type,
                pred_loss_func,
                type_loss_mask,
                target_predicate_ids=opt.target_predicate_ids,
            )
            se_sum = Utils.time_loss(
                prediction[1], event_time, time_loss_mask
            )
            interval_valid = event_type[:, 1:].ne(Constants.PAD)
            event_count = (
                interval_valid & event_loss_mask[:, 1:].bool()
            ).sum()
            time_count = (
                interval_valid & time_loss_mask[:, 1:].bool()
            ).sum()

            """ note keeping """
            total_event_ll += -event_loss_sum.item()
            total_time_se += se_sum.item()
            total_event_rate += pred_num.item()
            total_num_event += event_count.item()
            total_num_pred += pred_count.item()
            total_num_time += time_count.item()
            _merge_target_stats(total_target_stats, target_stats)

            if prediction_path is not None:
                logits = prediction[0][:, :-1, :]
                probabilities = torch.softmax(logits, dim=-1)
                truth = event_type[:, 1:] - 1
                score_mask = truth.ne(-1) & type_loss_mask[:, 1:].bool()
                if opt.target_predicate_ids:
                    predicate = torch.div(
                        truth.clamp(min=0), 2, rounding_mode='floor'
                    )
                    target_mask = torch.zeros_like(score_mask)
                    for predicate_id in opt.target_predicate_ids:
                        target_mask |= predicate.eq(int(predicate_id))
                    score_mask &= target_mask
                    pair_start = predicate * 2
                    pair_indices = torch.stack(
                        (pair_start, pair_start + 1), dim=-1
                    ).clamp(max=logits.size(-1) - 1)
                    pair_logits = logits.gather(-1, pair_indices)
                    predicted_type = pair_start + pair_logits.argmax(dim=-1)
                else:
                    predicted_type = logits.argmax(dim=-1)

                predicted_delta = prediction[1].squeeze(-1)[:, :-1]
                true_delta = time_gap[:, 1:]

                # Keep event-level NLL consistent with the marked-TPP objective.
                non_pad = Utils.get_non_pad_mask(event_type).squeeze(2)
                _, scaled_duration, interval_mask = Utils._interval_values(
                    event_time, non_pad
                )
                history_hid = model.linear(enc_out[:, :-1, :])
                event_all_lambda = Utils.softplus(
                    history_hid
                    + model.alpha * scaled_duration.unsqueeze(-1),
                    model.beta,
                )
                event_lambda = event_all_lambda.gather(
                    -1, truth.clamp(min=0).unsqueeze(-1)
                ).squeeze(-1)
                if opt.integral_method == 'trapezoid':
                    integral = Utils.compute_integral_trapezoid(
                        model, enc_out, event_time, non_pad
                    )
                else:
                    integral = Utils.compute_integral_unbiased(
                        model, enc_out, event_time, non_pad,
                        num_samples=opt.mc_samples, deterministic=True,
                    )
                event_nll = -torch.log(event_lambda.clamp_min(1e-9)) + integral
                score_mask &= interval_mask
                for batch_index in range(event_type.size(0)):
                    event_positions = torch.nonzero(
                        score_mask[batch_index], as_tuple=False
                    ).flatten()
                    for position in event_positions.tolist():
                        prediction_rows.append({
                            'sequence_id': sequence_cursor + batch_index,
                            'event_index': position + 1,
                            'true_type': int(truth[batch_index, position].item()),
                            'predicted_type': int(
                                predicted_type[batch_index, position].item()
                            ),
                            'type_probabilities': probabilities[
                                batch_index, position
                            ].cpu().tolist(),
                            'true_delta_time': float(
                                true_delta[batch_index, position].item()
                            ),
                            'predicted_delta_time': float(
                                predicted_delta[batch_index, position].item()
                            ),
                            'event_nll': float(
                                event_nll[batch_index, position].item()
                            ),
                        })
                sequence_cursor += event_type.size(0)

    if not total_num_event or not total_num_pred or not total_num_time:
        raise RuntimeError('An evaluation metric has no scored events')
    if prediction_path is not None:
        prediction_path = Path(prediction_path)
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(str(prediction_path), 'wt', encoding='utf-8') as handle:
            for row in prediction_rows:
                handle.write(json.dumps(row) + '\n')

    rmse = float(np.sqrt(total_time_se / total_num_time))
    return (
        total_event_ll / total_num_event,
        total_event_rate / total_num_pred,
        rmse,
        _target_accuracies(total_target_stats, opt.target_predicate_ids),
    )


def train(model, training_data, validation_data, optimizer, scheduler, pred_loss_func, opt):
    """ Start training. """

    valid_event_losses = []  # validation log-likelihood
    valid_pred_losses = []  # validation event type prediction accuracy
    valid_rmse = []  # validation event time prediction RMSE
    best_epoch = None
    best_value = None
    for epoch_i in range(opt.epoch):
        epoch = epoch_i + 1
        print('[ Epoch', epoch, ']')

        start = time.time()
        train_event, train_type, train_time, train_targets = train_epoch(
            model, training_data, optimizer, pred_loss_func, opt
        )
        print('  - (Training)    loglikelihood: {ll: 8.5f}, '
              'accuracy: {type: 8.5f}, RMSE: {rmse: 8.5f}, '
              'elapse: {elapse:3.3f} min{targets}'
              .format(ll=train_event, type=train_type, rmse=train_time,
                      elapse=(time.time() - start) / 60,
                      targets=_format_target_accuracies(train_targets)))

        start = time.time()
        valid_event, valid_type, valid_time, valid_targets = eval_epoch(
            model, validation_data, pred_loss_func, opt
        )
        print('  - (Validation)  loglikelihood: {ll: 8.5f}, '
              'accuracy: {type: 8.5f}, RMSE: {rmse: 8.5f}, '
              'elapse: {elapse:3.3f} min{targets}'
              .format(ll=valid_event, type=valid_type, rmse=valid_time,
                      elapse=(time.time() - start) / 60,
                      targets=_format_target_accuracies(valid_targets)))

        valid_event_losses += [valid_event]
        valid_pred_losses += [valid_type]
        valid_rmse += [valid_time]
        print('  - [Info] Maximum ll: {event: 8.5f}, '
              'Maximum accuracy: {pred: 8.5f}, Minimum RMSE: {rmse: 8.5f}'
              .format(event=max(valid_event_losses), pred=max(valid_pred_losses), rmse=min(valid_rmse)))

        selection_values = {
            'll': valid_event,
            'accuracy': valid_type,
            'rmse': valid_time,
        }
        current_value = selection_values[opt.selection_metric]
        improved = (
            best_value is None
            or (opt.selection_metric != 'rmse' and current_value > best_value)
            or (opt.selection_metric == 'rmse' and current_value < best_value)
        )
        if improved:
            best_epoch = epoch
            best_value = current_value
            torch.save({
                'epoch': epoch,
                'selection_metric': opt.selection_metric,
                'selection_value': current_value,
                'dev_loglikelihood': valid_event,
                'dev_accuracy': valid_type,
                'dev_target_accuracy': valid_targets,
                'dev_rmse': valid_time,
                'model_state_dict': model.state_dict(),
            }, opt.save)
            print('  - [Info] Saved best checkpoint: {}'.format(opt.save))

        # Write both curves.  Keeping one row per split makes the CSV directly
        # consumable by the experiment runner and plotting tools.
        with open(opt.log, 'a') as f:
            for split, event, accuracy, rmse, split_targets in (
                    ('train', train_event, train_type, train_time, train_targets),
                    ('validation', valid_event, valid_type, valid_time, valid_targets)):
                target_values = ''.join(
                    ', {:8.5f}'.format(split_targets[predicate_id])
                    for predicate_id in opt.target_predicate_ids
                )
                f.write(
                    '{epoch}, {split}, {ll: 8.5f}, {acc: 8.5f}, '
                    '{rmse: 8.5f}{targets}\n'
                    .format(epoch=epoch, split=split, ll=event,
                            acc=accuracy, rmse=rmse, targets=target_values)
                )

        scheduler.step()

    return best_epoch, best_value


def main():
    """ Main function. """

    parser = argparse.ArgumentParser()

    parser.add_argument('-data', required=True)

    parser.add_argument('-epoch', type=int, default=30)
    parser.add_argument(
        '-batch_size', '-batch', dest='batch_size', type=int, default=16
    )
    parser.add_argument('-num_workers', type=int, default=0)

    parser.add_argument('-d_model', type=int, default=64)
    parser.add_argument('-d_rnn', type=int, default=256)
    parser.add_argument(
        '-d_inner_hid', '-d_inner', dest='d_inner_hid', type=int, default=128
    )
    parser.add_argument('-d_k', type=int, default=16)
    parser.add_argument('-d_v', type=int, default=16)

    parser.add_argument('-n_head', type=int, default=4)
    parser.add_argument('-n_layers', type=int, default=4)

    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-lr', type=float, default=1e-4)
    parser.add_argument('-smooth', type=float, default=0.1)
    parser.add_argument('-seed', type=int, default=2024)
    parser.add_argument('-device', type=str, default=None)
    parser.add_argument(
        '-target_predicate_ids', type=int, nargs='*', default=None,
        help='Predicate IDs scored as conditional Boolean state targets.',
    )
    parser.add_argument(
        '-integral_method', choices=('trapezoid', 'mc'), default='trapezoid'
    )
    parser.add_argument('-mc_samples', type=int, default=20)
    parser.add_argument('-event_loss_weight', type=float, default=1.0)
    parser.add_argument('-type_loss_weight', type=float, default=1.0)
    parser.add_argument('-time_loss_weight', type=float, default=0.1)
    parser.add_argument('-grad_clip', type=float, default=1.0)

    parser.add_argument('-log', type=str, default='log.txt')
    parser.add_argument('-save', type=str, default=None)
    parser.add_argument('-test_log', type=str, default=None)
    parser.add_argument('-prediction_log', type=str, default=None)
    parser.add_argument('-load', type=str, default=None)
    parser.add_argument('-evaluate_only', action='store_true')
    parser.add_argument(
        '-selection_metric', choices=('ll', 'accuracy', 'rmse'), default=None
    )

    opt = parser.parse_args()

    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(opt.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    requested_device = opt.device or (
        'cuda' if torch.cuda.is_available() else 'cpu'
    )
    opt.device = torch.device(requested_device)
    if opt.device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is not available')

    log_path = Path(opt.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if opt.save is None:
        opt.save = str(log_path.with_name(log_path.stem + '_best.pt'))
    if opt.test_log is None:
        opt.test_log = str(log_path.with_name(log_path.stem + '_test.csv'))
    Path(opt.save).parent.mkdir(parents=True, exist_ok=True)
    Path(opt.test_log).parent.mkdir(parents=True, exist_ok=True)

    """ prepare dataloader """
    (
        trainloader, devloader, testloader, num_types, data_metadata,
    ) = prepare_dataloader(opt)
    metadata_targets = tuple(
        int(value)
        for value in data_metadata.get('target_predicate_ids', ())
    )
    if opt.target_predicate_ids is None:
        opt.target_predicate_ids = metadata_targets
    else:
        opt.target_predicate_ids = tuple(opt.target_predicate_ids)
        if metadata_targets and opt.target_predicate_ids != metadata_targets:
            raise ValueError(
                'Requested targets {} do not match the data masks for {}. '
                'Regenerate the adapted data with matching stratify_pred_ids.'
                .format(opt.target_predicate_ids, metadata_targets)
            )
    for predicate_id in opt.target_predicate_ids:
        if predicate_id < 0 or 2 * predicate_id + 1 >= num_types:
            raise ValueError(
                'Target predicate {} is outside dim_process={}'.format(
                    predicate_id, num_types
                )
            )
    if opt.selection_metric is None:
        opt.selection_metric = (
            'accuracy' if opt.target_predicate_ids else 'll'
        )

    # setup the development-metrics log after resolving dataset targets
    target_headers = ''.join(
        ', Target_{}_Accuracy'.format(predicate_id)
        for predicate_id in opt.target_predicate_ids
    )
    with open(opt.log, 'w') as f:
        f.write(
            'Epoch, Split, Log-likelihood, Accuracy, RMSE{}\n'.format(
                target_headers
            )
        )

    print('[Info] parameters: {}'.format(opt))

    """ prepare model """
    model = Transformer(
        num_types=num_types,
        d_model=opt.d_model,
        d_rnn=opt.d_rnn,
        d_inner=opt.d_inner_hid,
        n_layers=opt.n_layers,
        n_head=opt.n_head,
        d_k=opt.d_k,
        d_v=opt.d_v,
        dropout=opt.dropout,
    )
    model.to(opt.device)

    """ optimizer and scheduler """
    optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.parameters()),
                           opt.lr, betas=(0.9, 0.999), eps=1e-05)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 20, gamma=0.5)

    """ prediction loss function, either cross entropy or label smoothing """
    if opt.smooth > 0:
        pred_loss_func = Utils.LabelSmoothingLoss(opt.smooth, num_types, ignore_index=-1)
    else:
        pred_loss_func = nn.CrossEntropyLoss(ignore_index=-1, reduction='none')

    """ number of parameters """
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('[Info] Number of parameters: {}'.format(num_params))

    if opt.load is not None:
        initial = torch.load(opt.load, map_location=opt.device)
        model.load_state_dict(initial.get('model_state_dict', initial))
        print('[Info] Loaded initial checkpoint {}'.format(opt.load))

    if opt.evaluate_only:
        best_epoch, best_value = 0, float('nan')
    else:
        """ train the model """
        best_epoch, best_value = train(
            model, trainloader, devloader, optimizer, scheduler,
            pred_loss_func, opt
        )
        checkpoint = torch.load(opt.save, map_location=opt.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print('[Info] Restored epoch {} selected by {}={:.5f}'.format(
            best_epoch, opt.selection_metric, best_value
        ))

    start = time.time()
    test_event, test_type, test_time, test_targets = eval_epoch(
        model, testloader, pred_loss_func, opt,
        prediction_path=opt.prediction_log,
    )
    print('  - (Final Test)   loglikelihood: {ll: 8.5f}, '
          'accuracy: {type: 8.5f}, RMSE: {rmse: 8.5f}, '
          'elapse: {elapse:3.3f} min{targets}'
          .format(ll=test_event, type=test_type, rmse=test_time,
                  elapse=(time.time() - start) / 60,
                  targets=_format_target_accuracies(test_targets)))
    with open(opt.test_log, 'w') as f:
        target_headers = ''.join(
            ', Target_{}_Accuracy'.format(predicate_id)
            for predicate_id in opt.target_predicate_ids
        )
        target_values = ''.join(
            ', {:8.5f}'.format(test_targets[predicate_id])
            for predicate_id in opt.target_predicate_ids
        )
        f.write(
            'BestEpoch, SelectionMetric, Log-likelihood, Accuracy, RMSE{}\n'
            .format(target_headers)
        )
        f.write(
            '{epoch}, {metric}, {ll: 8.5f}, {acc: 8.5f}, '
            '{rmse: 8.5f}{targets}\n'
            .format(epoch=best_epoch, metric=opt.selection_metric,
                    ll=test_event, acc=test_type, rmse=test_time,
                    targets=target_values)
        )
    print('[Info] Final test metrics saved to {}'.format(opt.test_log))


if __name__ == '__main__':
    main()
