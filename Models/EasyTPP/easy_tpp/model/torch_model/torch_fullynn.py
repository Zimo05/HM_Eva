import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import grad

from easy_tpp.model.torch_model.torch_basemodel import TorchBaseModel


class CumulHazardFunctionNetwork(nn.Module):
    """Cumulative Hazard Function Network
    ref: https://github.com/wassname/torch-neuralpointprocess
    """

    def __init__(self, model_config):
        super(CumulHazardFunctionNetwork, self).__init__()
        self.hidden_size = model_config.hidden_size
        self.num_mlp_layers = model_config.model_specs['num_mlp_layers']
        self.num_event_types = model_config.num_event_types
        self.factorized_marks = model_config.model_specs.get("factorized_marks", False)
        self.proper_marked_intensities = model_config.model_specs.get(
            "proper_marked_intensities", True
        )
        self.log_time = model_config.model_specs.get("log_time", False)
        self.time_epsilon = float(
            model_config.model_specs.get("time_epsilon", 1e-8)
        )
        self.register_buffer(
            "log_time_mean",
            torch.tensor(
                float(model_config.model_specs.get("log_time_mean", 0.0)),
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "log_time_std",
            torch.tensor(
                max(float(model_config.model_specs.get("log_time_std", 1.0)), 1e-6),
                dtype=torch.float32,
            ),
        )
        output_size = 1 if self.factorized_marks else self.num_event_types

        # The paper constrains the elapsed-time path, but not the history-to-first-
        # hidden-layer path, to be positive. Keep these projections separate so
        # history can have both excitatory and inhibitory effects.
        self.layer_time = nn.Linear(
            in_features=1, out_features=self.hidden_size, bias=False
        )
        self.layer_history = nn.Linear(
            in_features=self.hidden_size, out_features=self.hidden_size
        )

        self.hidden_layers = nn.ModuleList(
            [nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size) for _ in
             range(self.num_mlp_layers - 1)])

        self.layer_output = nn.Linear(
            in_features=self.hidden_size, out_features=output_size
        )

        self.params_eps = torch.finfo(torch.float32).eps  # ensure positiveness of parameters

        self.init_weights_positive()

    def positive_weight_parameters(self):
        yield self.layer_time.weight
        for layer in self.hidden_layers:
            yield layer.weight
        yield self.layer_output.weight

    def init_weights_positive(self):
        with torch.no_grad():
            for parameter in self.positive_weight_parameters():
                parameter.copy_(parameter.abs().clamp(min=self.params_eps))

    def normalize_time(self, time_delta_seqs):
        if not self.log_time:
            return time_delta_seqs
        log_time = torch.log(time_delta_seqs.clamp_min(0.0) + self.time_epsilon)
        return (log_time - self.log_time_mean) / self.log_time_std

    def forward(self, hidden_states, time_delta_seqs):
        # Project the constrained weights back to the positive orthant after an
        # optimizer update, matching the original FullyNN implementation.
        self.init_weights_positive()

        time_delta_seqs.requires_grad_(True)

        normalized_time = self.normalize_time(time_delta_seqs).unsqueeze(dim=-1)
        out = torch.tanh(
            self.layer_time(normalized_time) + self.layer_history(hidden_states)
        )
        for layer in self.hidden_layers:
            out = torch.tanh(layer(out))

        integral_lambda = F.softplus(self.layer_output(out))

        if self.proper_marked_intensities or self.factorized_marks:
            derivative_integral_lambdas = []
            for i in range(integral_lambda.shape[-1]):  # iterate over marks
                derivative_integral_lambdas.append(grad(
                    integral_lambda[..., i].sum(),
                    time_delta_seqs,
                    # Training needs the derivative graph so gradients can
                    # flow from intensity back into the cumulative-hazard
                    # parameters. Evaluation only needs the time derivative
                    # value and should not retain that expensive graph.
                    create_graph=self.training, retain_graph=True)[0])
            derivative_integral_lambda = torch.stack(derivative_integral_lambdas, dim=-1)  # TODO: Check that it is okay to iterate over marks like this
        else:
            derivative_integral_lambda = grad( 
                integral_lambda.sum(),
                time_delta_seqs,
                create_graph=self.training, retain_graph=True)[0]
            derivative_integral_lambda = derivative_integral_lambda.unsqueeze(-1).expand(*derivative_integral_lambda.shape, self.num_event_types) / self.num_event_types

        return integral_lambda, derivative_integral_lambda


class FullyNN(TorchBaseModel):
    """Torch implementation of
        Fully Neural Network based Model for General Temporal Point Processes, NeurIPS 2019.
        https://arxiv.org/abs/1905.09690

        ref: https://github.com/KanghoonYoon/torch-neuralpointprocess/blob/master/module.py;
            https://github.com/wassname/torch-neuralpointprocess
    """

    def __init__(self, model_config):
        """Initialize the model

        Args:
            model_config (EasyTPP.ModelConfig): config of model specs.
        """
        super(FullyNN, self).__init__(model_config)

        self.factorized_marks = model_config.model_specs.get(
            "factorized_marks", False
        )
        self.target_predicate_ids = tuple(
            int(value) for value in model_config.model_specs.get(
                "target_predicate_ids", []
            )
        )
        self.rnn_type = model_config.rnn_type
        self.rnn_list = [nn.LSTM, nn.RNN, nn.GRU]
        self.n_layers = model_config.num_layers
        self.dropout_rate = model_config.dropout_rate
        for sub_rnn_class in self.rnn_list:
            if sub_rnn_class.__name__ == self.rnn_type:
                self.layer_rnn = sub_rnn_class(input_size=1 + self.hidden_size,
                                               hidden_size=self.hidden_size,
                                               num_layers=self.n_layers,
                                               batch_first=True,
                                               dropout=self.dropout_rate)

        self.layer_intensity = CumulHazardFunctionNetwork(model_config)
        self.mark_linear = (
            nn.Linear(self.hidden_size, self.num_event_types)
            if self.factorized_marks else None
        )
        self.last_loglike_components = None

    def forward(self, time_seqs, time_delta_seqs, type_seqs):
        """Call the model

        Args:
            time_seqs (tensor): [batch_size, seq_len], timestamp seqs.
            time_delta_seqs (tensor): [batch_size, seq_len], inter-event time seqs.
            type_seqs (tensor): [batch_size, seq_len], event type seqs.

        Returns:
            tensor: hidden states at event times.
        """
        # [batch_size, seq_len, hidden_size]
        type_embedding = self.layer_type_emb(type_seqs)

        normalized_time = self.layer_intensity.normalize_time(time_delta_seqs)
        rnn_input = torch.cat((type_embedding, normalized_time.unsqueeze(-1)), dim=-1)

        # [batch_size, seq_len, hidden_size]
        # states right after the event
        hidden_states, _ = self.layer_rnn(rnn_input)

        return hidden_states

    def loglike_loss(self, batch):
        """Compute the loglike loss.

        Args:
            batch (tuple, list): batch input.

        Returns:
            list: loglike loss, num events.
        """
        # [batch_size, seq_len]
        time_seqs, time_delta_seqs, type_seqs, batch_non_pad_mask, _ = batch[:5]

        # [batch_size, seq_len, hidden_size]
        hidden_states = self.forward(
            time_seqs[:, :-1],
            time_delta_seqs[:, :-1],
            type_seqs[:, :-1],
        )
        # [batch_size, seq_len, num_event_types]
        integral_lambda, derivative_integral_lambda = self.layer_intensity(hidden_states, time_delta_seqs[:, 1:])

        event_mask = torch.logical_and(
            batch_non_pad_mask[:, 1:], type_seqs[:, 1:] != self.pad_token_id
        )

        derivative_integral_lambda += self.eps

        if self.factorized_marks:
            total_event_ll = derivative_integral_lambda.squeeze(-1).log()
            mark_log_probs = F.log_softmax(self.mark_linear(hidden_states), dim=-1)
            mark_ll = -F.nll_loss(
                mark_log_probs.permute(0, 2, 1),
                target=type_seqs[:, 1:],
                ignore_index=self.pad_token_id,
                reduction="none",
            )
            time_ll = (
                total_event_ll - integral_lambda.squeeze(-1)
            ) * event_mask
        else:
            log_marked_event_lambdas = derivative_integral_lambda.log()
            marked_event_ll = -F.nll_loss(
                log_marked_event_lambdas.permute(0, 2, 1),
                target=type_seqs[:, 1:],
                ignore_index=self.pad_token_id,
                reduction="none",
            )
            total_event_ll = derivative_integral_lambda.sum(-1).log()
            mark_ll = (marked_event_ll - total_event_ll) * event_mask
            time_ll = (
                total_event_ll - integral_lambda.sum(-1)
            ) * event_mask

        mark_ll = mark_ll * event_mask
        joint_ll = time_ll + mark_ll
        num_events = int(event_mask.sum().item())
        loss = -joint_ll.sum()
        self.last_loglike_components = {
            "time_loglike_sum": float(time_ll.detach().sum().item()),
            "mark_loglike_sum": float(mark_ll.detach().sum().item()),
            "joint_loglike_sum": float(joint_ll.detach().sum().item()),
        }

        return loss, num_events

    def predict_one_step_at_every_event(self, batch):
        standard_batch = tuple(batch[:5])
        predicted_dtimes, predicted_types = super().predict_one_step_at_every_event(
            standard_batch
        )
        if not self.factorized_marks:
            return predicted_dtimes, predicted_types

        time_seqs, time_delta_seqs, type_seqs, _, _ = standard_batch
        hidden_states = self.forward(
            time_seqs[:, :-1], time_delta_seqs[:, :-1], type_seqs[:, :-1]
        )
        mark_logits = self.mark_linear(hidden_states)
        if self.target_predicate_ids:
            # MIMIC evaluation conditions on which predicate is queried and
            # predicts only its Boolean state from the corresponding pair.
            target_marks = type_seqs[:, 1:]
            predicate_ids = torch.div(
                target_marks.clamp(max=self.num_event_types - 1),
                2,
                rounding_mode="floor",
            )
            state_zero = 2 * predicate_ids
            state_one = state_zero + 1
            zero_logits = torch.gather(mark_logits, -1, state_zero.unsqueeze(-1))
            one_logits = torch.gather(mark_logits, -1, state_one.unsqueeze(-1))
            predicted_state = (one_logits > zero_logits).long().squeeze(-1)
            predicted_types = 2 * predicate_ids + predicted_state
        else:
            predicted_types = torch.argmax(mark_logits, dim=-1)
        return predicted_dtimes, predicted_types

    def compute_intensities_at_sample_times(self,
                                            time_seqs,
                                            time_delta_seqs,
                                            type_seqs,
                                            sample_dtimes,
                                            **kwargs):
        """Compute hidden states at sampled times.

        Args:
            time_seqs (tensor): [batch_size, seq_len], times seqs.
            time_delta_seqs (tensor): [batch_size, seq_len], time delta seqs.
            type_seqs (tensor): [batch_size, seq_len], event type seqs.
            sample_dtimes (tensor): [batch_size, seq_len, num_samples], sampled inter-event timestamps.

        Returns:
            tensor: [batch_size, seq_len, num_samples, num_event_types], intensity at all sampled times.
        """

        compute_last_step_only = kwargs.get('compute_last_step_only', False)

        # [batch_size, seq_len, hidden_size]
        hidden_states = self.forward(
            time_seqs=time_seqs,
            time_delta_seqs=time_delta_seqs,
            type_seqs=type_seqs,
        )

        num_samples = sample_dtimes.size()[-1]
        batch_size, seq_len, hidden_size = hidden_states.shape

        hidden_states_ = hidden_states[..., None, :].expand(batch_size, seq_len, num_samples, hidden_size)
        _, derivative_integral_lambda = self.layer_intensity.forward(
            hidden_states=hidden_states_,
            time_delta_seqs=sample_dtimes,
        )

        if compute_last_step_only:
            lambdas = derivative_integral_lambda[:, -1:, :, :]
        else:
            # [batch_size, seq_len, num_samples, num_event_types]
            lambdas = derivative_integral_lambda
        return lambdas
