import numpy as np
import torch
import torch.utils.data

from transformer import Constants


class EventData(torch.utils.data.Dataset):
    """ Event stream dataset. """

    def __init__(self, data):
        """
        Data should be a list of event streams; each event stream is a list of dictionaries;
        each dictionary contains: time_since_start, time_since_last_event, type_event
        """
        self.time = [[elem['time_since_start'] for elem in inst] for inst in data]
        self.time_gap = [[elem['time_since_last_event'] for elem in inst] for inst in data]
        # plus 1 since there could be event type 0, but we use 0 as padding
        self.event_type = [[elem['type_event'] + 1 for elem in inst] for inst in data]
        self.event_loss_mask = [[
            bool(elem.get('event_loss_mask', elem.get('loss_mask', True)))
            for elem in inst
        ] for inst in data]
        self.type_loss_mask = [[
            bool(elem.get('type_loss_mask', elem.get('loss_mask', True)))
            for elem in inst
        ] for inst in data]
        self.time_loss_mask = [[
            bool(elem.get('time_loss_mask', elem.get('loss_mask', True)))
            for elem in inst
        ] for inst in data]

        self.length = len(data)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """ Each returned element is a list, which represents an event stream """
        return (
            self.time[idx],
            self.time_gap[idx],
            self.event_type[idx],
            self.event_loss_mask[idx],
            self.type_loss_mask[idx],
            self.time_loss_mask[idx],
        )


def pad_time(insts):
    """ Pad the instance to the max seq length in batch. """

    max_len = max(len(inst) for inst in insts)

    batch_seq = np.array([
        inst + [Constants.PAD] * (max_len - len(inst))
        for inst in insts])

    return torch.tensor(batch_seq, dtype=torch.float32)


def pad_type(insts):
    """ Pad the instance to the max seq length in batch. """

    max_len = max(len(inst) for inst in insts)

    batch_seq = np.array([
        inst + [Constants.PAD] * (max_len - len(inst))
        for inst in insts])

    return torch.tensor(batch_seq, dtype=torch.long)


def pad_loss_mask(insts):
    """Pad per-event scoring masks; padded positions never contribute loss."""

    max_len = max(len(inst) for inst in insts)
    batch_seq = np.array([
        inst + [False] * (max_len - len(inst))
        for inst in insts
    ])
    return torch.tensor(batch_seq, dtype=torch.bool)


def collate_fn(insts):
    """ Collate function, as required by PyTorch. """

    (
        time, time_gap, event_type, event_loss_mask, type_loss_mask,
        time_loss_mask,
    ) = list(zip(*insts))
    time = pad_time(time)
    time_gap = pad_time(time_gap)
    event_type = pad_type(event_type)
    event_loss_mask = pad_loss_mask(event_loss_mask)
    type_loss_mask = pad_loss_mask(type_loss_mask)
    time_loss_mask = pad_loss_mask(time_loss_mask)
    return (
        time, time_gap, event_type, event_loss_mask, type_loss_mask,
        time_loss_mask,
    )


def get_dataloader(data, batch_size, shuffle=True, num_workers=0):
    """ Prepare dataloader. """

    ds = EventData(data)
    dl = torch.utils.data.DataLoader(
        ds,
        num_workers=int(num_workers),
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=shuffle
    )
    return dl
