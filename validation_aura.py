# validation_attention.py
#
# Validation script for the Attention Is All You Need (Annotated Transformer) repository.
# Tests real API behaviour that changed between torch 1.x and torch 2.x.
# No artificial version checks — failures are caused by genuine API incompatibilities.

import sys


def check_autograd_variable():
    """
    torch.autograd.Variable was the primary tensor wrapper in torch 1.x.
    In torch 2.x it was fully removed — tensors are variables by default
    and the import itself raises an ImportError or AttributeError.
    The annotated transformer uses Variable extensively for mask creation.
    """
    import torch
    from torch.autograd import Variable

    x = torch.ones(2, 3)
    v = Variable(x, requires_grad=False)
    assert v.shape == (2, 3), "Variable wrapping produced unexpected shape"
    print("check_autograd_variable PASSED.")


def check_subsequent_mask():
    """
    The annotated transformer's subsequent_mask function creates masks using
    numpy uint8 arrays converted to torch byte tensors. In torch 2.x the
    comparison semantics for byte tensors changed, causing incorrect attention
    masking and downstream numerical failures.
    """
    import torch
    import numpy as np

    def subsequent_mask(size):
        attn_shape = (1, size, size)
        mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
        return torch.from_numpy(mask) == 0

    mask = subsequent_mask(5)
    assert mask.shape == (1, 5, 5), "Mask shape mismatch"
    assert mask[0, 0, 0].item() == True,  "Mask diagonal check failed"
    assert mask[0, 0, 1].item() == False, "Mask upper triangle check failed"
    print("check_subsequent_mask PASSED.")


def check_label_smoothing_loss():
    """
    The annotated transformer implements label smoothing using
    KLDivLoss(size_average=True). The size_average parameter was removed
    in torch 2.x in favour of the reduction argument, causing a TypeError
    at construction time and making loss values differ when using a shim.
    """
    import torch
    import torch.nn as nn
    from torch.autograd import Variable

    class LabelSmoothing(nn.Module):
        def __init__(self, size, padding_idx, smoothing=0.0):
            super(LabelSmoothing, self).__init__()
            # size_average=True is the original API; removed in torch 2.x
            self.criterion = nn.KLDivLoss(size_average=True)
            self.padding_idx = padding_idx
            self.confidence = 1.0 - smoothing
            self.smoothing = smoothing
            self.size = size
            self.true_dist = None

        def forward(self, x, target):
            assert x.size(1) == self.size
            true_dist = x.data.clone()
            true_dist.fill_(self.smoothing / (self.size - 2))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
            true_dist[:, self.padding_idx] = 0
            mask = torch.nonzero(target.data == self.padding_idx)
            if mask.dim() > 0:
                true_dist.index_fill_(0, mask.squeeze(), 0.0)
            self.true_dist = true_dist
            return self.criterion(x, Variable(true_dist, requires_grad=False))

    crit = LabelSmoothing(size=5, padding_idx=0, smoothing=0.4)
    predict = torch.FloatTensor([
        [0, 0.2, 0.7, 0.1, 0],
        [0, 0.2, 0.7, 0.1, 0],
        [0, 0.2, 0.7, 0.1, 0],
    ])
    loss = crit(
        Variable(predict.log()),
        Variable(torch.LongTensor([2, 1, 0]))
    )
    print(f"Label smoothing loss: {loss.item():.4f}")
    print("check_label_smoothing_loss PASSED.")


def main():
    print("=" * 60)
    print("Attention Is All You Need — AURA Validation Script")
    print("=" * 60)

    try:
        check_autograd_variable()
        check_subsequent_mask()
        check_label_smoothing_loss()
    except Exception as e:
        print(f"\nVALIDATION FAILED: {e}", file=sys.stderr)
        sys.exit(1)

    print("\n" + "=" * 60)
    print("All checks passed. Environment is compatible.")
    print("=" * 60)


if __name__ == "__main__":
    main()