import torch
from prototypical_loss import prototypical_loss


def test_forKownInput_lossFunctionOutputShouldBeExpectedOne():
    # arrange
    model_output = torch.load("./model_output.pt")
    targets = torch.load("./targets.pt")

    # act
    loss, acc = prototypical_loss(model_output, target=targets, n_support=5)

    # assert
    expected_loss = 1.8843656778335571
    expected_acc = 0.47999998927116394
    assert abs(expected_loss - loss.item()) < 1e-9 # on compare des floats alors on veut éviter des fp errors
    assert abs(expected_acc - acc.item()) < 1e-9


if __name__ == '__main__':
    test_forKownInput_lossFunctionOutputShouldBeExpectedOne()
