import torch
import time
from prototypical_loss import prototypical_loss


def test_forKownInput_lossFunctionOutputShouldBeExpectedOne():
    # arrange
    model_output = torch.load("./model_output.pt")
    targets = torch.load("./targets.pt")

    # act
    begining_time = time.time()
    loss, acc = prototypical_loss(model_output, target=targets, n_support=5)
    ending_time = time.time()
    duration = ending_time - begining_time
    print("Calculation of loss took {:.3f} seconds".format(duration))
    print("Previous duration was about {:.3f} seconds".format(0.011900186538696289))

    # assert
    expected_loss = 1.8843656778335571
    expected_acc = 0.47999998927116394
    assert abs(expected_loss - loss.item()) < 1e-9 # on compare des floats alors on veut éviter des fp errors
    assert abs(expected_acc - acc.item()) < 1e-9


test_forKownInput_lossFunctionOutputShouldBeExpectedOne()
