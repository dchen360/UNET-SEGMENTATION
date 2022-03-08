"""
A class for loading the preprocessed data into a built model and start training and testing.
"""
import sys
# Call the dice_coeff function from dice_score file
from torchvision.utils import save_image
from utils.dice_score import *


class UNET_Train:
    """
    Parameters
    ----------
    model:                The UNET model that is already initialized.
    train_dataloader:     A already defined Python iterable over the training dataset.
    test_dataloader:      A already defined Python iterable over the testing dataset.
    criterion:            The loss function used for training the model.
    optimizer:            The optimizer used for training the model.
    Output
    ------
    dice loss:            The sum of all batches' loss within each epoch when training the model.
    Created by Danni Chen on 09/27/2021.
    Modified by Peter Lias on 10/01/2021
    Modified by Danni Chen on 02/03/2022
    """

    def __init__(self, model, train_dataloader, test_dataloader, optimizer,
                 device=None, writer=None, verbose=False):

        # Auto-select device if none provided.
        if not device:
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # Set if class closed.
        self.closed = False
        # Set if class verbose.
        self.verbose = verbose

        # Assign device
        self.device = device
        # Assign writer
        self.writer = writer

        # train_dataloader attribute
        self.train_dataloader = train_dataloader
        # test_dataloader attribute
        self.test_dataloader = test_dataloader

        self.model = model

        # Auto-assign model to device when setting.
        self.model = self.model.to(self.device)

        # Loss function
        # self.criterion = criterion

        # Optimizer
        self.optimizer = optimizer

    def close(self):
        self.closed = True
        self.writer.close()
        torch.cuda.empty_cache()

    def check_closed(self):
        if self.closed:
            raise RuntimeError("Cannot perform operations on a closed instance.")

    def save(self, fp):
        self.check_closed()
        torch.save(self.model.state_dict(), fp)

    def model_training(self, numOfEpoch):
        # Ensure no resources have been released yet
        self.check_closed()

        pct = 1 / numOfEpoch

        metrics_dict = {
            'epoch': [],
            'train_loss': [],
            'test_loss': [],
        }

        for epoch in range(numOfEpoch):

            # Train the model.
            self.model.train()

            #     ## Epoch Start Time ##
            #     epoch_start = datetime.now()

            # initiate epoch_loss
            epoch_loss = 0.0

            for ith_batch, batch_data in enumerate(self.train_dataloader):
                # obtain the images and labels
                img_batch, labels_batch = batch_data['image'], batch_data['lung_mask']
                img_batch = img_batch.float().to(self.device)
                labels_batch = labels_batch.to(self.device)

                # change labels_batch's dtype from torch.float64 to torch.float32
                labels_batch = labels_batch.type(torch.float32)

                # zero the parameter gradients (necessary because .backward()
                # accumulate gradient for each parameter after each iteration)
                self.optimizer.zero_grad()

                # forward + backward + optimize
                # feed the img_batch (input) into the network
                # Record the predictions and ground-truth

                # print(img_batch.shape)
                # Original: torch.Size([1, 512, 512])
                # Required: batch_size, [channel], length, width
                # use torch.unsqueeze to solve the shape error
                outputs = self.model(torch.unsqueeze(img_batch, 1))

                # use torch.softmax to turn the outputs to a number between 0 and 1
                outputs = torch.softmax(outputs, dim=1)
                # the outputs are two channels that are complementary to each other. choose 1
                outputs = outputs[:, 1]

                # calculate the diceLoss of each batch
                diceLoss = dice_coeff(outputs, labels_batch)

                # backward
                diceLoss.requires_grad_(True)
                diceLoss.backward()
                # perform parameter update based on current gradient (stored in .grad) and update rule of SGD
                self.optimizer.step()

                # the epoch_loss would be the sum of the diceLoss of each batch
                epoch_loss += diceLoss.item()  # .item() extracts diceLoss values as floats
                if self.verbose:
                    sys.stdout.write('\rBatch diceLoss (batch {:d}/{:d}): {:.3f}'.format(
                        ith_batch + 1, len(self.train_dataloader), diceLoss.item()))
                else:
                    sys.stdout.write('\r{:.2f}% complete.'.format(
                        (epoch + 1 + (ith_batch + 1) / len(self.train_dataloader)) * pct
                    ))
                # end of one batch
            # end of all batches in one epoch
            print()

            # train_loss is the averaged diceLoss for each image
            train_loss = epoch_loss / len(self.train_dataloader)
            if self.verbose:
                print('Average epoch {:d} training diceLoss: {:.3f}'.format(epoch + 1, train_loss))

            # Generate testing loss
            test_loss = self.model_testing()

            # Update the metrics_dict.
            list(map(lambda x, y: metrics_dict[x].append(y),
                     ['epoch', 'train_loss', 'test_loss'],
                     [epoch, train_loss, test_loss]))

            if self.writer:
                self.writer.add_scalars('Loss', {'training': train_loss, 'testing': test_loss}, epoch + 1)

            if self.verbose:
                print('Average epoch {:d} training diceLoss: {:.3f} testing diceLoss: {:.3f}'.format(epoch + 1,
                                                                                                     train_loss,
                                                                                                     test_loss))

        print('Done.')

        return metrics_dict

    def model_testing(self):

        # Ensure no resources have been released yet
        self.check_closed()

        # Set the eval mode flag on the model (not important here but good practice)
        self.model.eval()

        # sum of all batch_loss in one epoch
        epoch_loss = 0.0

        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for batch_indx, data in enumerate(self.test_dataloader):  # iterate through the data
                images, labels = data['image'], data['lung_mask']
                images = images.float().to(self.device)
                labels = labels.to(self.device)

                # change labels dtype from torch.float64 to torch.float32
                labels = labels.type(torch.float32)
                # images = images.type(torch.double)
                # calculate outputs by running images through the model
                outputs = self.model(torch.unsqueeze(images, 1))

                # use torch.softmax to turn the outputs to a number between 0 and 1
                outputs = torch.softmax(outputs, dim=1)
                # the outputs are two channels that are complementary to each other. choose 1
                outputs = outputs[:, 1]

                ## Saving images in testing batch ##
                save_image(images, 'images{:d}.png'.format(batch_indx))
                save_image(labels, 'labels{:d}.png'.format(batch_indx))
                save_image(outputs, 'outputs{:d}.png'.format(batch_indx))

                # epoch_loss += dice_coeff(outputs, labels).cpu().detach()
                epoch_loss += dice_coeff(outputs, labels)

            # return the averaged dice_loss for each testing image
            test_loss = epoch_loss / len(self.test_dataloader)
            return test_loss
