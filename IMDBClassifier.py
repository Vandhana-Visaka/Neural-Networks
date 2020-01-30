import numpy as np
import torch
import torch.nn as tnn
import torch.nn.functional as F
import torch.optim as topti
from torchtext import data
from torchtext.vocab import GloVe
from imdb_dataloader import IMDB

class Network(tnn.Module):
    def __init__(self):
        super(Network, self).__init__()
        """
        TODO:
        Create and initialise weights and biases for the layers.
        """
        self.lstm = tnn.LSTM(input_size=50,hidden_size=100,num_layers=2,dropout=0.2,bidirectional=True,batch_first=True)
        self.dropout = tnn.Dropout(0.2)
        self.fc1 = tnn.Linear(in_features=200,out_features=64)
        self.fc2 = tnn.Linear(in_features=64,out_features=1)


    def forward(self, input, length):
        """
        DO NOT MODIFY FUNCTION SIGNATURE
        TODO:
        Create the forward pass through the network.
        """
        #the lstm is created using pack_padded_sequence to pass into the lstm layer
        padded_in = torch.nn.utils.rnn.pack_padded_sequence(input, length, batch_first=True)
        #an extra empty tuple consisting of hx and cx is created for multilayer lstm with bidirectional network
        #hidden_size of 100
        #If (h_0, c_0) is not provided, both h_0 and c_0 default to zero.
        #h_0 of shape (num_layers * num_directions, batch, hidden_size)
        hx = torch.zeros(2*2,input.size(0),100,device="cuda:0")
        #c_0 of shape (num_layers * num_directions, batch, hidden_size)
        cx = torch.zeros(2*2,input.size(0),100,device="cuda:0")
        #the padded sequence 'padded' is passed to the lstm layer with hx and cx
        padded_out,_ = self.lstm(padded_in,(hx,cx))
        #unpacking the output
        out,_ = tnn.utils.rnn.pad_packed_sequence(padded_out)
        #output is passed into two linear layers
        ht = self.fc1(out[1])
        ht = torch.nn.functional.relu(ht)
        #ht = self.dropout(ht)
        ht = self.fc2(ht)
        ht = ht.view(-1)
        return ht

class PreProcessing():
    def pre(x):
        """Called after tokenization"""
        return x

    def post(batch, vocab):
        """Called after numericalization but prior to vectorization"""
        return batch

    text_field = data.Field(lower=True, include_lengths=True, batch_first=True, preprocessing=pre, postprocessing=post)


def lossFunc():
    """
    Define a loss function appropriate for the above networks that will
    add a sigmoid to the output and calculate the binary cross-entropy.
    """
    return torch.nn.BCEWithLogitsLoss()


def main():
    # Use a GPU if available, as it should be faster.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device: " + str(device))

    # Load the training dataset, and create a data loader to generate a batch.
    textField = PreProcessing.text_field
    labelField = data.Field(sequential=False)

    train, dev = IMDB.splits(textField, labelField, train="train", validation="dev")

    textField.build_vocab(train, dev, vectors=GloVe(name="6B", dim=50))
    labelField.build_vocab(train, dev)

    trainLoader, testLoader = data.BucketIterator.splits((train, dev), shuffle=True, batch_size=64,
                                                         sort_key=lambda x: len(x.text), sort_within_batch=True)


    # for i,batch in enumerate(trainLoader):
    #     print(textField.vocab.vectors[batch.text[0]].shape)
    #     print(textField.vocab.vectors[batch.text[1]].shape)
    #     break

    net = Network().to(device)
    criterion =lossFunc()
    optimiser = topti.Adam(net.parameters(), lr=0.001)  # Minimise the loss using the Adam algorithm.

    for epoch in range(15):
        running_loss = 0

        for i, batch in enumerate(trainLoader):
            # Get a batch and potentially send it to GPU memory.
            inputs, length, labels = textField.vocab.vectors[batch.text[0]].to(device), batch.text[1].to(
                device), batch.label.type(torch.FloatTensor).to(device)

            labels -= 1

            # PyTorch calculates gradients by accumulating contributions to them (useful for
            # RNNs).  Hence we must manually set them to zero before calculating them.
            optimiser.zero_grad()

            # Forward pass through the network.
            output = net(inputs, length)

            loss = criterion(output, labels)

            # Calculate gradients.
            loss.backward()

            # Minimise the loss according to the gradient.
            optimiser.step()

            running_loss += loss.item()

            if i % 32 == 31:
                print("Epoch: %2d, Batch: %4d, Loss: %.3f" % (epoch + 1, i + 1, running_loss / 32))
                running_loss = 0

    num_correct = 0

    # Save mode
    torch.save(net.state_dict(), "./model.pth")
    print("Saved model")

    # Evaluate network on the test dataset.  We aren't calculating gradients, so disable autograd to speed up
    # computations and reduce memory usage.
    with torch.no_grad():
        for batch in testLoader:
            # Get a batch and potentially send it to GPU memory.
            inputs, length, labels = textField.vocab.vectors[batch.text[0]].to(device), batch.text[1].to(
                device), batch.label.type(torch.FloatTensor).to(device)

            labels -= 1

            # Get predictions
            outputs = torch.sigmoid(net(inputs, length))
            predicted = torch.round(outputs)

            num_correct += torch.sum(labels == predicted).item()

    accuracy = 100 * num_correct / len(dev)

    print(f"Classification accuracy: {accuracy}")

if __name__ == '__main__':
    main()
