import logging

from bilstm import BiLSTM


class EncoderNetwork:

    def __init__(self, model, input_dim, output_dim, rnn_dropout_rate=0.1):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model = model
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.rnn_dropout_rate = rnn_dropout_rate

        self.encoder_rnn = BiLSTM(
            in_dim=self.input_dim,
            out_dim=self.output_dim / 2,  # Divide by two since it is a BiLSTM
            model=self.model,
            dropout_rate=self.rnn_dropout_rate
        )

        self.logger.debug(
            "Initialized EncoderNetwork (%s, %s) with a dropout of %s." % (
                self.input_dim, self.output_dim, self.rnn_dropout_rate
            ))

    def get_encoded_sequence(self, input_sequence):
        """
        Returns the encoded vectors for every item in the input sequence.
        :param input_sequence:
        :return:
        """
        return self.encoder_rnn.get_sequence_vectors(
            sequence=input_sequence,
            dropout=True
        )

    def __call__(self, input_sequence):
        """
        Encoded the input sequence into the concatenation of the last states
        of the BiRNN.
        :param input_sequence:
        :return:
        """
        return self.encoder_rnn.get_sequence_vector(
            sequence=input_sequence,
            dropout=True
        )
