import logging

from bilstm import BiLSTM


class EncoderNetwork:

    def __init__(self, model, input_dim, output_dim, rnn_dropout_rate=0.1):
        self.logger = logging.getLogger(__name__)
        self.model = model
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.rnn_dropout_rate = rnn_dropout_rate

        self.encoder_rnn = BiLSTM(
            in_dim=self.input_dim,
            out_dim=self.output_dim,
            model=self.model,
            dropout_rate=self.rnn_dropout_rate
        )

        self.logger.debug(
            "Initialized EncoderNetwork (%s, %s) with a dropout of %s." % (
                self.input_dim, self.output_dim, self.rnn_dropout_rate
            ))

    def __call__(self, input_sequence):
        return self.encoder_rnn.get_sequence_vector(
            sequence=input_sequence,
            dropout=True
        )
