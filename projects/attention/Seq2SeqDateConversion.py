import pandas as pd
import numpy as np
import calendar
import logging
from optparse import OptionParser
from matplotlib import pyplot as plt

from keras.models import Model
from keras.layers import Input, Dense, Embedding, SimpleRNN, SimpleRNNCell, RNN
from keras.callbacks import TensorBoard, History


logging.basicConfig(format='%(asctime)-15s %(levelname)s : %(message)s', level=logging.DEBUG)

def create_dataset(N, append_eod_token=True):
    dateformats = {
        "DD/MM/YY": lambda x: x.strftime("%d/%m/%Y"),
        "YY/MM/DD": lambda x: x.strftime("%Y/%m/%d"),
        "Month D, Yr": lambda x: calendar.month_name[x.month] + " " + str(x.day) + ", " + str(x.year)[-2:],
        "D Month, Yr": lambda x: str(x.day) + " " + calendar.month_name[x.month] + ", " + str(x.year)[-2:],
        "Yr, Month D": lambda x: str(x.year)[-2:] + ", " + calendar.month_name[x.month] + " " + str(x.day),
        "Mon-DD-YYYY": lambda x: "-".join([calendar.month_abbr[x.month], leadzero(str(x.day)), str(x.year)]),
        "DD-Mon-YYYY": lambda x: "-".join([leadzero(str(x.day)), calendar.month_abbr[x.month], str(x.year)]),
        "Mon DD, YYYY": lambda x: calendar.month_abbr[x.month] + " " + leadzero(str(x.day)) + ", " + str(x.year),
        "Mon D, YYYY": lambda x: calendar.month_abbr[x.month] + " " + str(x.day) + ", " + str(x.year),
        "DD Mon, YYYY": lambda x: leadzero(str(x.day)) + " " + calendar.month_abbr[x.month] + ", " + str(x.year),
        "D Mon, YYYY": lambda x: str(x.day) + " " + calendar.month_abbr[x.month] + ", " + str(x.year),
        "YYYY, Mon DD": lambda x: str(x.year) + ", " + calendar.month_abbr[x.month] + " " + leadzero(str(x.day)),
        "YYYY, Mon D": lambda x: str(x.year) + ", " + calendar.month_abbr[x.month] + " " + str(x.day)
    }
    dffmt = pd.DataFrame({"Format":list(dateformats.keys()), "Func":list(dateformats.values())})
    dfdats = pd.DataFrame({"Output": [pd.Timestamp(year=2019, month=m, day=d) for y,m,d in
                                    zip(np.random.randint(low=2018, high=2020, size=N),
                                        np.random.randint(low=1, high=13, size=N),
                                        np.random.randint(low=1, high=28, size=N))],
                           "Func" : dffmt.Func.iloc[np.random.randint(low=0,high=len(dffmt),size=N)].values
                          })
    dfdats.loc[:,"Input"] = dfdats.apply(lambda x : x["Func"](x["Output"]), axis=1)
    dfdats.loc[:,"Output"] = dfdats.Output.apply(lambda x : x.strftime("%Y-%m-%d"))
    if append_eod_token:
        dfdats.loc[:,"Output"] = dfdats.Output.values + "\t"
    return dfdats.loc[:,["Output", "Input"]]


def leadzero(s):
    if int(s) < 10:
        return "0"+s
    return s


def create_vocabulary(sequences):
    vocab = []
    for sequence in sequences:
        for element in sequence:
            if element not in vocab:
                vocab.append(element)
    index = dict([(char, i) for i, char in enumerate(sorted(vocab))])
    return vocab, index


def string_to_vocabindices(s, vocab_index):
    res = []
    for ss in s:
        res.append(vocab_index[ss])
    return res


def prepend_zeros(l, max_length):
    return (max_length - len(l)) * [0] + l


def append_zeros(l, max_length):
    return l + (max_length - len(l)) * [0]


def onehot_encoding_of_text(x_string, vocab_index, max_sequence_length):
    x_indices = string_to_vocabindices(x_string, vocab_index)
    x_onehot = np.zeros((max_sequence_length, len(vocab_index)))
    for i, word_idx in enumerate(reversed(x_indices)):
        x_onehot[-i-1, word_idx] = 1
    return x_onehot


def onehot_decoding_to_text(onehot_array, vocab_index):
    d = dict(zip(vocab_index.values(), vocab_index.keys()))
    return ''.join([d[i] for i in np.where(onehot_array)[1]])


def decode_vocab_indices(text_indices, vocab_index):
    tmp = dict([(v, k) for k,v in vocab_index.items()])
    return "".join([tmp[idx] for idx in text_indices])


def encoder_decoder_trainingdata(dataset, input_vocab, input_index, output_vocab, output_index):
    print(f"Input vocab size {len(input_vocab)} | Output vocab size {len(output_vocab)}")
    max_input_length = dataset.Input.apply(lambda x: len(x)).max()
    max_output_length = dataset.Output.apply(lambda x: len(x)).max()
    print(f"Maximum input sequence length: {max_input_length} | Maximum output sequence length: {max_output_length}")

    encoder_inputs = []
    decoder_inputs = []
    decoder_targets = []
    for input_text, output_text in zip(dataset.Input, dataset.Output):
        encoder_inputs.append(
            prepend_zeros(string_to_vocabindices(input_text, input_index), max_input_length))
        decoder_inputs.append(
            prepend_zeros(string_to_vocabindices("\t" + output_text.replace("\t", ""), output_index),
                          max_output_length))
        decoder_targets.append(
            # prepend_zeros(string_to_vocabindices(output_text, output_index), max_output_length))
            onehot_encoding_of_text(output_text, output_index, max_output_length))
    encoder_inputs = np.array(encoder_inputs)
    decoder_inputs = np.array(decoder_inputs)
    decoder_targets = np.array(decoder_targets)
    return encoder_inputs, decoder_inputs, decoder_targets


def seq2seq_model(encoder_input_maxlength, decoder_input_maxlength, size_input_vocab, size_output_vocab):
    embedding_dim = 12
    rnn_dim = 32

    encoder_input = Input(shape=(encoder_input_maxlength,),
                          name="encoder_input")  # (16,)-dimensional tensor (max_input_length = 16)
    encoder_embedding = Embedding(input_dim=size_input_vocab, output_dim=embedding_dim, name="encoder_embedding")(
        encoder_input)  # (12,16) dim embedding tensor
    _, encoder_rnn_state = SimpleRNN(rnn_dim, activation='relu', return_sequences=False, return_state=True,
                                     name="encoder_rnn")(encoder_embedding)

    decoder_input = Input(shape=(decoder_input_maxlength,),
                          name="decoder_input")  # gets the shifted ground truth e.g. ('\t2019-01-01') which is 11-dimensional
    decoder_embedding = Embedding(input_dim=size_output_vocab, output_dim=embedding_dim, name="decoder_embedding")(decoder_input)  # (12,11) dim embedding
    decoder_rnn, _ = RNN(SimpleRNNCell(rnn_dim, activation='relu'), return_sequences=True, return_state=True,
                         name="decoder_rnn")(decoder_embedding, initial_state=encoder_rnn_state)
    decoder_predictions = Dense(size_output_vocab, activation='softmax', name="decoder_predictions")(decoder_rnn)

    model = Model([encoder_input, decoder_input], decoder_predictions)
    return model


def train(num_training_samples, num_epochs):
    dataset = create_dataset(num_training_samples)
    input_vocab, input_index = create_vocabulary(dataset.Input.values)
    output_vocab, output_index = create_vocabulary(dataset.Output.values)

    encoder_inputs, decoder_inputs, decoder_targets = encoder_decoder_trainingdata(
        dataset=dataset,
        input_vocab=input_vocab,
        input_index=input_index,
        output_vocab=output_vocab,
        output_index=output_index
    )
    model = seq2seq_model(
        encoder_input_maxlength=encoder_inputs.shape[1],
        decoder_input_maxlength=decoder_inputs.shape[1],
        size_input_vocab=len(input_vocab),
        size_output_vocab=len(output_vocab)
    )
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    callbacks = [
        TensorBoard(log_dir='./logs', update_freq='batch')
    ]
    training_history = model.fit(
        x=[encoder_inputs, decoder_inputs],
        y=decoder_targets,
        epochs=num_epochs,
        validation_split=0.2,
        callbacks=callbacks
    )
    return model, training_history, input_index, output_index


def inference_encoder_decoder(model):
    encoder = Model(model.get_layer(name="encoder_input").input, model.get_layer(name="encoder_rnn").output[-1])

    rnn_dim = model.get_layer(name="decoder_rnn").input_shape[-1][-1]
    decoder_input = model.get_layer(name="decoder_input").input
    decoder_initstate = Input(shape=(rnn_dim,), name="decoder_initstate")
    decoder_embedding = model.get_layer(name="decoder_embedding")(decoder_input)
    decoder_rnn, decoder_state = model.get_layer(name="decoder_rnn")(decoder_embedding, initial_state=decoder_initstate)
    decoder_predictions = model.get_layer(name="decoder_predictions")(decoder_rnn)
    decoder = Model([decoder_input, decoder_initstate], [decoder_predictions, decoder_state])
    return encoder, decoder


def infer(xin, encoder, decoder, input_index, output_index):
    max_input_length = encoder.get_layer(name="encoder_input").input_shape[-1]
    max_output_length = decoder.get_layer(name="decoder_input").input_shape[-1]
    assert len(xin) <= max_input_length, f"Input size cannot be larger than {max_input_length}"
    enc_in = np.array(prepend_zeros(string_to_vocabindices(xin, input_index), max_input_length))
    dec_in = np.array(append_zeros(string_to_vocabindices("\t", output_index), max_output_length))

    enc_in = np.reshape(enc_in, newshape=(1, len(enc_in)))
    dec_in = np.reshape(dec_in, newshape=(1, len(dec_in)))

    enc_last_state = encoder.predict(enc_in)
    finished = False
    inferred_sample = ''
    while not finished:
        decoder_output = decoder.predict(x=[dec_in, enc_last_state])
        decoder_onehot_pred = decoder_output[0][0, len(inferred_sample), :]
        decoder_str_pred = decode_vocab_indices([np.argmax(decoder_onehot_pred)], output_index)
        dec_in = np.array(
            append_zeros(string_to_vocabindices("\t" + decoder_str_pred, output_index), max_output_length))
        dec_in = np.reshape(dec_in, newshape=(1, len(dec_in)))
        inferred_sample += decoder_str_pred
        if (decoder_str_pred == "\t") or (len(inferred_sample) == max_output_length):
            finished = True
    return inferred_sample


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option(
        "-n",
        "--num-train",
        dest="num_training_samples",
        help="number of training samples",
        default=2000,
        type="int"
    )
    parser.add_option(
        "-t",
        "--num-test",
        dest="num_test_samples",
        help="number of test samples",
        default=100,
        type="int"
    )
    parser.add_option(
        "-e",
        "--num-epochs",
        dest="num_epochs",
        help="number of epochs",
        default=10,
        type="int"
    )
    (options, args) = parser.parse_args()

    num_training_samples = options.num_training_samples
    num_test_samples = options.num_test_samples
    num_epochs = options.num_epochs

    logging.info(f"Training model on {num_training_samples} training samples for {num_epochs} epochs")

    model, training_history, input_index, output_index = train(num_training_samples, num_epochs)
    encoder, decoder = inference_encoder_decoder(model)

    logging.info(f"Testing model on {num_test_samples} test samples")
    testset = create_dataset(num_test_samples, append_eod_token=False)
    inferred_samples = []
    for i, x in enumerate(testset.Input):
        if i % 100 == 0:
            logging.info(f"Testing progress: {i} / {num_test_samples} done")
        inferred_samples.append(infer(x, encoder, decoder, input_index, output_index))
    testset.loc[:,"PredOutput"] = inferred_samples
    testaccuracy = len(testset.loc[testset.Output == testset.PredOutput]) / num_test_samples
    logging.info(f"Test accuracy: {testaccuracy}")
    #return training_history, testset



