import logging
import gensim
import smart_open
import csv
from pathlib import Path

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

MAX_FEATURE = 50
HEADER = [i for i in range(MAX_FEATURE)]


def read_corpus(fname, tokens_only=False):
    with smart_open.open(fname, encoding="iso-8859-1") as f:
        for i, line in enumerate(f):
            tokens = gensim.utils.simple_preprocess(line)
            if tokens_only:
                yield tokens
            else:
                # For training data, add tags
                yield gensim.models.doc2vec.TaggedDocument(tokens, [i])


def main():
    base = Path(f"../data/")

    benign_file = base / "example-paragraph" / "benign-paragraph.csv"
    benign_output = base / "example-feature-vector" / "benign-fv.csv"

    anomalous_file = base / "example-paragraph" / "anomaly-paragraph.csv"
    anomalous_output = base / "example-feature-vector" / "anomaly-fv.csv"

    model_out_f = Path("models") / "doc2vec.bin"

    train_corpus = list(read_corpus(benign_file))
    benign_corpus_token = list(read_corpus(benign_file, tokens_only=True))
    anomalous_corpus_token = list(read_corpus(anomalous_file, tokens_only=True))

    # build a model
    # model initialization - I
    model = gensim.models.doc2vec.Doc2Vec(vector_size=MAX_FEATURE, min_count=1, epochs=100, dm=1, workers=4)
    model.build_vocab(train_corpus)

    # model train - II
    model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)

    # save the model - III
    with open(model_out_f, 'wb') as out_f:
        model.save(out_f)

    # infer the feature vector and write to file
    with open(benign_output, "w", newline='') as file1:
        employee_writer = csv.writer(file1, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for t in benign_corpus_token:
            fv = model.infer_vector(t)
            employee_writer.writerow(fv)

    with open(anomalous_output, "w", newline='') as file3:
        employee_writer = csv.writer(file3, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        for t in anomalous_corpus_token:
            fv = model.infer_vector(t)
            employee_writer.writerow(fv)


if __name__ == '__main__':
    main()