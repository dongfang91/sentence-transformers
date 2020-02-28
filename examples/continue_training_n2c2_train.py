"""
This example loads the pre-trained bert-base-nli-mean-tokens models from the server.
It then fine-tunes this model for some epochs on the STS benchmark dataset.
"""
from torch.utils.data import DataLoader
import math
from sentence_transformers import SentenceTransformer,  SentencesDataset, LoggingHandler, losses
from sentence_transformers.evaluation import LabelAccuracyEvaluator
import CustomDataReader as CustomDataReader
import logging
from datetime import datetime


#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

# Read the dataset
model_name = '/extra/dongfangxu9/sentence_bert/bert-base-nli-mean-tokens/'
train_batch_size = 128
num_epochs = 10
model_save_path = '/xdisk/dongfangxu9/sentence_encoder/n2c2_train/continue_training-bert_nli_n2c2_train-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
n2c2_reader = CustomDataReader.CustomDataReader('datasets/n2c2_pair/a+b+c+d+e_pt/emall_all_sep_30_real/')

# Load a pre-trained sentence transformer model
model = SentenceTransformer(model_name)

# Convert the dataset to a DataLoader ready for training
logging.info("Read n2c2 train dataset")
train_data = SentencesDataset(n2c2_reader.get_examples('train.tsv'), model)
train_dataloader = DataLoader(train_data, shuffle=True, batch_size=train_batch_size)
train_loss = losses.SoftmaxLoss(model=model)


logging.info("Read n2c2 dev dataset")
dev_data = SentencesDataset(examples=n2c2_reader.get_examples('dev.tsv'), model=model)
dev_dataloader = DataLoader(dev_data, shuffle=False, batch_size=train_batch_size)
evaluator = LabelAccuracyEvaluator(dev_dataloader)


# Configure the training. We skip evaluation in this example
warmup_steps = math.ceil(len(train_data)*num_epochs/train_batch_size*0.1) #10% of train data for warm-up
logging.info("Warmup-steps: {}".format(warmup_steps))


# Train the model
model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=evaluator,
          epochs=num_epochs,
          evaluation_steps=1112,
          warmup_steps=warmup_steps,
          output_path=model_save_path)


##############################################################################
#
# Load the stored model and evaluate its performance on STS benchmark dataset
#
##############################################################################

model = SentenceTransformer(model_save_path)
test_data = SentencesDataset(examples=n2c2_reader.get_examples("test.tsv"), model=model)
test_dataloader = DataLoader(test_data, shuffle=False, batch_size=train_batch_size)
evaluator = LabelAccuracyEvaluator(test_dataloader)
model.evaluate(evaluator)
