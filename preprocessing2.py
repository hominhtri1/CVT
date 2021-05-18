import random

from base import utils, configure, embeddings
from task_specific.sentence_level import sentence_level_data


def main(data_dir='/content/data'):
  random.seed(0)

  utils.log("BUILDING WORD VOCABULARY/EMBEDDINGS")
  for pretrained in ['glove.6B.100d.txt']:
    config = configure.Config(data_dir=data_dir,
                              for_preprocessing=True,
                              pretrained_embeddings=pretrained,
                              word_embedding_size=100)
    embeddings.PretrainedEmbeddingLoader(config).build()

  utils.log("WRITING LABEL MAPPINGS")
  for task_name in ["senclass"]:
    config = configure.Config(data_dir=data_dir,
                              for_preprocessing=True)
    loader = sentence_level_data.SentenceClassificationDataLoader(config, task_name)
    utils.log("WRITING LABEL MAPPING FOR", task_name.upper())
    utils.log(" ", len(loader.label_mapping), "classes")
    utils.write_cpickle(loader.label_mapping,
                        loader.label_mapping_path)


if __name__ == '__main__':
  main()
