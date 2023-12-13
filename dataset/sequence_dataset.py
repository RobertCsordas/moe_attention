import torch.utils.data
import framework


class SequenceDataset(torch.utils.data.Dataset):
    in_vocabulary: framework.data_structures.vocabulary.Vocabulary
    out_vocabulary: framework.data_structures.vocabulary.Vocabulary
