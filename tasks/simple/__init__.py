# from .low_rank_scan import LowRankScan
# from .simple_scan import SimpleScan
# from .pcfg_set import PCFGSet
# from .pcfg_transformer import PCFGTransformer
# from .cogs_transofrmer import COGSTransformer
# from .simple_scan_transformer import SimpleScanTransformer
# from .class_based_scan import ClassBasedScan
# from .configurable_encdec_scan import ConfigurableEncdecScan
# from .compositional_table_lookup import SimpleCTL
# from .ctl_trafo import SimpleCTLTrafo
# from .ctl_lstm_curriculum import CurriculumCTL
# from .ctl_trafo_curriculum import CurriculumCTLTrafo
# from . import language_model
# from .harnn_scan import HARNNScan
# from .harnn_serial_scan import HARNNSerialScan
# from .harnn_ctl import HarnnCTL
# from .scan_resplit_transformer import ScanResplitTransformer
# from .pcfg_transformer_orthogonal import PCFGTransformerOrthogonalReg
# from .cfq_transformer import CFQTransformer
# from .dm_math_transformer import DMMathTransformer
# from .listops_trafo import ListopsTransformer
# from .listops_trafo_curriculum import ListopsTransformerCurriculum
# from .ctl_lstm_classifier import CtlLstmClassifier
# from .ctl_lstm import CtlLstm
# from .ctl_trafo_classifier import CTLTrafoClassifier

from .. import task_db
task_db.register_files()

from . import language_model