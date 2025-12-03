# Taks_T1 robot environment
from .taks_t1_mimic_config import TaksT1MimicCfg, TaksT1MimicCfgPPO
from .taks_t1_mimic import TaksT1Mimic
from .taks_t1_mimic_distill import TaksT1MimicDistill
from .taks_t1_mimic_distill_config import (
    TaksT1MimicPrivCfg, TaksT1MimicPrivCfgPPO, 
    TaksT1MimicStuCfg, TaksT1MimicStuCfgDAgger,
    TaksT1MimicStuRLCfg, TaksT1MimicStuRLCfgDAgger
)
from .taks_t1_mimic_future import TaksT1MimicFuture
from .taks_t1_mimic_future_config import TaksT1MimicStuFutureCfg, TaksT1MimicStuFutureCfgDAgger
