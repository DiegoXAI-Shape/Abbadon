# =============================================================================
# Backward compatibility — este archivo re-exporta todo desde los módulos
# nuevos para que los imports existentes sigan funcionando sin cambios.
#
# Archivos nuevos:
#   - blocks.py         → conv3x3, BloqueResidual, UpSampling, AttentionGates, EncoderBlockT
#   - datasets.py       → CustomDS_Med, CustomDS
#   - daowa_maad.py     → Daowa_maad, TransformerDaowa_maad
#   - mendicant_bias.py → Mendicant_Biasv3
# =============================================================================

try:
    # Cuando se importa como paquete (e.g. from utils.models.utils_med import ...)
    from .blocks import conv3x3, BloqueResidual, UpSampling, AttentionGates, EncoderBlockT
    from .datasets import CustomDS_Med, CustomDS
    from .daowa_maad import Daowa_maad, TransformerDaowa_maad
    from .mendicant_bias import Mendicant_Biasv3
except ImportError:
    # Cuando se importa directo con sys.path (e.g. from utils_med import ...)
    from blocks import conv3x3, BloqueResidual, UpSampling, AttentionGates, EncoderBlockT
    from datasets import CustomDS_Med, CustomDS
    from daowa_maad import Daowa_maad, TransformerDaowa_maad
    from mendicant_bias import Mendicant_Biasv3