# Imports lazy: no cargar training infrastructure al importar solo el modelo o la inferencia.
# Esto evita la cadena timm → torch._dynamo → circular import.

def __getattr__(name):
    if name == "CustomDistillationDataset":
        from utils.train.distillation_dataset import CustomDistillationDataset
        return CustomDistillationDataset
    if name == "train_model":
        from utils.train.trainer_distillation_knowledge import train_model
        return train_model
    if name == "get_augmentation_ade20k":
        from utils.models.copy_paste_augmentation import get_augmentation_ade20k
        return get_augmentation_ade20k
    if name == "train_model_adversarial":
        from utils.train.trainer_adversarial_finetune import train_model_adversarial
        return train_model_adversarial
    raise AttributeError(f"module 'utils' has no attribute {name!r}")
