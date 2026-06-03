from models.llava import Llava15Adapter


MODEL_REGISTRY = {
    "llava-1.5": Llava15Adapter,
}


def get_model_adapter(model_family: str):
    try:
        return MODEL_REGISTRY[model_family]()
    except KeyError as exc:
        supported = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(
            f"Unsupported model family: {model_family}. Supported families: {supported}."
        ) from exc
