from .image_metrics import image_auroc
from .pixel_metrics import pixel_auroc
class Evaluator:
    def evaluate(self, image_scores, image_labels, pixel_scores, masks):
        return {"image_auroc": image_auroc(image_scores, image_labels), "pixel_auroc": pixel_auroc(pixel_scores, masks)}
