from src.modeling.modeling import (
    ClipBertForPreTraining,
    ClipBertForSequenceClassification,
    ClipBertForMultipleChoice,
    ClipBertForRegression,
    ClipBertForVideoTextRetrieval)
from src.modeling.grid_feat import GridFeatBackbone
from torch import nn
from src.datasets.data_utils import repeat_tensor_rows
from src.utils.load_save import load_state_dict_with_mismatch
import poptorch


class ClipBert(nn.Module):
    def __init__(self, config, input_format="BGR",
                 detectron2_model_cfg=None,
                 transformer_cls=ClipBertForPreTraining):
        super(ClipBert, self).__init__()
        self.config = config
        self.detectron2_model_cfg = detectron2_model_cfg
        assert detectron2_model_cfg is not None
        cnn_cls = GridFeatBackbone
        print(f"cnn_cls {cnn_cls}")
        self.cnn = cnn_cls(
            detectron2_model_cfg=detectron2_model_cfg,
            config=config, input_format=input_format)
        self.transformer = transformer_cls(config)
        self.retrieval = transformer_cls == ClipBertForVideoTextRetrieval

    def forward(self,
                n_examples_list,
                text_input_ids,
                visual_inputs,
                text_input_mask,
                mlm_labels=None,
                itm_labels=None):
        # used to make visual feature copies
        repeat_counts = n_examples_list[0]
        visual_features = self.cnn(visual_inputs)
        visual_inputs = repeat_tensor_rows(
            visual_features, repeat_counts)
        if self.retrieval:
            batch["sample_size"] = len(repeat_counts)  # batch size
        outputs = self.transformer(text_input_ids,
                                    visual_inputs,
                                    text_input_mask,
                                    mlm_labels,
                                    itm_labels)
        mlm_loss, itm_loss = 0, 0
        if mlm_labels is not None:
            mlm_loss = outputs["mlm_loss"].mean()
        if itm_labels is not None:
            itm_loss = outputs["itm_loss"].mean()
        if mlm_labels is not None and itm_labels is not None:
            loss = mlm_loss + itm_loss
        loss = poptorch.identity_loss(loss, reduction='none')
        return loss

    def load_separate_ckpt(self, cnn_weights_path=None, bert_weights_path=None):
        if cnn_weights_path:
            self.cnn.load_state_dict(cnn_weights_path)

        if bert_weights_path:
            load_state_dict_with_mismatch(self.transformer, bert_weights_path)

    def freeze_cnn_backbone(self):
        for n, p in self.cnn.feature.named_parameters():
            p.requires_grad = False
