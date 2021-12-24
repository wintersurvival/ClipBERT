import math
from src.modeling.modeling import (
    ClipBertForPreTraining,
    ClipBertForSequenceClassification,
    ClipBertForMultipleChoice,
    ClipBertForRegression,
    ClipBertForVideoTextRetrieval)
from src.modeling.grid_feat import GridFeatBackbone
import torch
from torch import nn
from src.datasets.data_utils import repeat_tensor_rows
from src.utils.load_save import load_state_dict_with_mismatch
import poptorch


class SerializedLinear(nn.Linear):
    """
    Exactly equivalent to `nn.Linear` layer, but with the matrix multiplication replaced with
    a serialized matrix multiplication: `poptorch.serializedMatMul`.
    The matrix multiplication is split into separate smaller multiplications, calculated one after the other,
    to reduce the memory requirements of the multiplication and its gradient calculation.

    Args:
        in_features: Size of each input sample
        out_features: Size of each output sample
        factor: Number of serialized multiplications. Must be a factor of
            the dimension to serialize on.
        bias: If set to False, the layer will not learn an additive bias.
            Default: True
        mode: Which dimension of the matmul to serialize on:
            for matrix A (m by n) multiplied by matrix B (n by p).
            * InputChannels: Split across the input channels (dimension m).
            * ReducingDim: Split across the reducing dimension (n).
            * OutputChannels: Split across the output channels (dimension p).
            * Disabled: Same as an ordinary matrix multiplication.
    """
    def __init__(self, in_features, out_features, factor, bias=True,
                 mode=poptorch.MatMulSerializationMode.OutputChannels):
        super().__init__(in_features, out_features, bias)
        self.mode = mode
        self.factor = factor

    def forward(self, x):
        output = poptorch.serializedMatMul(x, self.weight.t(), self.mode, self.factor)
        if self.bias is not None:
            output += self.bias
        return output


def outline_attribute(module: nn.Module, value: str):
    """Adds an attribute to a module. This attribute will be used
        when comparing operation equivalence in outlining. For example:

        layer1 = nn.Linear(...)
        layer2 = nn.Linear(...)
        layer3 = nn.Linear(...)
        layer4 = nn.Linear(...)
        outline_attribute(layer1, "A")
        outline_attribute(layer2, "A")
        outline_attribute(layer3, "B")

        The code for layer1 can be reused for layer2.
        But it can't be used for layer3 or layer4.
    """
    context = poptorch.Attribute(__outline={"layer": value})

    def enable(*args):
        context.__enter__()

    def disable(*args):
        context.__exit__(None, None, None)
    module.register_forward_pre_hook(enable)
    module.register_forward_hook(disable)


def recomputation_checkpoint(module: nn.Module):
    """Annotates the output of a module to be checkpointed instead of
        recomputed"""
    def recompute_outputs(module, inputs, outputs):
        if type(outputs) is tuple:
            return tuple(poptorch.recomputationCheckpoint(y) for y in outputs)
        else:
            return poptorch.recomputationCheckpoint(outputs)
    module.register_forward_hook(recompute_outputs)


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
        if transformer_cls == ClipBertForPreTraining:
            config.max_grid_col_position_embeddings = 12
            config.max_grid_row_position_embeddings = 12
            embedding_serialization_factor = 2
        self.cnn = cnn_cls(
            detectron2_model_cfg=detectron2_model_cfg,
            config=config, input_format=input_format)
        self.transformer = transformer_cls(config)
        if transformer_cls == ClipBertForPreTraining:
            if True: #self.config.embedding_serialization_factor > 1:
                serialized_decoder = SerializedLinear(self.config.hidden_size,
                                                  config.vocab_size,
                                                  embedding_serialization_factor,
                                                  bias=True,
                                                  mode=poptorch.MatMulSerializationMode.OutputChannels)
                serialized_decoder.load_state_dict(self.transformer.cls.predictions.decoder.state_dict())
                self.transformer.cls.predictions.decoder = serialized_decoder
            '''
            # partition to 8 ipu
            #self.cnn = poptorch.BeginBlock(self.cnn, f"cnn_backbone_stem", ipu_id=0)
            self.cnn.backbone = poptorch.BeginBlock(self.cnn.backbone, f"cnn.backbone", ipu_id=0)
            self.cnn.backbone.res4[3] = poptorch.BeginBlock(self.cnn.backbone.res4[3], f"cnn.backbone.res4[3]", ipu_id=1)
            self.cnn.grid_encoder = poptorch.BeginBlock(self.cnn.grid_encoder, f"cnn.backbone.grid_encoder", ipu_id=2)
            self.transformer.bert.embeddings = poptorch.BeginBlock(self.transformer.bert.embeddings, f"transformer.bert.embeddings", ipu_id=3)
            outline_attribute(self.transformer.bert.embeddings.LayerNorm, "embeddings")
            for index, layer in enumerate(self.transformer.bert.encoder.layer):
                recomputation_checkpoint(layer)
            self.transformer.bert.encoder.layer[0] = poptorch.BeginBlock(self.transformer.bert.encoder.layer[0], f"transformer.bert.encoder.layer[0]", ipu_id=4)
            self.transformer.bert.encoder.layer[3] = poptorch.BeginBlock(self.transformer.bert.encoder.layer[3], f"transformer.bert.encoder.layer[3]", ipu_id=5)
            self.transformer.bert.encoder.layer[6] = poptorch.BeginBlock(self.transformer.bert.encoder.layer[6], f"transformer.bert.encoder.layer[6]", ipu_id=6)
            self.transformer.bert.encoder.layer[9] = poptorch.BeginBlock(self.transformer.bert.encoder.layer[9], f"transformer.bert.encoder.layer[9]", ipu_id=7)
            self.transformer.bert.pooler = poptorch.BeginBlock(self.transformer.bert.pooler, "Pooler", ipu_id=3)
            self.transformer.cls = poptorch.BeginBlock(self.transformer.cls, "Classifier", ipu_id=3)
            '''
            # partition to 4 ipu
            #self.cnn = poptorch.BeginBlock(self.cnn, f"cnn_backbone_stem", ipu_id=0)
            #self.cnn.backbone = poptorch.BeginBlock(self.cnn.backbone, f"cnn.backbone", ipu_id=0)
            #self.cnn.grid_encoder = poptorch.BeginBlock(self.cnn.grid_encoder, f"cnn.backbone.grid_encoder", ipu_id=2)
            self.transformer.bert.embeddings = poptorch.BeginBlock(self.transformer.bert.embeddings, f"transformer.bert.embeddings", ipu_id=0)
            outline_attribute(self.transformer.bert.embeddings.LayerNorm, "embeddings")
            self.cnn.backbone.res3[1] = poptorch.BeginBlock(self.cnn.backbone.res3[1], f"cnn.backbone.res3[1]", ipu_id=1)
            for index, layer in enumerate(self.transformer.bert.encoder.layer):
                recomputation_checkpoint(layer)
            self.transformer.bert.encoder.layer[0] = poptorch.BeginBlock(self.transformer.bert.encoder.layer[0], f"transformer.bert.encoder.layer[0]", ipu_id=2)
            self.transformer.bert.encoder.layer[6] = poptorch.BeginBlock(self.transformer.bert.encoder.layer[6], f"transformer.bert.encoder.layer[6]", ipu_id=3)
            self.transformer.bert.pooler = poptorch.BeginBlock(self.transformer.bert.pooler, "Pooler", ipu_id=0)
            self.transformer.cls = poptorch.BeginBlock(self.transformer.cls, "Classifier", ipu_id=0)
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

        #mlm
        if mlm_labels is not None:
            mlm_loss = outputs["mlm_loss"].mean()
            mlm_mask = mlm_labels != 100  # (B, Lt)  -100 is the ignored label for cross entropy
            n_mlm_tokens = mlm_mask.sum().item()
            n_mlm_corrects = (
                    outputs["mlm_scores"][mlm_mask].max(
                        dim=-1)[1] == mlm_labels[mlm_mask]).sum().item()
            if n_mlm_tokens != 0:
                mlm_acc = torch.tensor(float(n_mlm_corrects / n_mlm_tokens))

        # itm
        if itm_labels is not None:
            itm_loss = outputs["itm_loss"].mean()
            n_itm_ex = len(itm_labels)
            n_itm_corrects = (
                    outputs["itm_scores"].max(
                        dim=-1)[1] == outputs["itm_labels"]).sum().item()
            if n_itm_ex != 0:
                itm_acc = torch.tensor(float(n_itm_corrects / n_itm_ex))

        if mlm_labels is not None and itm_labels is not None:
            loss = mlm_loss + itm_loss
        loss = poptorch.identity_loss(loss, reduction='none')
        if mlm_labels is not None and itm_labels is not None:
            return loss, mlm_loss, mlm_acc, itm_loss, itm_acc
        else:
            return loss

    def load_separate_ckpt(self, cnn_weights_path=None, bert_weights_path=None):
        if cnn_weights_path:
            self.cnn.load_state_dict(cnn_weights_path)

        if bert_weights_path:
            load_state_dict_with_mismatch(self.transformer, bert_weights_path)

    def freeze_cnn_backbone(self):
        for n, p in self.cnn.feature.named_parameters():
            p.requires_grad = False
